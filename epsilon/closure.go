// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package epsilon

import (
	"fmt"
	"math"
)

// Closure-dispatch interpreter (spike).
//
// Each function body is compiled once into a []closureInstr — one closure per
// WebAssembly instruction, with operands captured at compile time. Execution is
// an indirect-call dispatch loop instead of the bytecode switch:
//
//	for ip < len(code) { ip = code[ip](ctx) }
//
// This mirrors the design in PlanetScale's "Faster interpreters in Go" post.
// Compared to the switch in executeInstruction it removes runtime operand
// decode (frame.next) and replaces the switch with a single indirect branch,
// at the cost of an un-inlined call per instruction. The closures reuse the
// exact same value-stack operations as the switch, so the only difference under
// benchmark is the dispatch mechanism itself.
//
// This is a spike: only a subset of opcodes is supported (integer ALU, locals,
// consts, and structured control flow — enough for arithmetic loops). Any
// function containing an unsupported opcode fails to compile and falls back to
// the switch interpreter (a JIT-style deopt).

// Sentinel return values for a closureInstr. Non-negative returns are absolute
// next-instruction indices (branch targets).
const (
	closureNext = -1 // advance to ip+1
	closureHalt = -2 // stop; ctx.trap holds the error (or nil)
)

type closureInstr func(c *closureCtx) int

// operandWordCount returns the number of operand words that follow the opcode
// word, where operandStart indexes the first operand word (opcodeIndex+1). It
// mirrors the encoding produced by parser.readCode and is used to walk a body
// instruction by instruction.
func operandWordCount(op opcode, body []uint64, operandStart int) int {
	switch op {
	case block, loop, ifOp, i32Const, i64Const, f32Const, f64Const,
		br, brIf, call, localGet, localSet, localTee, globalGet, globalSet,
		tableGet, tableSet, memoryFill, dataDrop, elemDrop, tableGrow, tableSize,
		tableFill, refNull, refFunc, memorySize, memoryGrow,
		i8x16ExtractLaneS, i8x16ExtractLaneU, i16x8ExtractLaneS, i16x8ExtractLaneU,
		i32x4ExtractLane, i64x2ExtractLane, f32x4ExtractLane, f64x2ExtractLane,
		i8x16ReplaceLane, i16x8ReplaceLane, i32x4ReplaceLane, i64x2ReplaceLane,
		f32x4ReplaceLane, f64x2ReplaceLane:
		return 1
	case callIndirect, memoryInit, memoryCopy, tableInit, tableCopy, v128Const:
		return 2
	case i32Load, i64Load, f32Load, f64Load, i32Load8S, i32Load8U, i32Load16S,
		i32Load16U, i64Load8S, i64Load8U, i64Load16S, i64Load16U, i64Load32S,
		i64Load32U, i32Store, i64Store, f32Store, f64Store, i32Store8, i32Store16,
		i64Store8, i64Store16, i64Store32,
		v128Load, v128Load32Zero, v128Load64Zero, v128Load8Splat, v128Load16Splat,
		v128Load32Splat, v128Load64Splat, v128Load8x8S, v128Load8x8U,
		v128Load16x4S, v128Load16x4U, v128Load32x2S, v128Load32x2U, v128Store:
		return 3
	case v128Load8Lane, v128Load16Lane, v128Load32Lane, v128Load64Lane,
		v128Store8Lane, v128Store16Lane, v128Store32Lane, v128Store64Lane:
		return 4
	case i8x16Shuffle:
		return 16
	case brTable:
		// 1 count word + N label words + 1 default word.
		return int(body[operandStart]) + 2
	case selectT:
		// 1 count word + N type words.
		return int(body[operandStart]) + 1
	default:
		return 0
	}
}

// closureControl mirrors controlFrame but stores its branch target as an
// instruction index rather than a bytecode pc.
type closureControl struct {
	isLoop      bool
	targetIp    int
	stackHeight uint32
	arity       uint32
}

type closureCtx struct {
	vm     *vm
	locals []value
	module *ModuleInstance // for stateful ops: memory / global / table / call
	ctrl   []closureControl
	trap   error
}

// brToLabel mirrors vm.brToLabel in instruction-index space.
func (c *closureCtx) brToLabel(labelIndex int) int {
	targetIndex := len(c.ctrl) - labelIndex - 1
	target := c.ctrl[targetIndex]
	c.ctrl = c.ctrl[:targetIndex]
	c.vm.stack.unwind(target.stackHeight, target.arity)
	if target.isLoop {
		c.ctrl = append(c.ctrl, target)
	}
	return target.targetIp
}

// newClosureCtx builds the execution context for the current call frame and
// seeds its base control frame (the function body itself): a branch to the
// outermost label, a return, or fall-through unwinds to here.
// newClosureCtx returns the execution context for the current call frame. Both
// the context and its control stack are drawn from depth-indexed caches so a
// wasm call allocates nothing (critical for call-heavy code); beyond the
// preallocated depth they fall back to the heap. The returned pointer is stable
// for the duration of the call (the caches never reallocate).
func (vm *vm) newClosureCtx(code []closureInstr, resultArity uint32) *closureCtx {
	callDepth := len(vm.callStack) - 1
	frame := &vm.callStack[callDepth]

	var c *closureCtx
	var ctrl []closureControl
	if callDepth < vm.config.CallStackPreallocationSize {
		c = &vm.closureCtxCache[callDepth]
		base := callDepth * controlStackCacheSlotSize
		ctrl = vm.closureCtrlCache[base : base : base+controlStackCacheSlotSize]
	} else {
		c = &closureCtx{}
	}
	*c = closureCtx{vm: vm, locals: frame.locals, module: frame.module}
	c.ctrl = append(ctrl, closureControl{
		targetIp:    len(code),
		arity:       resultArity,
		stackHeight: vm.stack.size(),
	})
	return c
}

func (vm *vm) runClosureLoop(code []closureInstr, resultArity uint32) error {
	c := vm.newClosureCtx(code, resultArity)
	ip := 0
	for ip < len(code) {
		switch n := code[ip](c); n {
		case closureNext:
			ip++
		case closureHalt:
			vm.callStack = vm.callStack[:len(vm.callStack)-1]
			return c.trap
		default:
			ip = n
		}
	}
	vm.callStack = vm.callStack[:len(vm.callStack)-1]
	return nil
}

// runClosureLoopWithFuel mirrors runClosureLoop but decrements fuel once per
// dispatched instruction. Kept separate so the no-fuel loop pays nothing.
func (vm *vm) runClosureLoopWithFuel(code []closureInstr, resultArity uint32) error {
	c := vm.newClosureCtx(code, resultArity)
	ip := 0
	for ip < len(code) {
		if vm.fuel == 0 {
			vm.callStack = vm.callStack[:len(vm.callStack)-1]
			return errFuelExhausted
		}
		vm.fuel--
		switch n := code[ip](c); n {
		case closureNext:
			ip++
		case closureHalt:
			vm.callStack = vm.callStack[:len(vm.callStack)-1]
			return c.trap
		default:
			ip = n
		}
	}
	vm.callStack = vm.callStack[:len(vm.callStack)-1]
	return nil
}

// ensureCompiled lazily compiles a wasm function to closures on first use and
// caches the result on the function. It is idempotent and covers both module
// functions and the synthetic functions built by invokeExpression.
func (vm *vm) ensureCompiled(fn *wasmFunction) {
	if fn.closureCompiled {
		return
	}
	fn.closureCompiled = true
	if code, ok := vm.compileClosures(&fn.code, fn.module); ok {
		fn.closures = code
		fn.closuresOK = true
		// The bytecode and jump caches are only compile inputs; the closures are
		// now the sole runtime representation. Release them.
		fn.code.body = nil
		fn.code.jumpCache = nil
		fn.code.jumpElseCache = nil
	}
}

// compileClosures compiles a function body into a closure program. It returns
// (nil, false) if any opcode is unsupported, signalling the caller to use the
// switch interpreter instead.
func (vm *vm) compileClosures(fn *function, module *ModuleInstance) ([]closureInstr, bool) {
	body := fn.body

	// First pass: assign an instruction index to every bytecode boundary so
	// branch targets (which the parser recorded as bytecode pcs) can be
	// translated into instruction indices.
	pcToIp := make(map[uint32]int, len(body))
	starts := make([]int, 0, len(body))
	for i := 0; i < len(body); {
		pcToIp[uint32(i)] = len(starts)
		starts = append(starts, i)
		op := opcode(body[i])
		i += 1 + operandWordCount(op, body, i+1)
	}
	pcToIp[uint32(len(body))] = len(starts)

	// Second pass: emit one closure per instruction.
	code := make([]closureInstr, 0, len(starts))
	for _, pc := range starts {
		instr, ok := vm.compileInstr(fn, module, pc, pcToIp)
		if !ok {
			return nil, false
		}
		code = append(code, instr)
	}
	return code, true
}

// compileInstr builds the closure for the instruction at bytecode index pc.
func (vm *vm) compileInstr(
	fn *function, module *ModuleInstance, pc int, pcToIp map[uint32]int,
) (closureInstr, bool) {
	body := fn.body
	op := opcode(body[pc])

	switch op {
	case nop:
		return func(c *closureCtx) int { return closureNext }, true
	case unreachable:
		return func(c *closureCtx) int { c.trap = errUnreachable; return closureHalt }, true
	case drop:
		return func(c *closureCtx) int { c.vm.stack.drop(); return closureNext }, true
	case selectOp:
		return func(c *closureCtx) int { c.vm.handleSelect(); return closureNext }, true
	case selectT:
		// The type-vector operand is for validation only; semantics match select.
		return func(c *closureCtx) int { c.vm.handleSelect(); return closureNext }, true

	case localGet:
		idx := body[pc+1]
		return func(c *closureCtx) int { c.vm.stack.push(c.locals[idx]); return closureNext }, true
	case localSet:
		idx := body[pc+1]
		return func(c *closureCtx) int { c.locals[idx] = c.vm.stack.pop(); return closureNext }, true
	case localTee:
		idx := body[pc+1]
		return func(c *closureCtx) int {
			c.locals[idx] = c.vm.stack.data[len(c.vm.stack.data)-1]
			return closureNext
		}, true

	case i32Const:
		v := int32(body[pc+1])
		return func(c *closureCtx) int { c.vm.stack.pushInt32(v); return closureNext }, true
	case i64Const:
		v := int64(body[pc+1])
		return func(c *closureCtx) int { c.vm.stack.pushInt64(v); return closureNext }, true

	// --- control flow ------------------------------------------------------
	case block:
		blockType := int32(body[pc+1])
		afterEndIp := pcToIp[fn.jumpCache[uint32(pc+2)]]
		inputCount := vm.getInputCount(module, blockType)
		outputCount := vm.getOutputCount(module, blockType)
		return func(c *closureCtx) int {
			c.ctrl = append(c.ctrl, closureControl{
				targetIp:    afterEndIp,
				arity:       outputCount,
				stackHeight: c.vm.stack.size() - inputCount,
			})
			return closureNext
		}, true
	case loop:
		blockType := int32(body[pc+1])
		bodyIp := pcToIp[uint32(pc+2)]
		inputCount := vm.getInputCount(module, blockType)
		return func(c *closureCtx) int {
			c.ctrl = append(c.ctrl, closureControl{
				isLoop:      true,
				targetIp:    bodyIp,
				arity:       inputCount,
				stackHeight: c.vm.stack.size() - inputCount,
			})
			return closureNext
		}, true
	case ifOp:
		blockType := int32(body[pc+1])
		afterEndIp := pcToIp[fn.jumpCache[uint32(pc+2)]]
		elseIp := pcToIp[fn.jumpElseCache[uint32(pc+2)]]
		inputCount := vm.getInputCount(module, blockType)
		outputCount := vm.getOutputCount(module, blockType)
		return func(c *closureCtx) int {
			condition := c.vm.stack.popInt32()
			c.ctrl = append(c.ctrl, closureControl{
				targetIp:    afterEndIp,
				arity:       outputCount,
				stackHeight: c.vm.stack.size() - inputCount,
			})
			if condition == 0 {
				return elseIp
			}
			return closureNext
		}, true
	case elseOp:
		return func(c *closureCtx) int {
			target := c.ctrl[len(c.ctrl)-1].targetIp
			c.ctrl = c.ctrl[:len(c.ctrl)-1]
			return target
		}, true
	case end:
		return func(c *closureCtx) int {
			if len(c.ctrl) > 0 {
				c.ctrl = c.ctrl[:len(c.ctrl)-1]
			}
			return closureNext
		}, true
	case br:
		label := int(body[pc+1])
		return func(c *closureCtx) int { return c.brToLabel(label) }, true
	case brIf:
		label := int(body[pc+1])
		return func(c *closureCtx) int {
			if c.vm.stack.popInt32() != 0 {
				return c.brToLabel(label)
			}
			return closureNext
		}, true
	case brTable:
		count := uint32(body[pc+1])
		labels := make([]uint32, count)
		for i := range labels {
			labels[i] = uint32(body[pc+2+i])
		}
		defaultLabel := uint32(body[pc+2+int(count)])
		return func(c *closureCtx) int {
			index := uint32(c.vm.stack.popInt32())
			if index < count {
				return c.brToLabel(int(labels[index]))
			}
			return c.brToLabel(int(defaultLabel))
		}, true
	case returnOp:
		return func(c *closureCtx) int { return c.brToLabel(len(c.ctrl) - 1) }, true

	case call:
		funcLocalIndex := body[pc+1]
		return func(c *closureCtx) int {
			function := c.vm.store.funcs[c.module.funcAddrs[funcLocalIndex]]
			if err := c.vm.invokeFunction(function); err != nil {
				c.trap = err
				return closureHalt
			}
			return closureNext
		}, true
	case callIndirect:
		typeIndex := body[pc+1]
		tableLocalIndex := body[pc+2]
		return func(c *closureCtx) int {
			expectedType := c.module.types[typeIndex]
			table := c.vm.getTable(c.module, tableLocalIndex)
			elementIndex := c.vm.stack.popInt32()
			tableElement, err := table.Get(elementIndex)
			if err != nil {
				c.trap = err
				return closureHalt
			}
			if tableElement == NullReference {
				c.trap = fmt.Errorf("uninitialized element %d", elementIndex)
				return closureHalt
			}
			function := c.vm.store.funcs[tableElement]
			if !function.GetType().Equal(expectedType) {
				c.trap = errIndirectCallTypeMismatch
				return closureHalt
			}
			if err := c.vm.invokeFunction(function); err != nil {
				c.trap = err
				return closureHalt
			}
			return closureNext
		}, true

	case f32Const:
		v := math.Float32frombits(uint32(body[pc+1]))
		return func(c *closureCtx) int { c.vm.stack.pushFloat32(v); return closureNext }, true
	case f64Const:
		v := math.Float64frombits(body[pc+1])
		return func(c *closureCtx) int { c.vm.stack.pushFloat64(v); return closureNext }, true

	case globalGet:
		idx := body[pc+1]
		return func(c *closureCtx) int {
			c.vm.stack.push(c.vm.getGlobal(c.module, idx).value)
			return closureNext
		}, true
	case globalSet:
		idx := body[pc+1]
		return func(c *closureCtx) int {
			c.vm.getGlobal(c.module, idx).value = c.vm.stack.pop()
			return closureNext
		}, true

	case tableGet:
		idx := body[pc+1]
		return safe(func(c *closureCtx) error {
			table := c.vm.getTable(c.module, idx)
			element, err := table.Get(c.vm.stack.popInt32())
			if err != nil {
				return err
			}
			c.vm.stack.pushInt32(element)
			return nil
		})
	case tableSet:
		idx := body[pc+1]
		return safe(func(c *closureCtx) error {
			table := c.vm.getTable(c.module, idx)
			reference := c.vm.stack.popInt32()
			return table.Set(c.vm.stack.popInt32(), reference)
		})

	case memorySize:
		idx := body[pc+1]
		return func(c *closureCtx) int {
			c.vm.stack.pushInt32(c.vm.getMemory(c.module, idx).Size())
			return closureNext
		}, true
	case memoryGrow:
		idx := body[pc+1]
		return func(c *closureCtx) int {
			memory := c.vm.getMemory(c.module, idx)
			c.vm.stack.pushInt32(memory.Grow(c.vm.stack.popInt32()))
			return closureNext
		}, true

	case refNull:
		return func(c *closureCtx) int { c.vm.stack.pushInt32(NullReference); return closureNext }, true
	case refIsNull:
		return func(c *closureCtx) int {
			c.vm.stack.pushInt32(boolToInt32(c.vm.stack.popInt32() == NullReference))
			return closureNext
		}, true
	case refFunc:
		funcLocalIndex := body[pc+1]
		return func(c *closureCtx) int {
			c.vm.stack.pushInt32(int32(c.module.funcAddrs[funcLocalIndex]))
			return closureNext
		}, true

	case memoryInit:
		dataIdx, memIdx := body[pc+1], body[pc+2]
		return safe(func(c *closureCtx) error {
			data := c.vm.getData(c.module, dataIdx)
			memory := c.vm.getMemory(c.module, memIdx)
			n, s, d := c.vm.stack.pop3Int32()
			return memory.Init(uint32(n), uint32(s), uint32(d), data.content)
		})
	case dataDrop:
		dataIdx := body[pc+1]
		return func(c *closureCtx) int {
			c.vm.getData(c.module, dataIdx).content = nil
			return closureNext
		}, true
	case memoryCopy:
		destIdx, srcIdx := body[pc+1], body[pc+2]
		return safe(func(c *closureCtx) error {
			destMemory := c.vm.getMemory(c.module, destIdx)
			srcMemory := c.vm.getMemory(c.module, srcIdx)
			n, s, d := c.vm.stack.pop3Int32()
			return srcMemory.Copy(destMemory, uint32(n), uint32(s), uint32(d))
		})
	case memoryFill:
		memIdx := body[pc+1]
		return safe(func(c *closureCtx) error {
			memory := c.vm.getMemory(c.module, memIdx)
			n, val, offset := c.vm.stack.pop3Int32()
			return memory.Fill(uint32(n), uint32(offset), byte(val))
		})
	case tableInit:
		elemIdx, tableIdx := body[pc+1], body[pc+2]
		return safe(func(c *closureCtx) error {
			element := c.vm.getElement(c.module, elemIdx)
			table := c.vm.getTable(c.module, tableIdx)
			n, s, d := c.vm.stack.pop3Int32()
			return table.Init(n, d, s, element.functionIndexes)
		})
	case elemDrop:
		elemIdx := body[pc+1]
		return func(c *closureCtx) int {
			c.vm.getElement(c.module, elemIdx).functionIndexes = nil
			return closureNext
		}, true
	case tableCopy:
		destIdx, srcIdx := body[pc+1], body[pc+2]
		return safe(func(c *closureCtx) error {
			destTable := c.vm.getTable(c.module, destIdx)
			srcTable := c.vm.getTable(c.module, srcIdx)
			n, s, d := c.vm.stack.pop3Int32()
			return srcTable.Copy(destTable, n, s, d)
		})
	case tableGrow:
		tableIdx := body[pc+1]
		return func(c *closureCtx) int {
			table := c.vm.getTable(c.module, tableIdx)
			n := c.vm.stack.popInt32()
			val := c.vm.stack.popInt32()
			c.vm.stack.pushInt32(table.Grow(n, val))
			return closureNext
		}, true
	case tableSize:
		tableIdx := body[pc+1]
		return func(c *closureCtx) int {
			c.vm.stack.pushInt32(int32(c.vm.getTable(c.module, tableIdx).Size()))
			return closureNext
		}, true
	case tableFill:
		tableIdx := body[pc+1]
		return safe(func(c *closureCtx) error {
			table := c.vm.getTable(c.module, tableIdx)
			n, val, i := c.vm.stack.pop3Int32()
			return table.Fill(n, i, val)
		})

	// Integer loads. memarg is [align(unused), memIndex, offset] at pc+1..pc+3.
	case i32Load:
		return closureLoad(body, pc, vm.stack.pushInt32, (*Memory).LoadUint32, uint32ToInt32), true
	case i64Load:
		return closureLoad(body, pc, vm.stack.pushInt64, (*Memory).LoadUint64, uint64ToInt64), true
	case f32Load:
		return closureLoad(body, pc, vm.stack.pushFloat32, (*Memory).LoadUint32, math.Float32frombits), true
	case f64Load:
		return closureLoad(body, pc, vm.stack.pushFloat64, (*Memory).LoadUint64, math.Float64frombits), true
	case i32Load8S:
		return closureLoad(body, pc, vm.stack.pushInt32, (*Memory).LoadByte, signExtend8To32), true
	case i32Load8U:
		return closureLoad(body, pc, vm.stack.pushInt32, (*Memory).LoadByte, zeroExtend8To32), true
	case i32Load16S:
		return closureLoad(body, pc, vm.stack.pushInt32, (*Memory).LoadUint16, signExtend16To32), true
	case i32Load16U:
		return closureLoad(body, pc, vm.stack.pushInt32, (*Memory).LoadUint16, zeroExtend16To32), true
	case i64Load8S:
		return closureLoad(body, pc, vm.stack.pushInt64, (*Memory).LoadByte, signExtend8To64), true
	case i64Load8U:
		return closureLoad(body, pc, vm.stack.pushInt64, (*Memory).LoadByte, zeroExtend8To64), true
	case i64Load16S:
		return closureLoad(body, pc, vm.stack.pushInt64, (*Memory).LoadUint16, signExtend16To64), true
	case i64Load16U:
		return closureLoad(body, pc, vm.stack.pushInt64, (*Memory).LoadUint16, zeroExtend16To64), true
	case i64Load32S:
		return closureLoad(body, pc, vm.stack.pushInt64, (*Memory).LoadUint32, signExtend32To64), true
	case i64Load32U:
		return closureLoad(body, pc, vm.stack.pushInt64, (*Memory).LoadUint32, zeroExtend32To64), true

	// Integer stores. The value is popped first, then the address index.
	case i32Store:
		return closureStore(body, pc, vm.stack.popInt32, func(m *Memory, o, i uint32, v int32) error { return m.StoreUint32(o, i, uint32(v)) }), true
	case i64Store:
		return closureStore(body, pc, vm.stack.popInt64, func(m *Memory, o, i uint32, v int64) error { return m.StoreUint64(o, i, uint64(v)) }), true
	case f32Store:
		return closureStore(body, pc, vm.stack.popFloat32, func(m *Memory, o, i uint32, v float32) error { return m.StoreUint32(o, i, math.Float32bits(v)) }), true
	case f64Store:
		return closureStore(body, pc, vm.stack.popFloat64, func(m *Memory, o, i uint32, v float64) error { return m.StoreUint64(o, i, math.Float64bits(v)) }), true
	case i32Store8:
		return closureStore(body, pc, vm.stack.popInt32, func(m *Memory, o, i uint32, v int32) error { return m.StoreByte(o, i, byte(v)) }), true
	case i32Store16:
		return closureStore(body, pc, vm.stack.popInt32, func(m *Memory, o, i uint32, v int32) error { return m.StoreUint16(o, i, uint16(v)) }), true
	case i64Store8:
		return closureStore(body, pc, vm.stack.popInt64, func(m *Memory, o, i uint32, v int64) error { return m.StoreByte(o, i, byte(v)) }), true
	case i64Store16:
		return closureStore(body, pc, vm.stack.popInt64, func(m *Memory, o, i uint32, v int64) error { return m.StoreUint16(o, i, uint16(v)) }), true
	case i64Store32:
		return closureStore(body, pc, vm.stack.popInt64, func(m *Memory, o, i uint32, v int64) error { return m.StoreUint32(o, i, uint32(v)) }), true

	// --- SIMD with operands ---
	case v128Load:
		return closureLoad(body, pc, vm.stack.pushV128, (*Memory).LoadV128, identityV128), true
	case v128Store:
		return closureStore(body, pc, vm.stack.popV128, (*Memory).StoreV128), true
	case v128Load8x8S:
		return closureLoadV128FromBytes(body, pc, simdV128Load8x8S, 8), true
	case v128Load8x8U:
		return closureLoadV128FromBytes(body, pc, simdV128Load8x8U, 8), true
	case v128Load16x4S:
		return closureLoadV128FromBytes(body, pc, simdV128Load16x4S, 8), true
	case v128Load16x4U:
		return closureLoadV128FromBytes(body, pc, simdV128Load16x4U, 8), true
	case v128Load32x2S:
		return closureLoadV128FromBytes(body, pc, simdV128Load32x2S, 8), true
	case v128Load32x2U:
		return closureLoadV128FromBytes(body, pc, simdV128Load32x2U, 8), true
	case v128Load8Splat:
		return closureLoadV128FromBytes(body, pc, simdI8x16SplatFromBytes, 1), true
	case v128Load16Splat:
		return closureLoadV128FromBytes(body, pc, simdI16x8SplatFromBytes, 2), true
	case v128Load32Splat:
		return closureLoadV128FromBytes(body, pc, simdI32x4SplatFromBytes, 4), true
	case v128Load64Splat:
		return closureLoadV128FromBytes(body, pc, simdI64x2SplatFromBytes, 8), true
	case v128Load32Zero:
		return closureLoadV128FromBytes(body, pc, simdV128Load32Zero, 4), true
	case v128Load64Zero:
		return closureLoadV128FromBytes(body, pc, simdV128Load64Zero, 8), true
	case v128Load8Lane:
		return closureSimdLoadLane(body, pc, 8), true
	case v128Load16Lane:
		return closureSimdLoadLane(body, pc, 16), true
	case v128Load32Lane:
		return closureSimdLoadLane(body, pc, 32), true
	case v128Load64Lane:
		return closureSimdLoadLane(body, pc, 64), true
	case v128Store8Lane:
		return closureSimdStoreLane(body, pc, 8), true
	case v128Store16Lane:
		return closureSimdStoreLane(body, pc, 16), true
	case v128Store32Lane:
		return closureSimdStoreLane(body, pc, 32), true
	case v128Store64Lane:
		return closureSimdStoreLane(body, pc, 64), true
	case v128Const:
		v := V128Value{Low: body[pc+1], High: body[pc+2]}
		return func(c *closureCtx) int { c.vm.stack.pushV128(v); return closureNext }, true
	case i8x16Shuffle:
		var lanes [16]byte
		for i := range lanes {
			lanes[i] = byte(body[pc+1+i])
		}
		return func(c *closureCtx) int {
			v2 := c.vm.stack.popV128()
			v1 := c.vm.stack.popV128()
			c.vm.stack.pushV128(simdI8x16Shuffle(v1, v2,
				lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7],
				lanes[8], lanes[9], lanes[10], lanes[11], lanes[12], lanes[13], lanes[14], lanes[15]))
			return closureNext
		}, true
	case i8x16ExtractLaneS:
		return closureExtractLane(body, pc, vm.stack.pushInt32, simdI8x16ExtractLaneS), true
	case i8x16ExtractLaneU:
		return closureExtractLane(body, pc, vm.stack.pushInt32, simdI8x16ExtractLaneU), true
	case i16x8ExtractLaneS:
		return closureExtractLane(body, pc, vm.stack.pushInt32, simdI16x8ExtractLaneS), true
	case i16x8ExtractLaneU:
		return closureExtractLane(body, pc, vm.stack.pushInt32, simdI16x8ExtractLaneU), true
	case i32x4ExtractLane:
		return closureExtractLane(body, pc, vm.stack.pushInt32, simdI32x4ExtractLane), true
	case i64x2ExtractLane:
		return closureExtractLane(body, pc, vm.stack.pushInt64, simdI64x2ExtractLane), true
	case f32x4ExtractLane:
		return closureExtractLane(body, pc, vm.stack.pushFloat32, simdF32x4ExtractLane), true
	case f64x2ExtractLane:
		return closureExtractLane(body, pc, vm.stack.pushFloat64, simdF64x2ExtractLane), true
	case i8x16ReplaceLane:
		return closureReplaceLane(body, pc, vm.stack.popInt32, simdI8x16ReplaceLane), true
	case i16x8ReplaceLane:
		return closureReplaceLane(body, pc, vm.stack.popInt32, simdI16x8ReplaceLane), true
	case i32x4ReplaceLane:
		return closureReplaceLane(body, pc, vm.stack.popInt32, simdI32x4ReplaceLane), true
	case i64x2ReplaceLane:
		return closureReplaceLane(body, pc, vm.stack.popInt64, simdI64x2ReplaceLane), true
	case f32x4ReplaceLane:
		return closureReplaceLane(body, pc, vm.stack.popFloat32, simdF32x4ReplaceLane), true
	case f64x2ReplaceLane:
		return closureReplaceLane(body, pc, vm.stack.popFloat64, simdF64x2ReplaceLane), true

	default:
		return vm.compileNumericInstr(op)
	}
}

// simple wraps a body that always advances to the next instruction.
func simple(body func(c *closureCtx)) (closureInstr, bool) {
	return func(c *closureCtx) int { body(c); return closureNext }, true
}

// safe wraps a body that may trap; a non-nil error halts with the trap set.
func safe(body func(c *closureCtx) error) (closureInstr, bool) {
	return func(c *closureCtx) int {
		if err := body(c); err != nil {
			c.trap = err
			return closureHalt
		}
		return closureNext
	}, true
}

// closureLoad builds a memory-load closure, capturing the memory index and
// offset from the memarg at body[pc+1..pc+3] (align is unused).
func closureLoad[T any, R any](
	body []uint64, pc int,
	push func(R),
	load func(*Memory, uint32, uint32) (T, error),
	convert func(T) R,
) closureInstr {
	memIdx := body[pc+2]
	offset := uint32(body[pc+3])
	return func(c *closureCtx) int {
		memory := c.vm.getMemory(c.module, memIdx)
		index := uint32(c.vm.stack.popInt32())
		v, err := load(memory, offset, index)
		if err != nil {
			c.trap = err
			return closureHalt
		}
		push(convert(v))
		return closureNext
	}
}

// closureStore builds a memory-store closure. The value is popped first, then
// the address index, matching the switch interpreter.
func closureStore[T any](
	body []uint64, pc int,
	pop func() T,
	store func(*Memory, uint32, uint32, T) error,
) closureInstr {
	memIdx := body[pc+2]
	offset := uint32(body[pc+3])
	return func(c *closureCtx) int {
		val := pop()
		memory := c.vm.getMemory(c.module, memIdx)
		index := uint32(c.vm.stack.popInt32())
		if err := store(memory, offset, index, val); err != nil {
			c.trap = err
			return closureHalt
		}
		return closureNext
	}
}

// closureLoadV128FromBytes mirrors vm.handleLoadV128FromBytes.
func closureLoadV128FromBytes(
	body []uint64, pc int, fromBytes func([]byte) V128Value, sizeBytes uint32,
) closureInstr {
	memIdx := body[pc+2]
	offset := uint32(body[pc+3])
	return func(c *closureCtx) int {
		memory := c.vm.getMemory(c.module, memIdx)
		index := c.vm.stack.popInt32()
		data, err := memory.Get(offset, uint32(index), sizeBytes)
		if err != nil {
			c.trap = err
			return closureHalt
		}
		c.vm.stack.pushV128(fromBytes(data))
		return closureNext
	}
}

// closureSimdLoadLane mirrors vm.handleSimdLoadLane.
func closureSimdLoadLane(body []uint64, pc int, laneSize uint32) closureInstr {
	memIdx := body[pc+2]
	offset := uint32(body[pc+3])
	laneIndex := uint32(body[pc+4])
	return func(c *closureCtx) int {
		memory := c.vm.getMemory(c.module, memIdx)
		v := c.vm.stack.popV128()
		index := c.vm.stack.popInt32()
		laneValue, err := memory.Get(offset, uint32(index), laneSize/8)
		if err != nil {
			c.trap = err
			return closureHalt
		}
		c.vm.stack.pushV128(simdLoadLane(v, laneIndex, laneValue))
		return closureNext
	}
}

// closureSimdStoreLane mirrors vm.handleSimdStoreLane.
func closureSimdStoreLane(body []uint64, pc int, laneSize uint32) closureInstr {
	memIdx := body[pc+2]
	offset := uint32(body[pc+3])
	laneIndex := uint32(body[pc+4])
	return func(c *closureCtx) int {
		memory := c.vm.getMemory(c.module, memIdx)
		v := c.vm.stack.popV128()
		index := c.vm.stack.popInt32()

		lanesPerUint64 := 64 / laneSize
		shift := (laneIndex % lanesPerUint64) * laneSize
		var val uint64
		if laneIndex < lanesPerUint64 {
			val = v.Low >> shift
		} else {
			val = v.High >> shift
		}

		var err error
		switch laneSize {
		case 8:
			err = memory.StoreByte(offset, uint32(index), byte(val))
		case 16:
			err = memory.StoreUint16(offset, uint32(index), uint16(val))
		case 32:
			err = memory.StoreUint32(offset, uint32(index), uint32(val))
		case 64:
			err = memory.StoreUint64(offset, uint32(index), val)
		}
		if err != nil {
			c.trap = err
			return closureHalt
		}
		return closureNext
	}
}

// closureExtractLane mirrors handleSimdExtractLane; the lane index is at pc+1.
func closureExtractLane[R wasmNumber](
	body []uint64, pc int, push func(R), op func(V128Value, uint32) R,
) closureInstr {
	laneIndex := uint32(body[pc+1])
	return func(c *closureCtx) int {
		push(op(c.vm.stack.popV128(), laneIndex))
		return closureNext
	}
}

// closureReplaceLane mirrors handleSimdReplaceLane; the lane index is at pc+1.
func closureReplaceLane[T wasmNumber](
	body []uint64, pc int, pop func() T, op func(V128Value, uint32, T) V128Value,
) closureInstr {
	laneIndex := uint32(body[pc+1])
	return func(c *closureCtx) int {
		laneValue := pop()
		vector := c.vm.stack.popV128()
		c.vm.stack.pushV128(op(vector, laneIndex, laneValue))
		return closureNext
	}
}

// compileNumericInstr handles the operand-free numeric, comparison, and
// conversion opcodes. Bodies mirror the corresponding switch cases exactly.
func (vm *vm) compileNumericInstr(op opcode) (closureInstr, bool) {
	switch op {
	// --- i32 comparisons ---
	case i32Eqz:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(boolToInt32(c.vm.stack.popInt32() == 0)) })
	case i32Eq:
		return cmp32(equal[int32])
	case i32Ne:
		return cmp32(notEqual[int32])
	case i32LtS:
		return cmp32(lessThan[int32])
	case i32LtU:
		return cmp32(lessThanU32)
	case i32GtS:
		return cmp32(greaterThan[int32])
	case i32GtU:
		return cmp32(greaterThanU32)
	case i32LeS:
		return cmp32(lessOrEqual[int32])
	case i32LeU:
		return cmp32(lessOrEqualU32)
	case i32GeS:
		return cmp32(greaterOrEqual[int32])
	case i32GeU:
		return cmp32(greaterOrEqualU32)

	// --- i64 comparisons ---
	case i64Eqz:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(boolToInt32(c.vm.stack.popInt64() == 0)) })
	case i64Eq:
		return cmp64(equal[int64])
	case i64Ne:
		return cmp64(notEqual[int64])
	case i64LtS:
		return cmp64(lessThan[int64])
	case i64LtU:
		return cmp64(lessThanU64)
	case i64GtS:
		return cmp64(greaterThan[int64])
	case i64GtU:
		return cmp64(greaterThanU64)
	case i64LeS:
		return cmp64(lessOrEqual[int64])
	case i64LeU:
		return cmp64(lessOrEqualU64)
	case i64GeS:
		return cmp64(greaterOrEqual[int64])
	case i64GeU:
		return cmp64(greaterOrEqualU64)

	// --- float comparisons ---
	case f32Eq:
		return cmpf32(equal[float32])
	case f32Ne:
		return cmpf32(notEqual[float32])
	case f32Lt:
		return cmpf32(lessThan[float32])
	case f32Gt:
		return cmpf32(greaterThan[float32])
	case f32Le:
		return cmpf32(lessOrEqual[float32])
	case f32Ge:
		return cmpf32(greaterOrEqual[float32])
	case f64Eq:
		return cmpf64(equal[float64])
	case f64Ne:
		return cmpf64(notEqual[float64])
	case f64Lt:
		return cmpf64(lessThan[float64])
	case f64Gt:
		return cmpf64(greaterThan[float64])
	case f64Le:
		return cmpf64(lessOrEqual[float64])
	case f64Ge:
		return cmpf64(greaterOrEqual[float64])

	// --- i32 unary / binary ---
	case i32Clz:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(clz32(c.vm.stack.popInt32())) })
	case i32Ctz:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(ctz32(c.vm.stack.popInt32())) })
	case i32Popcnt:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(popcnt32(c.vm.stack.popInt32())) })
	case i32Add:
		return alu32(func(a, b int32) int32 { return a + b })
	case i32Sub:
		return alu32(func(a, b int32) int32 { return a - b })
	case i32Mul:
		return alu32(func(a, b int32) int32 { return a * b })
	case i32DivS:
		return safe(func(c *closureCtx) error { return c.vm.handleBinarySafeInt32(divS32) })
	case i32DivU:
		return safe(func(c *closureCtx) error { return c.vm.handleBinarySafeInt32(divU32) })
	case i32RemS:
		return safe(func(c *closureCtx) error { return c.vm.handleBinarySafeInt32(remS32) })
	case i32RemU:
		return safe(func(c *closureCtx) error { return c.vm.handleBinarySafeInt32(remU32) })
	case i32And:
		return alu32(func(a, b int32) int32 { return a & b })
	case i32Or:
		return alu32(func(a, b int32) int32 { return a | b })
	case i32Xor:
		return alu32(func(a, b int32) int32 { return a ^ b })
	case i32Shl:
		return alu32(func(a, b int32) int32 { return a << (uint32(b) % 32) })
	case i32ShrS:
		return alu32(func(a, b int32) int32 { return a >> (uint32(b) % 32) })
	case i32ShrU:
		return alu32(shrU32)
	case i32Rotl:
		return alu32(rotl32)
	case i32Rotr:
		return alu32(rotr32)

	// --- i64 unary / binary ---
	case i64Clz:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(clz64(c.vm.stack.popInt64())) })
	case i64Ctz:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(ctz64(c.vm.stack.popInt64())) })
	case i64Popcnt:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(popcnt64(c.vm.stack.popInt64())) })
	case i64Add:
		return alu64(func(a, b int64) int64 { return a + b })
	case i64Sub:
		return alu64(func(a, b int64) int64 { return a - b })
	case i64Mul:
		return alu64(func(a, b int64) int64 { return a * b })
	case i64DivS:
		return safe(func(c *closureCtx) error { return c.vm.handleBinarySafeInt64(divS64) })
	case i64DivU:
		return safe(func(c *closureCtx) error { return c.vm.handleBinarySafeInt64(divU64) })
	case i64RemS:
		return safe(func(c *closureCtx) error { return c.vm.handleBinarySafeInt64(remS64) })
	case i64RemU:
		return safe(func(c *closureCtx) error { return c.vm.handleBinarySafeInt64(remU64) })
	case i64And:
		return alu64(func(a, b int64) int64 { return a & b })
	case i64Or:
		return alu64(func(a, b int64) int64 { return a | b })
	case i64Xor:
		return alu64(func(a, b int64) int64 { return a ^ b })
	case i64Shl:
		return alu64(shl64)
	case i64ShrS:
		return alu64(shrS64)
	case i64ShrU:
		return alu64(shrU64)
	case i64Rotl:
		return alu64(rotl64)
	case i64Rotr:
		return alu64(rotr64)

	// --- f32 unary / binary ---
	case f32Abs:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(abs(c.vm.stack.popFloat32())) })
	case f32Neg:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(-c.vm.stack.popFloat32()) })
	case f32Ceil:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(ceil(c.vm.stack.popFloat32())) })
	case f32Floor:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(floor(c.vm.stack.popFloat32())) })
	case f32Trunc:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(trunc(c.vm.stack.popFloat32())) })
	case f32Nearest:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(nearest(c.vm.stack.popFloat32())) })
	case f32Sqrt:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(sqrt(c.vm.stack.popFloat32())) })
	case f32Add:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat32(add[float32]) })
	case f32Sub:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat32(sub[float32]) })
	case f32Mul:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat32(mul[float32]) })
	case f32Div:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat32(div[float32]) })
	case f32Min:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat32(wasmMin[float32]) })
	case f32Max:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat32(wasmMax[float32]) })
	case f32Copysign:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat32(copysign[float32]) })

	// --- f64 unary / binary ---
	case f64Abs:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(abs(c.vm.stack.popFloat64())) })
	case f64Neg:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(-c.vm.stack.popFloat64()) })
	case f64Ceil:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(ceil(c.vm.stack.popFloat64())) })
	case f64Floor:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(floor(c.vm.stack.popFloat64())) })
	case f64Trunc:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(trunc(c.vm.stack.popFloat64())) })
	case f64Nearest:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(nearest(c.vm.stack.popFloat64())) })
	case f64Sqrt:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(sqrt(c.vm.stack.popFloat64())) })
	case f64Add:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat64(add[float64]) })
	case f64Sub:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat64(sub[float64]) })
	case f64Mul:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat64(mul[float64]) })
	case f64Div:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat64(div[float64]) })
	case f64Min:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat64(wasmMin[float64]) })
	case f64Max:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat64(wasmMax[float64]) })
	case f64Copysign:
		return simple(func(c *closureCtx) { c.vm.handleBinaryFloat64(copysign[float64]) })

	// --- conversions ---
	case i32WrapI64:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(wrapI64ToI32(c.vm.stack.popInt64())) })
	case i32TruncF32S:
		return safe(func(c *closureCtx) error { return c.vm.handleUnarySafeFloat32(truncF32SToI32) })
	case i32TruncF32U:
		return safe(func(c *closureCtx) error { return c.vm.handleUnarySafeFloat32(truncF32UToI32) })
	case i32TruncF64S:
		return safe(func(c *closureCtx) error { return c.vm.handleUnarySafeFloat64(truncF64SToI32) })
	case i32TruncF64U:
		return safe(func(c *closureCtx) error { return c.vm.handleUnarySafeFloat64(truncF64UToI32) })
	case i64ExtendI32S:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(extendI32SToI64(c.vm.stack.popInt32())) })
	case i64ExtendI32U:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(extendI32UToI64(c.vm.stack.popInt32())) })
	case i64TruncF32S:
		return safe(func(c *closureCtx) error { return c.vm.handleTruncFloat32Int64(truncF32SToI64) })
	case i64TruncF32U:
		return safe(func(c *closureCtx) error { return c.vm.handleTruncFloat32Int64(truncF32UToI64) })
	case i64TruncF64S:
		return safe(func(c *closureCtx) error { return c.vm.handleTruncFloat64Int64(truncF64SToI64) })
	case i64TruncF64U:
		return safe(func(c *closureCtx) error { return c.vm.handleTruncFloat64Int64(truncF64UToI64) })
	case f32ConvertI32S:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(convertI32SToF32(c.vm.stack.popInt32())) })
	case f32ConvertI32U:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(convertI32UToF32(c.vm.stack.popInt32())) })
	case f32ConvertI64S:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(convertI64SToF32(c.vm.stack.popInt64())) })
	case f32ConvertI64U:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(convertI64UToF32(c.vm.stack.popInt64())) })
	case f32DemoteF64:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(demoteF64ToF32(c.vm.stack.popFloat64())) })
	case f64ConvertI32S:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(convertI32SToF64(c.vm.stack.popInt32())) })
	case f64ConvertI32U:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(convertI32UToF64(c.vm.stack.popInt32())) })
	case f64ConvertI64S:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(convertI64SToF64(c.vm.stack.popInt64())) })
	case f64ConvertI64U:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(convertI64UToF64(c.vm.stack.popInt64())) })
	case f64PromoteF32:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(promoteF32ToF64(c.vm.stack.popFloat32())) })
	case i32ReinterpretF32:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(reinterpretF32ToI32(c.vm.stack.popFloat32())) })
	case i64ReinterpretF64:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(reinterpretF64ToI64(c.vm.stack.popFloat64())) })
	case f32ReinterpretI32:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat32(reinterpretI32ToF32(c.vm.stack.popInt32())) })
	case f64ReinterpretI64:
		return simple(func(c *closureCtx) { c.vm.stack.pushFloat64(reinterpretI64ToF64(c.vm.stack.popInt64())) })
	case i32Extend8S:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(extend8STo32(c.vm.stack.popInt32())) })
	case i32Extend16S:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(extend16STo32(c.vm.stack.popInt32())) })
	case i64Extend8S:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(extend8STo64(c.vm.stack.popInt64())) })
	case i64Extend16S:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(extend16STo64(c.vm.stack.popInt64())) })
	case i64Extend32S:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(extend32STo64(c.vm.stack.popInt64())) })
	case i32TruncSatF32S:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(truncSatF32SToI32(c.vm.stack.popFloat32())) })
	case i32TruncSatF32U:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(truncSatF32UToI32(c.vm.stack.popFloat32())) })
	case i32TruncSatF64S:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(truncSatF64SToI32(c.vm.stack.popFloat64())) })
	case i32TruncSatF64U:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(truncSatF64UToI32(c.vm.stack.popFloat64())) })
	case i64TruncSatF32S:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(truncSatF32SToI64(c.vm.stack.popFloat32())) })
	case i64TruncSatF32U:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(truncSatF32UToI64(c.vm.stack.popFloat32())) })
	case i64TruncSatF64S:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(truncSatF64SToI64(c.vm.stack.popFloat64())) })
	case i64TruncSatF64U:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt64(truncSatF64UToI64(c.vm.stack.popFloat64())) })

	default:
		return vm.compileSimdInstr(op)
	}
}

// v128Unary builds a closure for an operand-free V128 -> V128 instruction.
func v128Unary(op func(V128Value) V128Value) (closureInstr, bool) {
	return simple(func(c *closureCtx) { c.vm.stack.pushV128(op(c.vm.stack.popV128())) })
}

// v128Binary builds a closure for an operand-free (V128, V128) -> V128 instruction.
func v128Binary(op func(a, b V128Value) V128Value) (closureInstr, bool) {
	return simple(func(c *closureCtx) { c.vm.handleBinaryV128(op) })
}

// compileSimdInstr handles the operand-free SIMD opcodes. Bodies mirror the
// corresponding switch cases exactly.
func (vm *vm) compileSimdInstr(op opcode) (closureInstr, bool) {
	switch op {
	// splats
	case i8x16Splat:
		return simple(func(c *closureCtx) { c.vm.stack.pushV128(simdI8x16Splat(c.vm.stack.popInt32())) })
	case i16x8Splat:
		return simple(func(c *closureCtx) { c.vm.stack.pushV128(simdI16x8Splat(c.vm.stack.popInt32())) })
	case i32x4Splat:
		return simple(func(c *closureCtx) { c.vm.stack.pushV128(simdI32x4Splat(c.vm.stack.popInt32())) })
	case i64x2Splat:
		return simple(func(c *closureCtx) { c.vm.stack.pushV128(simdI64x2Splat(c.vm.stack.popInt64())) })
	case f32x4Splat:
		return simple(func(c *closureCtx) { c.vm.stack.pushV128(simdF32x4Splat(c.vm.stack.popFloat32())) })
	case f64x2Splat:
		return simple(func(c *closureCtx) { c.vm.stack.pushV128(simdF64x2Splat(c.vm.stack.popFloat64())) })
	case i8x16Swizzle:
		return v128Binary(simdI8x16Swizzle)

	// bitwise / boolean
	case v128Not:
		return v128Unary(simdV128Not)
	case v128And:
		return v128Binary(simdV128And)
	case v128Andnot:
		return v128Binary(simdV128Andnot)
	case v128Or:
		return v128Binary(simdV128Or)
	case v128Xor:
		return v128Binary(simdV128Xor)
	case v128Bitselect:
		return simple(func(c *closureCtx) { c.vm.handleSimdTernary(simdV128Bitselect) })
	case v128AnyTrue:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(boolToInt32(simdV128AnyTrue(c.vm.stack.popV128()))) })

	// comparisons
	case i8x16Eq:
		return v128Binary(simdI8x16Eq)
	case i8x16Ne:
		return v128Binary(simdI8x16Ne)
	case i8x16LtS:
		return v128Binary(simdI8x16LtS)
	case i8x16LtU:
		return v128Binary(simdI8x16LtU)
	case i8x16GtS:
		return v128Binary(simdI8x16GtS)
	case i8x16GtU:
		return v128Binary(simdI8x16GtU)
	case i8x16LeS:
		return v128Binary(simdI8x16LeS)
	case i8x16LeU:
		return v128Binary(simdI8x16LeU)
	case i8x16GeS:
		return v128Binary(simdI8x16GeS)
	case i8x16GeU:
		return v128Binary(simdI8x16GeU)
	case i16x8Eq:
		return v128Binary(simdI16x8Eq)
	case i16x8Ne:
		return v128Binary(simdI16x8Ne)
	case i16x8LtS:
		return v128Binary(simdI16x8LtS)
	case i16x8LtU:
		return v128Binary(simdI16x8LtU)
	case i16x8GtS:
		return v128Binary(simdI16x8GtS)
	case i16x8GtU:
		return v128Binary(simdI16x8GtU)
	case i16x8LeS:
		return v128Binary(simdI16x8LeS)
	case i16x8LeU:
		return v128Binary(simdI16x8LeU)
	case i16x8GeS:
		return v128Binary(simdI16x8GeS)
	case i16x8GeU:
		return v128Binary(simdI16x8GeU)
	case i32x4Eq:
		return v128Binary(simdI32x4Eq)
	case i32x4Ne:
		return v128Binary(simdI32x4Ne)
	case i32x4LtS:
		return v128Binary(simdI32x4LtS)
	case i32x4LtU:
		return v128Binary(simdI32x4LtU)
	case i32x4GtS:
		return v128Binary(simdI32x4GtS)
	case i32x4GtU:
		return v128Binary(simdI32x4GtU)
	case i32x4LeS:
		return v128Binary(simdI32x4LeS)
	case i32x4LeU:
		return v128Binary(simdI32x4LeU)
	case i32x4GeS:
		return v128Binary(simdI32x4GeS)
	case i32x4GeU:
		return v128Binary(simdI32x4GeU)
	case f32x4Eq:
		return v128Binary(simdF32x4Eq)
	case f32x4Ne:
		return v128Binary(simdF32x4Ne)
	case f32x4Lt:
		return v128Binary(simdF32x4Lt)
	case f32x4Gt:
		return v128Binary(simdF32x4Gt)
	case f32x4Le:
		return v128Binary(simdF32x4Le)
	case f32x4Ge:
		return v128Binary(simdF32x4Ge)
	case f64x2Eq:
		return v128Binary(simdF64x2Eq)
	case f64x2Ne:
		return v128Binary(simdF64x2Ne)
	case f64x2Lt:
		return v128Binary(simdF64x2Lt)
	case f64x2Gt:
		return v128Binary(simdF64x2Gt)
	case f64x2Le:
		return v128Binary(simdF64x2Le)
	case f64x2Ge:
		return v128Binary(simdF64x2Ge)
	case i64x2Eq:
		return v128Binary(simdI64x2Eq)
	case i64x2Ne:
		return v128Binary(simdI64x2Ne)
	case i64x2LtS:
		return v128Binary(simdI64x2LtS)
	case i64x2GtS:
		return v128Binary(simdI64x2GtS)
	case i64x2LeS:
		return v128Binary(simdI64x2LeS)
	case i64x2GeS:
		return v128Binary(simdI64x2GeS)

	// shifts (pop count + vector)
	case i8x16Shl:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI8x16Shl) })
	case i8x16ShrS:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI8x16ShrS) })
	case i8x16ShrU:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI8x16ShrU) })
	case i16x8Shl:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI16x8Shl) })
	case i16x8ShrS:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI16x8ShrS) })
	case i16x8ShrU:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI16x8ShrU) })
	case i32x4Shl:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI32x4Shl) })
	case i32x4ShrS:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI32x4ShrS) })
	case i32x4ShrU:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI32x4ShrU) })
	case i64x2Shl:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI64x2Shl) })
	case i64x2ShrS:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI64x2ShrS) })
	case i64x2ShrU:
		return simple(func(c *closureCtx) { c.vm.handleSimdShift(simdI64x2ShrU) })

	// i8x16 unary / arithmetic
	case i8x16Abs:
		return v128Unary(simdI8x16Abs)
	case i8x16Neg:
		return v128Unary(simdI8x16Neg)
	case i8x16Popcnt:
		return v128Unary(simdI8x16Popcnt)
	case i8x16AllTrue:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(boolToInt32(simdI8x16AllTrue(c.vm.stack.popV128()))) })
	case i8x16Bitmask:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(simdI8x16Bitmask(c.vm.stack.popV128())) })
	case i8x16NarrowI16x8S:
		return v128Binary(simdI8x16NarrowI16x8S)
	case i8x16NarrowI16x8U:
		return v128Binary(simdI8x16NarrowI16x8U)
	case i8x16Add:
		return v128Binary(simdI8x16Add)
	case i8x16AddSatS:
		return v128Binary(simdI8x16AddSatS)
	case i8x16AddSatU:
		return v128Binary(simdI8x16AddSatU)
	case i8x16Sub:
		return v128Binary(simdI8x16Sub)
	case i8x16SubSatS:
		return v128Binary(simdI8x16SubSatS)
	case i8x16SubSatU:
		return v128Binary(simdI8x16SubSatU)
	case i8x16MinS:
		return v128Binary(simdI8x16MinS)
	case i8x16MinU:
		return v128Binary(simdI8x16MinU)
	case i8x16MaxS:
		return v128Binary(simdI8x16MaxS)
	case i8x16MaxU:
		return v128Binary(simdI8x16MaxU)
	case i8x16AvgrU:
		return v128Binary(simdI8x16AvgrU)

	// i16x8
	case i16x8ExtaddPairwiseI8x16S:
		return v128Unary(simdI16x8ExtaddPairwiseI8x16S)
	case i16x8ExtaddPairwiseI8x16U:
		return v128Unary(simdI16x8ExtaddPairwiseI8x16U)
	case i16x8Abs:
		return v128Unary(simdI16x8Abs)
	case i16x8Neg:
		return v128Unary(simdI16x8Neg)
	case i16x8Q15mulrSatS:
		return v128Binary(simdI16x8Q15mulrSatS)
	case i16x8AllTrue:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(boolToInt32(simdI16x8AllTrue(c.vm.stack.popV128()))) })
	case i16x8Bitmask:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(simdI16x8Bitmask(c.vm.stack.popV128())) })
	case i16x8NarrowI32x4S:
		return v128Binary(simdI16x8NarrowI32x4S)
	case i16x8NarrowI32x4U:
		return v128Binary(simdI16x8NarrowI32x4U)
	case i16x8ExtendLowI8x16S:
		return v128Unary(simdI16x8ExtendLowI8x16S)
	case i16x8ExtendHighI8x16S:
		return v128Unary(simdI16x8ExtendHighI8x16S)
	case i16x8ExtendLowI8x16U:
		return v128Unary(simdI16x8ExtendLowI8x16U)
	case i16x8ExtendHighI8x16U:
		return v128Unary(simdI16x8ExtendHighI8x16U)
	case i16x8Add:
		return v128Binary(simdI16x8Add)
	case i16x8AddSatS:
		return v128Binary(simdI16x8AddSatS)
	case i16x8AddSatU:
		return v128Binary(simdI16x8AddSatU)
	case i16x8Sub:
		return v128Binary(simdI16x8Sub)
	case i16x8SubSatS:
		return v128Binary(simdI16x8SubSatS)
	case i16x8SubSatU:
		return v128Binary(simdI16x8SubSatU)
	case i16x8Mul:
		return v128Binary(simdI16x8Mul)
	case i16x8MinS:
		return v128Binary(simdI16x8MinS)
	case i16x8MinU:
		return v128Binary(simdI16x8MinU)
	case i16x8MaxS:
		return v128Binary(simdI16x8MaxS)
	case i16x8MaxU:
		return v128Binary(simdI16x8MaxU)
	case i16x8AvgrU:
		return v128Binary(simdI16x8AvgrU)
	case i16x8ExtmulLowI8x16S:
		return v128Binary(simdI16x8ExtmulLowI8x16S)
	case i16x8ExtmulHighI8x16S:
		return v128Binary(simdI16x8ExtmulHighI8x16S)
	case i16x8ExtmulLowI8x16U:
		return v128Binary(simdI16x8ExtmulLowI8x16U)
	case i16x8ExtmulHighI8x16U:
		return v128Binary(simdI16x8ExtmulHighI8x16U)

	// i32x4
	case i32x4ExtaddPairwiseI16x8S:
		return v128Unary(simdI32x4ExtaddPairwiseI16x8S)
	case i32x4ExtaddPairwiseI16x8U:
		return v128Unary(simdI32x4ExtaddPairwiseI16x8U)
	case i32x4Abs:
		return v128Unary(simdI32x4Abs)
	case i32x4Neg:
		return v128Unary(simdI32x4Neg)
	case i32x4AllTrue:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(boolToInt32(simdI32x4AllTrue(c.vm.stack.popV128()))) })
	case i32x4Bitmask:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(simdI32x4Bitmask(c.vm.stack.popV128())) })
	case i32x4ExtendLowI16x8S:
		return v128Unary(simdI32x4ExtendLowI16x8S)
	case i32x4ExtendHighI16x8S:
		return v128Unary(simdI32x4ExtendHighI16x8S)
	case i32x4ExtendLowI16x8U:
		return v128Unary(simdI32x4ExtendLowI16x8U)
	case i32x4ExtendHighI16x8U:
		return v128Unary(simdI32x4ExtendHighI16x8U)
	case i32x4Add:
		return v128Binary(simdI32x4Add)
	case i32x4Sub:
		return v128Binary(simdI32x4Sub)
	case i32x4Mul:
		return v128Binary(simdI32x4Mul)
	case i32x4MinS:
		return v128Binary(simdI32x4MinS)
	case i32x4MinU:
		return v128Binary(simdI32x4MinU)
	case i32x4MaxS:
		return v128Binary(simdI32x4MaxS)
	case i32x4MaxU:
		return v128Binary(simdI32x4MaxU)
	case i32x4DotI16x8S:
		return v128Binary(simdI32x4DotI16x8S)
	case i32x4ExtmulLowI16x8S:
		return v128Binary(simdI32x4ExtmulLowI16x8S)
	case i32x4ExtmulHighI16x8S:
		return v128Binary(simdI32x4ExtmulHighI16x8S)
	case i32x4ExtmulLowI16x8U:
		return v128Binary(simdI32x4ExtmulLowI16x8U)
	case i32x4ExtmulHighI16x8U:
		return v128Binary(simdI32x4ExtmulHighI16x8U)

	// i64x2
	case i64x2Abs:
		return v128Unary(simdI64x2Abs)
	case i64x2Neg:
		return v128Unary(simdI64x2Neg)
	case i64x2AllTrue:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(boolToInt32(simdI64x2AllTrue(c.vm.stack.popV128()))) })
	case i64x2Bitmask:
		return simple(func(c *closureCtx) { c.vm.stack.pushInt32(simdI64x2Bitmask(c.vm.stack.popV128())) })
	case i64x2ExtendLowI32x4S:
		return v128Unary(simdI64x2ExtendLowI32x4S)
	case i64x2ExtendHighI32x4S:
		return v128Unary(simdI64x2ExtendHighI32x4S)
	case i64x2ExtendLowI32x4U:
		return v128Unary(simdI64x2ExtendLowI32x4U)
	case i64x2ExtendHighI32x4U:
		return v128Unary(simdI64x2ExtendHighI32x4U)
	case i64x2Add:
		return v128Binary(simdI64x2Add)
	case i64x2Sub:
		return v128Binary(simdI64x2Sub)
	case i64x2Mul:
		return v128Binary(simdI64x2Mul)
	case i64x2ExtmulLowI32x4S:
		return v128Binary(simdI64x2ExtmulLowI32x4S)
	case i64x2ExtmulHighI32x4S:
		return v128Binary(simdI64x2ExtmulHighI32x4S)
	case i64x2ExtmulLowI32x4U:
		return v128Binary(simdI64x2ExtmulLowI32x4U)
	case i64x2ExtmulHighI32x4U:
		return v128Binary(simdI64x2ExtmulHighI32x4U)

	// f32x4 / f64x2
	case f32x4Ceil:
		return v128Unary(simdF32x4Ceil)
	case f32x4Floor:
		return v128Unary(simdF32x4Floor)
	case f32x4Trunc:
		return v128Unary(simdF32x4Trunc)
	case f32x4Nearest:
		return v128Unary(simdF32x4Nearest)
	case f32x4Abs:
		return v128Unary(simdF32x4Abs)
	case f32x4Neg:
		return v128Unary(simdF32x4Neg)
	case f32x4Sqrt:
		return v128Unary(simdF32x4Sqrt)
	case f32x4Add:
		return v128Binary(simdF32x4Add)
	case f32x4Sub:
		return v128Binary(simdF32x4Sub)
	case f32x4Mul:
		return v128Binary(simdF32x4Mul)
	case f32x4Div:
		return v128Binary(simdF32x4Div)
	case f32x4Min:
		return v128Binary(simdF32x4Min)
	case f32x4Max:
		return v128Binary(simdF32x4Max)
	case f32x4Pmin:
		return v128Binary(simdF32x4Pmin)
	case f32x4Pmax:
		return v128Binary(simdF32x4Pmax)
	case f64x2Ceil:
		return v128Unary(simdF64x2Ceil)
	case f64x2Floor:
		return v128Unary(simdF64x2Floor)
	case f64x2Trunc:
		return v128Unary(simdF64x2Trunc)
	case f64x2Nearest:
		return v128Unary(simdF64x2Nearest)
	case f64x2Abs:
		return v128Unary(simdF64x2Abs)
	case f64x2Neg:
		return v128Unary(simdF64x2Neg)
	case f64x2Sqrt:
		return v128Unary(simdF64x2Sqrt)
	case f64x2Add:
		return v128Binary(simdF64x2Add)
	case f64x2Sub:
		return v128Binary(simdF64x2Sub)
	case f64x2Mul:
		return v128Binary(simdF64x2Mul)
	case f64x2Div:
		return v128Binary(simdF64x2Div)
	case f64x2Min:
		return v128Binary(simdF64x2Min)
	case f64x2Max:
		return v128Binary(simdF64x2Max)
	case f64x2Pmin:
		return v128Binary(simdF64x2Pmin)
	case f64x2Pmax:
		return v128Binary(simdF64x2Pmax)

	// conversions
	case f32x4DemoteF64x2Zero:
		return v128Unary(simdF32x4DemoteF64x2Zero)
	case f64x2PromoteLowF32x4:
		return v128Unary(simdF64x2PromoteLowF32x4)
	case i32x4TruncSatF32x4S:
		return v128Unary(simdI32x4TruncSatF32x4S)
	case i32x4TruncSatF32x4U:
		return v128Unary(simdI32x4TruncSatF32x4U)
	case f32x4ConvertI32x4S:
		return v128Unary(simdF32x4ConvertI32x4S)
	case f32x4ConvertI32x4U:
		return v128Unary(simdF32x4ConvertI32x4U)
	case i32x4TruncSatF64x2SZero:
		return v128Unary(simdI32x4TruncSatF64x2SZero)
	case i32x4TruncSatF64x2UZero:
		return v128Unary(simdI32x4TruncSatF64x2UZero)
	case f64x2ConvertLowI32x4S:
		return v128Unary(simdF64x2ConvertLowI32x4S)
	case f64x2ConvertLowI32x4U:
		return v128Unary(simdF64x2ConvertLowI32x4U)

	default:
		return nil, false
	}
}

func cmp32(op func(a, b int32) bool) (closureInstr, bool) {
	return simple(func(c *closureCtx) { c.vm.handleBinaryBoolInt32(op) })
}

func cmp64(op func(a, b int64) bool) (closureInstr, bool) {
	return simple(func(c *closureCtx) { c.vm.handleBinaryBoolInt64(op) })
}

func cmpf32(op func(a, b float32) bool) (closureInstr, bool) {
	return simple(func(c *closureCtx) { c.vm.handleBinaryBoolFloat32(op) })
}

func cmpf64(op func(a, b float64) bool) (closureInstr, bool) {
	return simple(func(c *closureCtx) { c.vm.handleBinaryBoolFloat64(op) })
}

func alu64(op func(a, b int64) int64) (closureInstr, bool) {
	return simple(func(c *closureCtx) {
		b := c.vm.stack.popInt64()
		data := c.vm.stack.data
		last := len(data) - 1
		data[last] = i64(op(data[last].int64(), b))
	})
}

func alu32(op func(a, b int32) int32) (closureInstr, bool) {
	return func(c *closureCtx) int {
		b := c.vm.stack.popInt32()
		data := c.vm.stack.data
		last := len(data) - 1
		data[last] = i32(op(data[last].int32(), b))
		return closureNext
	}, true
}
