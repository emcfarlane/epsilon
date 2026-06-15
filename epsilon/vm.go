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
	"errors"
	"fmt"
	"math"
	"math/bits"
)

var (
	errUnreachable              = errors.New("unreachable")
	errCallStackExhausted       = errors.New("call stack exhausted")
	errFuelExhausted            = errors.New("fuel exhausted")
	errUnknownFunctionType      = errors.New("unknown function type")
	errIndirectCallTypeMismatch = errors.New("indirect call type mismatch")
	errHostResultCountMismatch  = errors.New("host func result count mismatch")
)

const (
	controlStackCacheSlotSize = 14 // Control stack slot size per call frame.
	localsCacheSlotSize       = 12 // Locals slot size per call frame.
)

// store represents all global state that can be manipulated by the vm. It
// consists of the runtime representation of all instances of functions, tables,
// memories, globals, element segments, and data segments that have been
// allocated during the vm life time.
type store struct {
	funcs    []FunctionInstance
	tables   []*Table
	memories []*Memory
	globals  []*Global
	elements []elementInstance
	datas    []dataInstance
}

// elementInstance is the runtime representation of an element segment.
// https://webassembly.github.io/spec/core/exec/runtime.html#element-instances.
// functionIndexes holds store-resolved function references; it is nil when the
// segment has been dropped (by elem.drop, or implicitly for active/declarative
// segments after instantiation).
type elementInstance struct {
	functionIndexes []int32
}

// dataInstance is the runtime representation of a data segment.
// https://webassembly.github.io/spec/core/exec/runtime.html#data-instances.
// content is nil when the segment has been dropped (by data.drop, or implicitly
// for active segments after instantiation).
type dataInstance struct {
	content []byte
}

type callFrame struct {
	controlStack []controlFrame
	locals       []value
	module       *ModuleInstance
	trap         error
}

// controlFrame represents a block of code that can be branched to.
type controlFrame struct {
	isLoop      bool
	targetIp    int32
	stackHeight uint32
	arity       uint32
}

// halt is returned by a handler to stop the run loop; callFrame.trap then holds
// the error (nil for a normal return). Read through an unsigned comparison in the
// loop, -1 wraps above any instruction count, so it ends the loop like running
// off the end of the function does.
const halt = -1

// instr is a decoded instruction in the threaded run loop: a shared handler
// plus its inline operands. The handler reads operands from the instr rather
// than from a captured closure, so no per-instruction closure is allocated.
type instr struct {
	fn   handler
	a, b uint64
}

type handler func(vm *vm, c *callFrame, ip int, in *instr) int

// brToLabel branches out labelIndex enclosing blocks, unwinding the value stack
// to the target's height (keeping its arity values) and returning the target
// instruction index.
func (c *callFrame) brToLabel(vm *vm, labelIndex int) int {
	targetIndex := len(c.controlStack) - labelIndex - 1
	target := c.controlStack[targetIndex]
	c.controlStack = c.controlStack[:targetIndex]
	vm.stack.unwind(target.stackHeight, target.arity)
	if target.isLoop {
		c.controlStack = append(c.controlStack, target)
	}
	return int(target.targetIp)
}

// vm is the WebAssembly Virtual Machine.
type vm struct {
	store             *store
	stack             *valueStack
	callDepth         int
	callStackCache    []callFrame
	controlStackCache []controlFrame
	localsCache       []value
	// brTables holds the label vectors of every br_table compiled into the
	// store, indexed by the instr operand. They are variable length, so they
	// live here rather than inline in the fixed-size instr.
	brTables [][]uint32
	// localsTop is the bump-allocation cursor into localsCache: each wasm call
	// carves its locals starting here, advances the cursor, and rewinds it when
	// the call returns. This lets a frame with many locals use the shared cache
	// rather than allocating them on the heap.
	localsTop int
	config    Config
	fuel      uint64
}

func newVm(config Config) *vm {
	ctrlCacheSize := config.CallStackPreallocationSize * controlStackCacheSlotSize
	localsCacheSize := config.CallStackPreallocationSize * localsCacheSlotSize
	return &vm{
		store:             &store{},
		stack:             newValueStack(),
		callStackCache:    make([]callFrame, config.CallStackPreallocationSize),
		controlStackCache: make([]controlFrame, ctrlCacheSize),
		localsCache:       make([]value, localsCacheSize),
		config:            config,
		fuel:              config.Fuel,
	}
}

func (vm *vm) instantiate(
	module *moduleDefinition,
	imports map[string]map[string]any,
) (*ModuleInstance, error) {
	validator := newValidator(vm.config)
	if err := validator.validateModule(module); err != nil {
		return nil, err
	}
	moduleInstance := &ModuleInstance{types: module.types, vm: vm}

	resolvedImports, err := resolveImports(module, moduleInstance, imports)
	if err != nil {
		return nil, err
	}

	for _, functionInstance := range resolvedImports.functions {
		storeIndex := uint32(len(vm.store.funcs))
		moduleInstance.funcAddrs = append(moduleInstance.funcAddrs, storeIndex)
		vm.store.funcs = append(vm.store.funcs, functionInstance)
	}

	for _, function := range module.funcs {
		storeIndex := uint32(len(vm.store.funcs))
		funType := module.types[function.typeIndex]
		wasmFunc := &wasmFunction{
			functionType: funType,
			module:       moduleInstance,
			code:         function,
		}
		if err := vm.compile(wasmFunc); err != nil {
			return nil, err
		}
		moduleInstance.funcAddrs = append(moduleInstance.funcAddrs, storeIndex)
		vm.store.funcs = append(vm.store.funcs, wasmFunc)
	}

	for _, table := range resolvedImports.tables {
		storeIndex := uint32(len(vm.store.tables))
		moduleInstance.tableAddrs = append(moduleInstance.tableAddrs, storeIndex)
		vm.store.tables = append(vm.store.tables, table)
	}

	for _, tableType := range module.tables {
		storeIndex := uint32(len(vm.store.tables))
		table := newTable(vm, tableType)
		moduleInstance.tableAddrs = append(moduleInstance.tableAddrs, storeIndex)
		vm.store.tables = append(vm.store.tables, table)
	}

	for _, memory := range resolvedImports.memories {
		storeIndex := uint32(len(vm.store.memories))
		moduleInstance.memAddrs = append(moduleInstance.memAddrs, storeIndex)
		vm.store.memories = append(vm.store.memories, memory)
	}

	for _, memoryType := range module.memories {
		storeIndex := uint32(len(vm.store.memories))
		memory := newMemory(vm, memoryType)
		moduleInstance.memAddrs = append(moduleInstance.memAddrs, storeIndex)
		vm.store.memories = append(vm.store.memories, memory)
	}

	for _, global := range resolvedImports.globals {
		storeIndex := uint32(len(vm.store.globals))
		moduleInstance.globalAddrs = append(moduleInstance.globalAddrs, storeIndex)
		vm.store.globals = append(vm.store.globals, global)
	}

	for _, variable := range module.globalVariables {
		valueType := variable.globalType.ValueType
		initExpression := variable.initExpression
		val, err := vm.invokeExpression(initExpression, valueType, moduleInstance)
		if err != nil {
			return nil, err
		}

		storeIndex := uint32(len(vm.store.globals))
		moduleInstance.globalAddrs = append(moduleInstance.globalAddrs, storeIndex)
		global := newGlobal(vm, val, variable.globalType.IsMutable, valueType)
		vm.store.globals = append(vm.store.globals, global)
	}

	for _, segment := range module.elementSegments {
		instance, err := vm.newElementInstance(segment, moduleInstance)
		if err != nil {
			return nil, err
		}
		storeIndex := uint32(len(vm.store.elements))
		moduleInstance.elemAddrs = append(moduleInstance.elemAddrs, storeIndex)
		vm.store.elements = append(vm.store.elements, instance)
	}

	// Apply active element segments to their target tables, then drop them.
	// Declarative segments are also dropped immediately.
	for i, segment := range module.elementSegments {
		if segment.mode == passiveElementMode {
			continue
		}
		elem := &vm.store.elements[moduleInstance.elemAddrs[i]]
		if segment.mode == activeElementMode {
			err := vm.applyActiveElementSegment(segment, elem, moduleInstance)
			if err != nil {
				return nil, err
			}
		}
		elem.functionIndexes = nil
	}

	// Allocate runtime data instances. The byte slice is shared with the parsed
	// dataSegment: the runtime only ever reads it (memory.init copies out) and
	// data.drop nils the instance's slice header, never the backing array, so the
	// moduleDefinition is left untouched and can be reinstantiated.
	for _, segment := range module.dataSegments {
		storeIndex := uint32(len(vm.store.datas))
		moduleInstance.dataAddrs = append(moduleInstance.dataAddrs, storeIndex)
		data := dataInstance{content: segment.content}
		vm.store.datas = append(vm.store.datas, data)
	}

	// Apply active data segments to their target memories, then drop them. After
	// this, memory.init against an active segment sees an empty content and traps
	// for any non-zero size.
	for i, segment := range module.dataSegments {
		if segment.mode != activeDataMode {
			continue
		}
		data := &vm.store.datas[moduleInstance.dataAddrs[i]]
		err := vm.applyActiveDataSegment(segment, data, moduleInstance)
		if err != nil {
			return nil, err
		}
		data.content = nil
	}

	if module.startIndex != nil {
		storeFunctionIndex := moduleInstance.funcAddrs[*module.startIndex]
		function := vm.store.funcs[storeFunctionIndex]
		if err := vm.invokeFunction(function); err != nil {
			return nil, err
		}
	}

	moduleInstance.exports = vm.resolveExports(module, moduleInstance)
	return moduleInstance, nil
}

func (vm *vm) invoke(function FunctionInstance, args []any) ([]any, error) {
	vm.stack.pushAll(args)
	if err := vm.invokeFunction(function); err != nil {
		return nil, err
	}
	return vm.stack.popValueTypes(function.GetType().ResultTypes), nil
}

func (vm *vm) invokeFunction(function FunctionInstance) error {
	switch f := function.(type) {
	case *wasmFunction:
		return vm.invokeWasmFunction(f)
	case *hostFunction:
		return vm.invokeHostFunction(f)
	default:
		return errUnknownFunctionType
	}
}

func (vm *vm) invokeWasmFunction(function *wasmFunction) error {
	if vm.callDepth >= vm.config.MaxCallStackDepth {
		return errCallStackExhausted
	}

	numParams := len(function.functionType.ParamTypes)
	numLocals := numParams + len(function.code.locals)

	// Carve this frame's locals at the cursor; nested calls bump past them, so
	// frames never overlap.
	localsMark := vm.localsTop
	var locals []value
	if end := localsMark + numLocals; end <= len(vm.localsCache) {
		locals = vm.localsCache[localsMark:end:end]
		vm.localsTop = end
		// Cache slots may hold stale values, and WASM allows reading
		// uninitialized locals, so zero the non-parameter locals. The parameter
		// slots are filled from the operand stack afterward.
		if function.code.defaultLocals != nil {
			copy(locals[numParams:], function.code.defaultLocals)
		} else {
			clear(locals[numParams:])
		}
	} else {
		// Not enough cache room: heap allocate (make already zeroes the locals).
		locals = make([]value, numLocals)
		if function.code.defaultLocals != nil {
			copy(locals[numParams:], function.code.defaultLocals)
		}
	}

	// Copy params and shrink stack by operating on the underlying slice directly.
	newLen := len(vm.stack.data) - numParams
	copy(locals[:numParams], vm.stack.data[newLen:])
	vm.stack.data = vm.stack.data[:newLen]

	// Threaded dispatch calls each handler through a function pointer, so the
	// frame pointer passed to it is opaque to escape analysis and a stack-local
	// frame would heap-allocate on every call. Hand out a frame from a
	// preallocated pool instead, so the escaping pointer targets long-lived
	// memory; only calls deeper than the pool fall back to the heap. The control
	// stack uses a matching per-depth slot from its own cache.
	var call *callFrame
	var controlStack []controlFrame
	if vm.callDepth < vm.config.CallStackPreallocationSize {
		call = &vm.callStackCache[vm.callDepth]
		blockDepth := vm.callDepth * controlStackCacheSlotSize
		max := blockDepth + controlStackCacheSlotSize
		controlStack = vm.controlStackCache[blockDepth:blockDepth:max]
	} else {
		call = &callFrame{}
	}
	// Push the function's entry block frame; branching to it returns past the
	// last instruction, ending the run loop.
	controlStack = append(controlStack, controlFrame{
		targetIp:    int32(len(function.instrs)),
		arity:       uint32(len(function.functionType.ResultTypes)),
		stackHeight: vm.stack.size(),
	})
	// Populate the frame, overwriting any stale data from a past call.
	*call = callFrame{
		controlStack: controlStack,
		locals:       locals,
		module:       function.module,
	}

	vm.callDepth++
	// Fuel checking on every instruction is wasteful when fuel is disabled, so
	// there are two loop implementations.
	var err error
	if vm.config.EnableFuel {
		err = vm.runLoopWithFuel(call, function.instrs, function.costs)
	} else {
		err = vm.runLoop(call, function.instrs)
	}
	// The run loop is done with this frame, so its locals slots are free to
	// reuse. Rewinding the cursor here is a no-op for the heap case.
	vm.localsTop = localsMark
	vm.callDepth--
	return err
}

func (vm *vm) runLoop(c *callFrame, instructions []instr) error {
	// Each handler returns the next instruction index, so dispatch is a single
	// indirect call with no per-instruction switch. The unsigned comparison ends
	// the loop both on a normal return (ip == len) and a trap (ip == halt, which
	// wraps above len); callFrame.trap distinguishes them.
	ip := 0
	for uint(ip) < uint(len(instructions)) {
		in := &instructions[ip]
		ip = in.fn(vm, c, ip, in)
	}
	return c.trap
}

func (vm *vm) runLoopWithFuel(
	c *callFrame, instructions []instr, costs []uint8,
) error {
	ip := 0
	for uint(ip) < uint(len(instructions)) {
		// A fused instr charges the fuel of every opcode it absorbed, so the
		// exhaustion point is identical to executing them unfused.
		cost := uint64(costs[ip])
		if vm.fuel < cost {
			return errFuelExhausted
		}
		vm.fuel -= cost
		in := &instructions[ip]
		ip = in.fn(vm, c, ip, in)
	}
	return c.trap
}

// operandWordCount returns the number of operand words after the opcode at
// body[operandStart-1], matching the encoding produced by parser.readCode.

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

// compile lowers a function to its instr; it runs once before the function is
// invoked. An unhandled opcode returns an error: validation has already
// accepted the module, so that signals a gap in the compiler, not bad input.

func (vm *vm) compile(fn *wasmFunction) error {
	instrs, costs, err := vm.compileBody(&fn.code, fn.module)
	if err != nil {
		return err
	}
	fn.instrs = instrs
	fn.costs = costs
	// instrs are the only runtime representation now; release the bytecode.
	fn.code.body = nil
	return nil
}

// ctrlEntry tracks an open block/loop/if during compilation so its branch
// targets can be back-patched into the header instr once they are known.
type ctrlEntry struct {
	op       opcode // block, loop, or ifOp
	headerIp int    // index of the header instr in code
	hasElse  bool
}

// compileBody lowers a function body to its instrs in a single pass. Block and
// if headers are emitted with placeholder targets and back-patched when their
// matching else/end is reached; br/br_if/br_table resolve their targets at run
// time against the control stack, so they need no patching here.
func (vm *vm) compileBody(
	fn *function, module *ModuleInstance,
) ([]instr, []uint8, error) {
	body := fn.body
	code := make([]instr, 0, len(body))
	var ctrl []ctrlEntry

	// Opcode-fusion bookkeeping. Fused instrs are emitted in the localGet and
	// i32Const cases below; fusing only ever merges adjacent data-stack opcodes,
	// which are never branch targets, so the back-patched control targets stay
	// correct. When fuel is enabled we record each fused instr's true cost so
	// runLoopWithFuel charges the absorbed opcodes; otherwise the cost slice is
	// left nil.
	fuelOn := vm.config.EnableFuel
	type fusedCost struct {
		ip   int
		cost uint8
	}
	var fused []fusedCost

	for pc := 0; pc < len(body); {
		op := opcode(body[pc])
		switch op {
		case block, ifOp:
			blockType := int32(body[pc+1])
			fn := opBlock
			if op == ifOp {
				fn = opIf
			}
			ctrl = append(ctrl, ctrlEntry{op: op, headerIp: len(code)})
			code = append(code, instr{
				fn: fn,
				b: uint64(vm.getOutputCount(module, blockType))<<32 |
					uint64(vm.getInputCount(module, blockType)),
			})
		case loop:
			blockType := int32(body[pc+1])
			ctrl = append(ctrl, ctrlEntry{op: loop, headerIp: len(code)})
			code = append(code, instr{
				fn: opLoop,
				a:  uint64(uint32(len(code) + 1)), // body starts after the header
				b:  uint64(vm.getInputCount(module, blockType)),
			})
		case elseOp:
			// elseIp is the first instr of the else body (after this opElse).
			top := &ctrl[len(ctrl)-1]
			code[top.headerIp].a |= uint64(uint32(len(code) + 1))
			top.hasElse = true
			code = append(code, instr{fn: opElse})
		case end:
			endIp := len(code)
			code = append(code, instr{fn: opEnd})
			if len(ctrl) > 0 {
				top := ctrl[len(ctrl)-1]
				ctrl = ctrl[:len(ctrl)-1]
				switch top.op {
				case block:
					code[top.headerIp].a = uint64(uint32(endIp + 1))
				case ifOp:
					code[top.headerIp].a |= uint64(uint32(endIp+1)) << 32
					if !top.hasElse {
						// A taken-false if with no else jumps to the end instr.
						code[top.headerIp].a |= uint64(uint32(endIp))
					}
				}
			}
		case unreachable:
			code = append(code, instr{fn: opUnreachable})
		case nop:
			code = append(code, instr{fn: opNop})
		case br:
			code = append(code, instr{fn: opBr, a: body[pc+1]})
		case brIf:
			code = append(code, instr{fn: opBrIf, a: body[pc+1]})
		case brTable:
			count := int(body[pc+1])
			labels := make([]uint32, count)
			for i := range labels {
				labels[i] = uint32(body[pc+2+i])
			}
			idx := len(vm.brTables)
			vm.brTables = append(vm.brTables, labels)
			code = append(code, instr{fn: opBrTable, a: uint64(idx), b: body[pc+2+count]})
		case returnOp:
			code = append(code, instr{fn: opReturn})
		case call:
			code = append(code, instr{fn: opCall, a: body[pc+1]})
		case callIndirect:
			code = append(code, instr{fn: opCallIndirect, a: body[pc+1], b: body[pc+2]})
		case drop:
			code = append(code, instr{fn: opDrop})
		case selectOp:
			code = append(code, instr{fn: opSelect})
		case selectT:
			code = append(code, instr{fn: opSelect})
		case localGet:
			localIdx := body[pc+1]
			next := pc + 2 // localGet is one operand word wide
			if next < len(body) && opcode(body[next]) == i32Const {
				constVal := body[next+1]
				after := next + 2 // i32Const is one operand word wide
				if after < len(body) && opcode(body[after]) == i32Add {
					// localGet, i32Const, i32Add -> push local[a] + const.
					code = append(code,
						instr{fn: opLocalGetI32ConstAdd, a: localIdx, b: constVal})
					if fuelOn {
						fused = append(fused, fusedCost{len(code) - 1, 3})
					}
					pc = after + 1
					continue
				}
				// localGet, i32Const -> push local[a] then const.
				code = append(code,
					instr{fn: opLocalGetI32Const, a: localIdx, b: constVal})
				if fuelOn {
					fused = append(fused, fusedCost{len(code) - 1, 2})
				}
				pc = after
				continue
			}
			if next < len(body) && opcode(body[next]) == localGet {
				// localGet, localGet -> push local[a] then local[b].
				code = append(code,
					instr{fn: opLocalGet2, a: localIdx, b: body[next+1]})
				if fuelOn {
					fused = append(fused, fusedCost{len(code) - 1, 2})
				}
				pc = next + 2
				continue
			}
			code = append(code, instr{fn: opLocalGet, a: localIdx})
		case localSet:
			code = append(code, instr{fn: opLocalSet, a: body[pc+1]})
		case localTee:
			code = append(code, instr{fn: opLocalTee, a: body[pc+1]})
		case globalGet:
			code = append(code, instr{fn: opGlobalGet, a: body[pc+1]})
		case globalSet:
			code = append(code, instr{fn: opGlobalSet, a: body[pc+1]})
		case tableGet:
			code = append(code, instr{fn: opTableGet, a: body[pc+1]})
		case tableSet:
			code = append(code, instr{fn: opTableSet, a: body[pc+1]})
		case i32Load:
			code = append(code, instr{fn: opI32Load, a: body[pc+2], b: body[pc+3]})
		case i64Load:
			code = append(code, instr{fn: opI64Load, a: body[pc+2], b: body[pc+3]})
		case f32Load:
			code = append(code, instr{fn: opF32Load, a: body[pc+2], b: body[pc+3]})
		case f64Load:
			code = append(code, instr{fn: opF64Load, a: body[pc+2], b: body[pc+3]})
		case i32Load8S:
			code = append(code, instr{fn: opI32Load8S, a: body[pc+2], b: body[pc+3]})
		case i32Load8U:
			code = append(code, instr{fn: opI32Load8U, a: body[pc+2], b: body[pc+3]})
		case i32Load16S:
			code = append(code, instr{fn: opI32Load16S, a: body[pc+2], b: body[pc+3]})
		case i32Load16U:
			code = append(code, instr{fn: opI32Load16U, a: body[pc+2], b: body[pc+3]})
		case i64Load8S:
			code = append(code, instr{fn: opI64Load8S, a: body[pc+2], b: body[pc+3]})
		case i64Load8U:
			code = append(code, instr{fn: opI64Load8U, a: body[pc+2], b: body[pc+3]})
		case i64Load16S:
			code = append(code, instr{fn: opI64Load16S, a: body[pc+2], b: body[pc+3]})
		case i64Load16U:
			code = append(code, instr{fn: opI64Load16U, a: body[pc+2], b: body[pc+3]})
		case i64Load32S:
			code = append(code, instr{fn: opI64Load32S, a: body[pc+2], b: body[pc+3]})
		case i64Load32U:
			code = append(code, instr{fn: opI64Load32U, a: body[pc+2], b: body[pc+3]})
		case i32Store:
			code = append(code, instr{fn: opI32Store, a: body[pc+2], b: body[pc+3]})
		case i64Store:
			code = append(code, instr{fn: opI64Store, a: body[pc+2], b: body[pc+3]})
		case f32Store:
			code = append(code, instr{fn: opF32Store, a: body[pc+2], b: body[pc+3]})
		case f64Store:
			code = append(code, instr{fn: opF64Store, a: body[pc+2], b: body[pc+3]})
		case i32Store8:
			code = append(code, instr{fn: opI32Store8, a: body[pc+2], b: body[pc+3]})
		case i32Store16:
			code = append(code, instr{fn: opI32Store16, a: body[pc+2], b: body[pc+3]})
		case i64Store8:
			code = append(code, instr{fn: opI64Store8, a: body[pc+2], b: body[pc+3]})
		case i64Store16:
			code = append(code, instr{fn: opI64Store16, a: body[pc+2], b: body[pc+3]})
		case i64Store32:
			code = append(code, instr{fn: opI64Store32, a: body[pc+2], b: body[pc+3]})
		case memorySize:
			code = append(code, instr{fn: opMemorySize, a: body[pc+1]})
		case memoryGrow:
			code = append(code, instr{fn: opMemoryGrow, a: body[pc+1]})
		case i32Const:
			constVal := body[pc+1]
			next := pc + 2 // i32Const is one operand word wide
			if next < len(body) && opcode(body[next]) == i32Add {
				// i32Const, i32Add -> add the constant to the stack top.
				code = append(code, instr{fn: opI32ConstAdd, a: constVal})
				if fuelOn {
					fused = append(fused, fusedCost{len(code) - 1, 2})
				}
				pc = next + 1
				continue
			}
			code = append(code, instr{fn: opI32Const, a: constVal})
		case i64Const:
			code = append(code, instr{fn: opI64Const, a: body[pc+1]})
		case f32Const:
			code = append(code, instr{fn: opF32Const, a: body[pc+1]})
		case f64Const:
			code = append(code, instr{fn: opF64Const, a: body[pc+1]})
		case i32Eqz:
			code = append(code, instr{fn: opI32Eqz})
		case i32Eq:
			code = append(code, instr{fn: opI32Eq})
		case i32Ne:
			code = append(code, instr{fn: opI32Ne})
		case i32LtS:
			code = append(code, instr{fn: opI32LtS})
		case i32LtU:
			code = append(code, instr{fn: opI32LtU})
		case i32GtS:
			code = append(code, instr{fn: opI32GtS})
		case i32GtU:
			code = append(code, instr{fn: opI32GtU})
		case i32LeS:
			code = append(code, instr{fn: opI32LeS})
		case i32LeU:
			code = append(code, instr{fn: opI32LeU})
		case i32GeS:
			code = append(code, instr{fn: opI32GeS})
		case i32GeU:
			code = append(code, instr{fn: opI32GeU})
		case i64Eqz:
			code = append(code, instr{fn: opI64Eqz})
		case i64Eq:
			code = append(code, instr{fn: opI64Eq})
		case i64Ne:
			code = append(code, instr{fn: opI64Ne})
		case i64LtS:
			code = append(code, instr{fn: opI64LtS})
		case i64LtU:
			code = append(code, instr{fn: opI64LtU})
		case i64GtS:
			code = append(code, instr{fn: opI64GtS})
		case i64GtU:
			code = append(code, instr{fn: opI64GtU})
		case i64LeS:
			code = append(code, instr{fn: opI64LeS})
		case i64LeU:
			code = append(code, instr{fn: opI64LeU})
		case i64GeS:
			code = append(code, instr{fn: opI64GeS})
		case i64GeU:
			code = append(code, instr{fn: opI64GeU})
		case f32Eq:
			code = append(code, instr{fn: opF32Eq})
		case f32Ne:
			code = append(code, instr{fn: opF32Ne})
		case f32Lt:
			code = append(code, instr{fn: opF32Lt})
		case f32Gt:
			code = append(code, instr{fn: opF32Gt})
		case f32Le:
			code = append(code, instr{fn: opF32Le})
		case f32Ge:
			code = append(code, instr{fn: opF32Ge})
		case f64Eq:
			code = append(code, instr{fn: opF64Eq})
		case f64Ne:
			code = append(code, instr{fn: opF64Ne})
		case f64Lt:
			code = append(code, instr{fn: opF64Lt})
		case f64Gt:
			code = append(code, instr{fn: opF64Gt})
		case f64Le:
			code = append(code, instr{fn: opF64Le})
		case f64Ge:
			code = append(code, instr{fn: opF64Ge})
		case i32Clz:
			code = append(code, instr{fn: opI32Clz})
		case i32Ctz:
			code = append(code, instr{fn: opI32Ctz})
		case i32Popcnt:
			code = append(code, instr{fn: opI32Popcnt})
		case i32Add:
			code = append(code, instr{fn: opI32Add})
		case i32Sub:
			code = append(code, instr{fn: opI32Sub})
		case i32Mul:
			code = append(code, instr{fn: opI32Mul})
		case i32DivS:
			code = append(code, instr{fn: opI32DivS})
		case i32DivU:
			code = append(code, instr{fn: opI32DivU})
		case i32RemS:
			code = append(code, instr{fn: opI32RemS})
		case i32RemU:
			code = append(code, instr{fn: opI32RemU})
		case i32And:
			code = append(code, instr{fn: opI32And})
		case i32Or:
			code = append(code, instr{fn: opI32Or})
		case i32Xor:
			code = append(code, instr{fn: opI32Xor})
		case i32Shl:
			code = append(code, instr{fn: opI32Shl})
		case i32ShrS:
			code = append(code, instr{fn: opI32ShrS})
		case i32ShrU:
			code = append(code, instr{fn: opI32ShrU})
		case i32Rotl:
			code = append(code, instr{fn: opI32Rotl})
		case i32Rotr:
			code = append(code, instr{fn: opI32Rotr})
		case i64Clz:
			code = append(code, instr{fn: opI64Clz})
		case i64Ctz:
			code = append(code, instr{fn: opI64Ctz})
		case i64Popcnt:
			code = append(code, instr{fn: opI64Popcnt})
		case i64Add:
			code = append(code, instr{fn: opI64Add})
		case i64Sub:
			code = append(code, instr{fn: opI64Sub})
		case i64Mul:
			code = append(code, instr{fn: opI64Mul})
		case i64DivS:
			code = append(code, instr{fn: opI64DivS})
		case i64DivU:
			code = append(code, instr{fn: opI64DivU})
		case i64RemS:
			code = append(code, instr{fn: opI64RemS})
		case i64RemU:
			code = append(code, instr{fn: opI64RemU})
		case i64And:
			code = append(code, instr{fn: opI64And})
		case i64Or:
			code = append(code, instr{fn: opI64Or})
		case i64Xor:
			code = append(code, instr{fn: opI64Xor})
		case i64Shl:
			code = append(code, instr{fn: opI64Shl})
		case i64ShrS:
			code = append(code, instr{fn: opI64ShrS})
		case i64ShrU:
			code = append(code, instr{fn: opI64ShrU})
		case i64Rotl:
			code = append(code, instr{fn: opI64Rotl})
		case i64Rotr:
			code = append(code, instr{fn: opI64Rotr})
		case f32Abs:
			code = append(code, instr{fn: opF32Abs})
		case f32Neg:
			code = append(code, instr{fn: opF32Neg})
		case f32Ceil:
			code = append(code, instr{fn: opF32Ceil})
		case f32Floor:
			code = append(code, instr{fn: opF32Floor})
		case f32Trunc:
			code = append(code, instr{fn: opF32Trunc})
		case f32Nearest:
			code = append(code, instr{fn: opF32Nearest})
		case f32Sqrt:
			code = append(code, instr{fn: opF32Sqrt})
		case f32Add:
			code = append(code, instr{fn: opF32Add})
		case f32Sub:
			code = append(code, instr{fn: opF32Sub})
		case f32Mul:
			code = append(code, instr{fn: opF32Mul})
		case f32Div:
			code = append(code, instr{fn: opF32Div})
		case f32Min:
			code = append(code, instr{fn: opF32Min})
		case f32Max:
			code = append(code, instr{fn: opF32Max})
		case f32Copysign:
			code = append(code, instr{fn: opF32Copysign})
		case f64Abs:
			code = append(code, instr{fn: opF64Abs})
		case f64Neg:
			code = append(code, instr{fn: opF64Neg})
		case f64Ceil:
			code = append(code, instr{fn: opF64Ceil})
		case f64Floor:
			code = append(code, instr{fn: opF64Floor})
		case f64Trunc:
			code = append(code, instr{fn: opF64Trunc})
		case f64Nearest:
			code = append(code, instr{fn: opF64Nearest})
		case f64Sqrt:
			code = append(code, instr{fn: opF64Sqrt})
		case f64Add:
			code = append(code, instr{fn: opF64Add})
		case f64Sub:
			code = append(code, instr{fn: opF64Sub})
		case f64Mul:
			code = append(code, instr{fn: opF64Mul})
		case f64Div:
			code = append(code, instr{fn: opF64Div})
		case f64Min:
			code = append(code, instr{fn: opF64Min})
		case f64Max:
			code = append(code, instr{fn: opF64Max})
		case f64Copysign:
			code = append(code, instr{fn: opF64Copysign})
		case i32WrapI64:
			code = append(code, instr{fn: opI32WrapI64})
		case i32TruncF32S:
			code = append(code, instr{fn: opI32TruncF32S})
		case i32TruncF32U:
			code = append(code, instr{fn: opI32TruncF32U})
		case i32TruncF64S:
			code = append(code, instr{fn: opI32TruncF64S})
		case i32TruncF64U:
			code = append(code, instr{fn: opI32TruncF64U})
		case i64ExtendI32S:
			code = append(code, instr{fn: opI64ExtendI32S})
		case i64ExtendI32U:
			code = append(code, instr{fn: opI64ExtendI32U})
		case i64TruncF32S:
			code = append(code, instr{fn: opI64TruncF32S})
		case i64TruncF32U:
			code = append(code, instr{fn: opI64TruncF32U})
		case i64TruncF64S:
			code = append(code, instr{fn: opI64TruncF64S})
		case i64TruncF64U:
			code = append(code, instr{fn: opI64TruncF64U})
		case f32ConvertI32S:
			code = append(code, instr{fn: opF32ConvertI32S})
		case f32ConvertI32U:
			code = append(code, instr{fn: opF32ConvertI32U})
		case f32ConvertI64S:
			code = append(code, instr{fn: opF32ConvertI64S})
		case f32ConvertI64U:
			code = append(code, instr{fn: opF32ConvertI64U})
		case f32DemoteF64:
			code = append(code, instr{fn: opF32DemoteF64})
		case f64ConvertI32S:
			code = append(code, instr{fn: opF64ConvertI32S})
		case f64ConvertI32U:
			code = append(code, instr{fn: opF64ConvertI32U})
		case f64ConvertI64S:
			code = append(code, instr{fn: opF64ConvertI64S})
		case f64ConvertI64U:
			code = append(code, instr{fn: opF64ConvertI64U})
		case f64PromoteF32:
			code = append(code, instr{fn: opF64PromoteF32})
		case i32ReinterpretF32:
			code = append(code, instr{fn: opI32ReinterpretF32})
		case i64ReinterpretF64:
			code = append(code, instr{fn: opI64ReinterpretF64})
		case f32ReinterpretI32:
			code = append(code, instr{fn: opF32ReinterpretI32})
		case f64ReinterpretI64:
			code = append(code, instr{fn: opF64ReinterpretI64})
		case i32Extend8S:
			code = append(code, instr{fn: opI32Extend8S})
		case i32Extend16S:
			code = append(code, instr{fn: opI32Extend16S})
		case i64Extend8S:
			code = append(code, instr{fn: opI64Extend8S})
		case i64Extend16S:
			code = append(code, instr{fn: opI64Extend16S})
		case i64Extend32S:
			code = append(code, instr{fn: opI64Extend32S})
		case refNull:
			code = append(code, instr{fn: opRefNull})
		case refIsNull:
			code = append(code, instr{fn: opRefIsNull})
		case refFunc:
			code = append(code, instr{fn: opRefFunc, a: body[pc+1]})
		case i32TruncSatF32S:
			code = append(code, instr{fn: opI32TruncSatF32S})
		case i32TruncSatF32U:
			code = append(code, instr{fn: opI32TruncSatF32U})
		case i32TruncSatF64S:
			code = append(code, instr{fn: opI32TruncSatF64S})
		case i32TruncSatF64U:
			code = append(code, instr{fn: opI32TruncSatF64U})
		case i64TruncSatF32S:
			code = append(code, instr{fn: opI64TruncSatF32S})
		case i64TruncSatF32U:
			code = append(code, instr{fn: opI64TruncSatF32U})
		case i64TruncSatF64S:
			code = append(code, instr{fn: opI64TruncSatF64S})
		case i64TruncSatF64U:
			code = append(code, instr{fn: opI64TruncSatF64U})
		case memoryInit:
			code = append(code, instr{fn: opMemoryInit, a: body[pc+1], b: body[pc+2]})
		case dataDrop:
			code = append(code, instr{fn: opDataDrop, a: body[pc+1]})
		case memoryCopy:
			code = append(code, instr{fn: opMemoryCopy, a: body[pc+1], b: body[pc+2]})
		case memoryFill:
			code = append(code, instr{fn: opMemoryFill, a: body[pc+1]})
		case tableInit:
			code = append(code, instr{fn: opTableInit, a: body[pc+1], b: body[pc+2]})
		case elemDrop:
			code = append(code, instr{fn: opElemDrop, a: body[pc+1]})
		case tableCopy:
			code = append(code, instr{fn: opTableCopy, a: body[pc+1], b: body[pc+2]})
		case tableGrow:
			code = append(code, instr{fn: opTableGrow, a: body[pc+1]})
		case tableSize:
			code = append(code, instr{fn: opTableSize, a: body[pc+1]})
		case tableFill:
			code = append(code, instr{fn: opTableFill, a: body[pc+1]})
		case v128Load:
			code = append(code, instr{fn: opV128Load, a: body[pc+2], b: body[pc+3]})
		case v128Load8x8S:
			code = append(code, instr{fn: opV128Load8x8S, a: body[pc+2], b: body[pc+3]})
		case v128Load8x8U:
			code = append(code, instr{fn: opV128Load8x8U, a: body[pc+2], b: body[pc+3]})
		case v128Load16x4S:
			code = append(code, instr{fn: opV128Load16x4S, a: body[pc+2], b: body[pc+3]})
		case v128Load16x4U:
			code = append(code, instr{fn: opV128Load16x4U, a: body[pc+2], b: body[pc+3]})
		case v128Load32x2S:
			code = append(code, instr{fn: opV128Load32x2S, a: body[pc+2], b: body[pc+3]})
		case v128Load32x2U:
			code = append(code, instr{fn: opV128Load32x2U, a: body[pc+2], b: body[pc+3]})
		case v128Load8Splat:
			code = append(code, instr{fn: opV128Load8Splat, a: body[pc+2], b: body[pc+3]})
		case v128Load16Splat:
			code = append(code, instr{fn: opV128Load16Splat, a: body[pc+2], b: body[pc+3]})
		case v128Load32Splat:
			code = append(code, instr{fn: opV128Load32Splat, a: body[pc+2], b: body[pc+3]})
		case v128Load64Splat:
			code = append(code, instr{fn: opV128Load64Splat, a: body[pc+2], b: body[pc+3]})
		case v128Store:
			code = append(code, instr{fn: opV128Store, a: body[pc+2], b: body[pc+3]})
		case v128Const:
			code = append(code, instr{fn: opV128Const, a: body[pc+1], b: body[pc+2]})
		case i8x16Shuffle:
			var a, b uint64
			for i := range 8 {
				a |= body[pc+1+i] << (8 * i)
				b |= body[pc+1+8+i] << (8 * i)
			}
			code = append(code, instr{fn: opI8x16Shuffle, a: a, b: b})
		case i8x16Swizzle:
			code = append(code, instr{fn: opI8x16Swizzle})
		case i8x16Splat:
			code = append(code, instr{fn: opI8x16Splat})
		case i16x8Splat:
			code = append(code, instr{fn: opI16x8Splat})
		case i32x4Splat:
			code = append(code, instr{fn: opI32x4Splat})
		case i64x2Splat:
			code = append(code, instr{fn: opI64x2Splat})
		case f32x4Splat:
			code = append(code, instr{fn: opF32x4Splat})
		case f64x2Splat:
			code = append(code, instr{fn: opF64x2Splat})
		case i8x16ExtractLaneS:
			code = append(code, instr{fn: opI8x16ExtractLaneS, a: body[pc+1]})
		case i8x16ExtractLaneU:
			code = append(code, instr{fn: opI8x16ExtractLaneU, a: body[pc+1]})
		case i8x16ReplaceLane:
			code = append(code, instr{fn: opI8x16ReplaceLane, a: body[pc+1]})
		case i16x8ExtractLaneS:
			code = append(code, instr{fn: opI16x8ExtractLaneS, a: body[pc+1]})
		case i16x8ExtractLaneU:
			code = append(code, instr{fn: opI16x8ExtractLaneU, a: body[pc+1]})
		case i16x8ReplaceLane:
			code = append(code, instr{fn: opI16x8ReplaceLane, a: body[pc+1]})
		case i32x4ExtractLane:
			code = append(code, instr{fn: opI32x4ExtractLane, a: body[pc+1]})
		case i32x4ReplaceLane:
			code = append(code, instr{fn: opI32x4ReplaceLane, a: body[pc+1]})
		case i64x2ExtractLane:
			code = append(code, instr{fn: opI64x2ExtractLane, a: body[pc+1]})
		case i64x2ReplaceLane:
			code = append(code, instr{fn: opI64x2ReplaceLane, a: body[pc+1]})
		case f32x4ExtractLane:
			code = append(code, instr{fn: opF32x4ExtractLane, a: body[pc+1]})
		case f32x4ReplaceLane:
			code = append(code, instr{fn: opF32x4ReplaceLane, a: body[pc+1]})
		case f64x2ExtractLane:
			code = append(code, instr{fn: opF64x2ExtractLane, a: body[pc+1]})
		case f64x2ReplaceLane:
			code = append(code, instr{fn: opF64x2ReplaceLane, a: body[pc+1]})
		case i8x16Eq:
			code = append(code, instr{fn: opI8x16Eq})
		case i8x16Ne:
			code = append(code, instr{fn: opI8x16Ne})
		case i8x16LtS:
			code = append(code, instr{fn: opI8x16LtS})
		case i8x16LtU:
			code = append(code, instr{fn: opI8x16LtU})
		case i8x16GtS:
			code = append(code, instr{fn: opI8x16GtS})
		case i8x16GtU:
			code = append(code, instr{fn: opI8x16GtU})
		case i8x16LeS:
			code = append(code, instr{fn: opI8x16LeS})
		case i8x16LeU:
			code = append(code, instr{fn: opI8x16LeU})
		case i8x16GeS:
			code = append(code, instr{fn: opI8x16GeS})
		case i8x16GeU:
			code = append(code, instr{fn: opI8x16GeU})
		case i16x8Eq:
			code = append(code, instr{fn: opI16x8Eq})
		case i16x8Ne:
			code = append(code, instr{fn: opI16x8Ne})
		case i16x8LtS:
			code = append(code, instr{fn: opI16x8LtS})
		case i16x8LtU:
			code = append(code, instr{fn: opI16x8LtU})
		case i16x8GtS:
			code = append(code, instr{fn: opI16x8GtS})
		case i16x8GtU:
			code = append(code, instr{fn: opI16x8GtU})
		case i16x8LeS:
			code = append(code, instr{fn: opI16x8LeS})
		case i16x8LeU:
			code = append(code, instr{fn: opI16x8LeU})
		case i16x8GeS:
			code = append(code, instr{fn: opI16x8GeS})
		case i16x8GeU:
			code = append(code, instr{fn: opI16x8GeU})
		case i32x4Eq:
			code = append(code, instr{fn: opI32x4Eq})
		case i32x4Ne:
			code = append(code, instr{fn: opI32x4Ne})
		case i32x4LtS:
			code = append(code, instr{fn: opI32x4LtS})
		case i32x4LtU:
			code = append(code, instr{fn: opI32x4LtU})
		case i32x4GtS:
			code = append(code, instr{fn: opI32x4GtS})
		case i32x4GtU:
			code = append(code, instr{fn: opI32x4GtU})
		case i32x4LeS:
			code = append(code, instr{fn: opI32x4LeS})
		case i32x4LeU:
			code = append(code, instr{fn: opI32x4LeU})
		case i32x4GeS:
			code = append(code, instr{fn: opI32x4GeS})
		case i32x4GeU:
			code = append(code, instr{fn: opI32x4GeU})
		case f32x4Eq:
			code = append(code, instr{fn: opF32x4Eq})
		case f32x4Ne:
			code = append(code, instr{fn: opF32x4Ne})
		case f32x4Lt:
			code = append(code, instr{fn: opF32x4Lt})
		case f32x4Gt:
			code = append(code, instr{fn: opF32x4Gt})
		case f32x4Le:
			code = append(code, instr{fn: opF32x4Le})
		case f32x4Ge:
			code = append(code, instr{fn: opF32x4Ge})
		case f64x2Eq:
			code = append(code, instr{fn: opF64x2Eq})
		case f64x2Ne:
			code = append(code, instr{fn: opF64x2Ne})
		case f64x2Lt:
			code = append(code, instr{fn: opF64x2Lt})
		case f64x2Gt:
			code = append(code, instr{fn: opF64x2Gt})
		case f64x2Le:
			code = append(code, instr{fn: opF64x2Le})
		case f64x2Ge:
			code = append(code, instr{fn: opF64x2Ge})
		case v128Not:
			code = append(code, instr{fn: opV128Not})
		case v128And:
			code = append(code, instr{fn: opV128And})
		case v128Andnot:
			code = append(code, instr{fn: opV128Andnot})
		case v128Or:
			code = append(code, instr{fn: opV128Or})
		case v128Xor:
			code = append(code, instr{fn: opV128Xor})
		case v128Bitselect:
			code = append(code, instr{fn: opV128Bitselect})
		case v128AnyTrue:
			code = append(code, instr{fn: opV128AnyTrue})
		case v128Load8Lane:
			code = append(code, instr{fn: opV128Load8Lane, a: body[pc+2], b: body[pc+3]<<32 | body[pc+4]})
		case v128Load16Lane:
			code = append(code, instr{fn: opV128Load16Lane, a: body[pc+2], b: body[pc+3]<<32 | body[pc+4]})
		case v128Load32Lane:
			code = append(code, instr{fn: opV128Load32Lane, a: body[pc+2], b: body[pc+3]<<32 | body[pc+4]})
		case v128Load64Lane:
			code = append(code, instr{fn: opV128Load64Lane, a: body[pc+2], b: body[pc+3]<<32 | body[pc+4]})
		case v128Store8Lane:
			code = append(code, instr{fn: opV128Store8Lane, a: body[pc+2], b: body[pc+3]<<32 | body[pc+4]})
		case v128Store16Lane:
			code = append(code, instr{fn: opV128Store16Lane, a: body[pc+2], b: body[pc+3]<<32 | body[pc+4]})
		case v128Store32Lane:
			code = append(code, instr{fn: opV128Store32Lane, a: body[pc+2], b: body[pc+3]<<32 | body[pc+4]})
		case v128Store64Lane:
			code = append(code, instr{fn: opV128Store64Lane, a: body[pc+2], b: body[pc+3]<<32 | body[pc+4]})
		case v128Load32Zero:
			code = append(code, instr{fn: opV128Load32Zero, a: body[pc+2], b: body[pc+3]})
		case v128Load64Zero:
			code = append(code, instr{fn: opV128Load64Zero, a: body[pc+2], b: body[pc+3]})
		case f32x4DemoteF64x2Zero:
			code = append(code, instr{fn: opF32x4DemoteF64x2Zero})
		case f64x2PromoteLowF32x4:
			code = append(code, instr{fn: opF64x2PromoteLowF32x4})
		case i8x16Abs:
			code = append(code, instr{fn: opI8x16Abs})
		case i8x16Neg:
			code = append(code, instr{fn: opI8x16Neg})
		case i8x16Popcnt:
			code = append(code, instr{fn: opI8x16Popcnt})
		case i8x16AllTrue:
			code = append(code, instr{fn: opI8x16AllTrue})
		case i8x16Bitmask:
			code = append(code, instr{fn: opI8x16Bitmask})
		case i8x16NarrowI16x8S:
			code = append(code, instr{fn: opI8x16NarrowI16x8S})
		case i8x16NarrowI16x8U:
			code = append(code, instr{fn: opI8x16NarrowI16x8U})
		case f32x4Ceil:
			code = append(code, instr{fn: opF32x4Ceil})
		case f32x4Floor:
			code = append(code, instr{fn: opF32x4Floor})
		case f32x4Trunc:
			code = append(code, instr{fn: opF32x4Trunc})
		case f32x4Nearest:
			code = append(code, instr{fn: opF32x4Nearest})
		case i8x16Shl:
			code = append(code, instr{fn: opI8x16Shl})
		case i8x16ShrU:
			code = append(code, instr{fn: opI8x16ShrU})
		case i8x16ShrS:
			code = append(code, instr{fn: opI8x16ShrS})
		case i8x16Add:
			code = append(code, instr{fn: opI8x16Add})
		case i8x16AddSatS:
			code = append(code, instr{fn: opI8x16AddSatS})
		case i8x16AddSatU:
			code = append(code, instr{fn: opI8x16AddSatU})
		case i8x16Sub:
			code = append(code, instr{fn: opI8x16Sub})
		case i8x16SubSatS:
			code = append(code, instr{fn: opI8x16SubSatS})
		case i8x16SubSatU:
			code = append(code, instr{fn: opI8x16SubSatU})
		case f64x2Ceil:
			code = append(code, instr{fn: opF64x2Ceil})
		case f64x2Floor:
			code = append(code, instr{fn: opF64x2Floor})
		case i8x16MinS:
			code = append(code, instr{fn: opI8x16MinS})
		case i8x16MinU:
			code = append(code, instr{fn: opI8x16MinU})
		case i8x16MaxS:
			code = append(code, instr{fn: opI8x16MaxS})
		case i8x16MaxU:
			code = append(code, instr{fn: opI8x16MaxU})
		case f64x2Trunc:
			code = append(code, instr{fn: opF64x2Trunc})
		case i8x16AvgrU:
			code = append(code, instr{fn: opI8x16AvgrU})
		case i16x8ExtaddPairwiseI8x16S:
			code = append(code, instr{fn: opI16x8ExtaddPairwiseI8x16S})
		case i16x8ExtaddPairwiseI8x16U:
			code = append(code, instr{fn: opI16x8ExtaddPairwiseI8x16U})
		case i32x4ExtaddPairwiseI16x8S:
			code = append(code, instr{fn: opI32x4ExtaddPairwiseI16x8S})
		case i32x4ExtaddPairwiseI16x8U:
			code = append(code, instr{fn: opI32x4ExtaddPairwiseI16x8U})
		case i16x8Abs:
			code = append(code, instr{fn: opI16x8Abs})
		case i16x8Neg:
			code = append(code, instr{fn: opI16x8Neg})
		case i16x8Q15mulrSatS:
			code = append(code, instr{fn: opI16x8Q15mulrSatS})
		case i16x8AllTrue:
			code = append(code, instr{fn: opI16x8AllTrue})
		case i16x8Bitmask:
			code = append(code, instr{fn: opI16x8Bitmask})
		case i16x8NarrowI32x4S:
			code = append(code, instr{fn: opI16x8NarrowI32x4S})
		case i16x8NarrowI32x4U:
			code = append(code, instr{fn: opI16x8NarrowI32x4U})
		case i16x8ExtendLowI8x16S:
			code = append(code, instr{fn: opI16x8ExtendLowI8x16S})
		case i16x8ExtendHighI8x16S:
			code = append(code, instr{fn: opI16x8ExtendHighI8x16S})
		case i16x8ExtendLowI8x16U:
			code = append(code, instr{fn: opI16x8ExtendLowI8x16U})
		case i16x8ExtendHighI8x16U:
			code = append(code, instr{fn: opI16x8ExtendHighI8x16U})
		case i16x8Shl:
			code = append(code, instr{fn: opI16x8Shl})
		case i16x8ShrS:
			code = append(code, instr{fn: opI16x8ShrS})
		case i16x8ShrU:
			code = append(code, instr{fn: opI16x8ShrU})
		case i16x8Add:
			code = append(code, instr{fn: opI16x8Add})
		case i16x8AddSatS:
			code = append(code, instr{fn: opI16x8AddSatS})
		case i16x8AddSatU:
			code = append(code, instr{fn: opI16x8AddSatU})
		case i16x8Sub:
			code = append(code, instr{fn: opI16x8Sub})
		case i16x8SubSatS:
			code = append(code, instr{fn: opI16x8SubSatS})
		case i16x8SubSatU:
			code = append(code, instr{fn: opI16x8SubSatU})
		case f64x2Nearest:
			code = append(code, instr{fn: opF64x2Nearest})
		case i16x8Mul:
			code = append(code, instr{fn: opI16x8Mul})
		case i16x8MinS:
			code = append(code, instr{fn: opI16x8MinS})
		case i16x8MinU:
			code = append(code, instr{fn: opI16x8MinU})
		case i16x8MaxS:
			code = append(code, instr{fn: opI16x8MaxS})
		case i16x8MaxU:
			code = append(code, instr{fn: opI16x8MaxU})
		case i16x8AvgrU:
			code = append(code, instr{fn: opI16x8AvgrU})
		case i16x8ExtmulLowI8x16S:
			code = append(code, instr{fn: opI16x8ExtmulLowI8x16S})
		case i16x8ExtmulHighI8x16S:
			code = append(code, instr{fn: opI16x8ExtmulHighI8x16S})
		case i16x8ExtmulLowI8x16U:
			code = append(code, instr{fn: opI16x8ExtmulLowI8x16U})
		case i16x8ExtmulHighI8x16U:
			code = append(code, instr{fn: opI16x8ExtmulHighI8x16U})
		case i32x4Abs:
			code = append(code, instr{fn: opI32x4Abs})
		case i32x4Neg:
			code = append(code, instr{fn: opI32x4Neg})
		case i32x4AllTrue:
			code = append(code, instr{fn: opI32x4AllTrue})
		case i32x4Bitmask:
			code = append(code, instr{fn: opI32x4Bitmask})
		case i32x4ExtendLowI16x8S:
			code = append(code, instr{fn: opI32x4ExtendLowI16x8S})
		case i32x4ExtendHighI16x8S:
			code = append(code, instr{fn: opI32x4ExtendHighI16x8S})
		case i32x4ExtendLowI16x8U:
			code = append(code, instr{fn: opI32x4ExtendLowI16x8U})
		case i32x4ExtendHighI16x8U:
			code = append(code, instr{fn: opI32x4ExtendHighI16x8U})
		case i32x4Shl:
			code = append(code, instr{fn: opI32x4Shl})
		case i32x4ShrS:
			code = append(code, instr{fn: opI32x4ShrS})
		case i32x4ShrU:
			code = append(code, instr{fn: opI32x4ShrU})
		case i32x4Add:
			code = append(code, instr{fn: opI32x4Add})
		case i32x4Sub:
			code = append(code, instr{fn: opI32x4Sub})
		case i32x4Mul:
			code = append(code, instr{fn: opI32x4Mul})
		case i32x4MinS:
			code = append(code, instr{fn: opI32x4MinS})
		case i32x4MinU:
			code = append(code, instr{fn: opI32x4MinU})
		case i32x4MaxS:
			code = append(code, instr{fn: opI32x4MaxS})
		case i32x4MaxU:
			code = append(code, instr{fn: opI32x4MaxU})
		case i32x4DotI16x8S:
			code = append(code, instr{fn: opI32x4DotI16x8S})
		case i32x4ExtmulLowI16x8S:
			code = append(code, instr{fn: opI32x4ExtmulLowI16x8S})
		case i32x4ExtmulHighI16x8S:
			code = append(code, instr{fn: opI32x4ExtmulHighI16x8S})
		case i32x4ExtmulLowI16x8U:
			code = append(code, instr{fn: opI32x4ExtmulLowI16x8U})
		case i32x4ExtmulHighI16x8U:
			code = append(code, instr{fn: opI32x4ExtmulHighI16x8U})
		case i64x2Abs:
			code = append(code, instr{fn: opI64x2Abs})
		case i64x2Neg:
			code = append(code, instr{fn: opI64x2Neg})
		case i64x2AllTrue:
			code = append(code, instr{fn: opI64x2AllTrue})
		case i64x2Bitmask:
			code = append(code, instr{fn: opI64x2Bitmask})
		case i64x2ExtendLowI32x4S:
			code = append(code, instr{fn: opI64x2ExtendLowI32x4S})
		case i64x2ExtendHighI32x4S:
			code = append(code, instr{fn: opI64x2ExtendHighI32x4S})
		case i64x2ExtendLowI32x4U:
			code = append(code, instr{fn: opI64x2ExtendLowI32x4U})
		case i64x2ExtendHighI32x4U:
			code = append(code, instr{fn: opI64x2ExtendHighI32x4U})
		case i64x2Shl:
			code = append(code, instr{fn: opI64x2Shl})
		case i64x2ShrS:
			code = append(code, instr{fn: opI64x2ShrS})
		case i64x2ShrU:
			code = append(code, instr{fn: opI64x2ShrU})
		case i64x2Add:
			code = append(code, instr{fn: opI64x2Add})
		case i64x2Sub:
			code = append(code, instr{fn: opI64x2Sub})
		case i64x2Mul:
			code = append(code, instr{fn: opI64x2Mul})
		case i64x2Eq:
			code = append(code, instr{fn: opI64x2Eq})
		case i64x2Ne:
			code = append(code, instr{fn: opI64x2Ne})
		case i64x2LtS:
			code = append(code, instr{fn: opI64x2LtS})
		case i64x2GtS:
			code = append(code, instr{fn: opI64x2GtS})
		case i64x2LeS:
			code = append(code, instr{fn: opI64x2LeS})
		case i64x2GeS:
			code = append(code, instr{fn: opI64x2GeS})
		case i64x2ExtmulLowI32x4S:
			code = append(code, instr{fn: opI64x2ExtmulLowI32x4S})
		case i64x2ExtmulHighI32x4S:
			code = append(code, instr{fn: opI64x2ExtmulHighI32x4S})
		case i64x2ExtmulLowI32x4U:
			code = append(code, instr{fn: opI64x2ExtmulLowI32x4U})
		case i64x2ExtmulHighI32x4U:
			code = append(code, instr{fn: opI64x2ExtmulHighI32x4U})
		case f32x4Abs:
			code = append(code, instr{fn: opF32x4Abs})
		case f32x4Neg:
			code = append(code, instr{fn: opF32x4Neg})
		case f32x4Sqrt:
			code = append(code, instr{fn: opF32x4Sqrt})
		case f32x4Add:
			code = append(code, instr{fn: opF32x4Add})
		case f32x4Sub:
			code = append(code, instr{fn: opF32x4Sub})
		case f32x4Mul:
			code = append(code, instr{fn: opF32x4Mul})
		case f32x4Div:
			code = append(code, instr{fn: opF32x4Div})
		case f32x4Min:
			code = append(code, instr{fn: opF32x4Min})
		case f32x4Max:
			code = append(code, instr{fn: opF32x4Max})
		case f32x4Pmin:
			code = append(code, instr{fn: opF32x4Pmin})
		case f32x4Pmax:
			code = append(code, instr{fn: opF32x4Pmax})
		case f64x2Abs:
			code = append(code, instr{fn: opF64x2Abs})
		case f64x2Neg:
			code = append(code, instr{fn: opF64x2Neg})
		case f64x2Sqrt:
			code = append(code, instr{fn: opF64x2Sqrt})
		case f64x2Add:
			code = append(code, instr{fn: opF64x2Add})
		case f64x2Sub:
			code = append(code, instr{fn: opF64x2Sub})
		case f64x2Mul:
			code = append(code, instr{fn: opF64x2Mul})
		case f64x2Div:
			code = append(code, instr{fn: opF64x2Div})
		case f64x2Min:
			code = append(code, instr{fn: opF64x2Min})
		case f64x2Max:
			code = append(code, instr{fn: opF64x2Max})
		case f64x2Pmin:
			code = append(code, instr{fn: opF64x2Pmin})
		case f64x2Pmax:
			code = append(code, instr{fn: opF64x2Pmax})
		case i32x4TruncSatF32x4S:
			code = append(code, instr{fn: opI32x4TruncSatF32x4S})
		case i32x4TruncSatF32x4U:
			code = append(code, instr{fn: opI32x4TruncSatF32x4U})
		case f32x4ConvertI32x4S:
			code = append(code, instr{fn: opF32x4ConvertI32x4S})
		case f32x4ConvertI32x4U:
			code = append(code, instr{fn: opF32x4ConvertI32x4U})
		case i32x4TruncSatF64x2SZero:
			code = append(code, instr{fn: opI32x4TruncSatF64x2SZero})
		case i32x4TruncSatF64x2UZero:
			code = append(code, instr{fn: opI32x4TruncSatF64x2UZero})
		case f64x2ConvertLowI32x4S:
			code = append(code, instr{fn: opF64x2ConvertLowI32x4S})
		case f64x2ConvertLowI32x4U:
			code = append(code, instr{fn: opF64x2ConvertLowI32x4U})
		default:
			return nil, nil, fmt.Errorf("unhandled opcode %d", opcode(body[pc]))
		}
		pc += 1 + operandWordCount(op, body, pc+1)
	}

	var costs []uint8
	if fuelOn {
		costs = make([]uint8, len(code))
		for i := range costs {
			costs[i] = 1
		}
		for _, fc := range fused {
			costs[fc.ip] = fc.cost
		}
	}
	return code, costs, nil
}

// ---- instruction handlers ----

func opNop(vm *vm, c *callFrame, ip int, in *instr) int { return ip + 1 }

func opUnreachable(vm *vm, c *callFrame, ip int, in *instr) int {
	c.trap = errUnreachable
	return halt
}

func opDrop(vm *vm, c *callFrame, ip int, in *instr) int { vm.stack.drop(); return ip + 1 }

func opI32Const(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(int32(in.a))
	return ip + 1
}

func opI64Const(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(int64(in.a))
	return ip + 1
}

func opF32Const(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(math.Float32frombits(uint32(in.a)))
	return ip + 1
}

func opF64Const(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(math.Float64frombits(in.a))
	return ip + 1
}

func opLocalGet(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.push(c.locals[in.a])
	return ip + 1
}

func opLocalSet(vm *vm, c *callFrame, ip int, in *instr) int {
	c.locals[in.a] = vm.stack.pop()
	return ip + 1
}

func opLocalTee(vm *vm, c *callFrame, ip int, in *instr) int {
	c.locals[in.a] = vm.stack.data[len(vm.stack.data)-1]
	return ip + 1
}

// opLocalGetI32Const fuses localGet+i32Const: push local[a] then the constant.
func opLocalGetI32Const(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.push(c.locals[in.a])
	vm.stack.pushInt32(int32(in.b))
	return ip + 1
}

// opLocalGet2 fuses localGet+localGet: push local[a] then local[b].
func opLocalGet2(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.push(c.locals[in.a])
	vm.stack.push(c.locals[in.b])
	return ip + 1
}

// opI32ConstAdd fuses i32Const+i32Add: add the constant to the stack top.
func opI32ConstAdd(vm *vm, c *callFrame, ip int, in *instr) int {
	top := len(vm.stack.data) - 1
	vm.stack.data[top] = i32(vm.stack.data[top].int32() + int32(in.a))
	return ip + 1
}

// opLocalGetI32ConstAdd fuses localGet+i32Const+i32Add: push local[a]+const.
func opLocalGetI32ConstAdd(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(c.locals[in.a].int32() + int32(in.b))
	return ip + 1
}

func opBlock(vm *vm, c *callFrame, ip int, in *instr) int {
	c.controlStack = append(c.controlStack, controlFrame{
		targetIp:    int32(in.a),
		arity:       uint32(in.b >> 32),
		stackHeight: vm.stack.size() - uint32(in.b),
	})
	return ip + 1
}

func opLoop(vm *vm, c *callFrame, ip int, in *instr) int {
	c.controlStack = append(c.controlStack, controlFrame{
		isLoop:      true,
		targetIp:    int32(in.a),
		arity:       uint32(in.b),
		stackHeight: vm.stack.size() - uint32(in.b),
	})
	return ip + 1
}

func opIf(vm *vm, c *callFrame, ip int, in *instr) int {
	condition := vm.stack.popInt32()
	c.controlStack = append(c.controlStack, controlFrame{
		targetIp:    int32(in.a >> 32),
		arity:       uint32(in.b >> 32),
		stackHeight: vm.stack.size() - uint32(in.b),
	})
	if condition == 0 {
		return int(int32(in.a))
	}
	return ip + 1
}

func opElse(vm *vm, c *callFrame, ip int, in *instr) int {
	target := c.controlStack[len(c.controlStack)-1].targetIp
	c.controlStack = c.controlStack[:len(c.controlStack)-1]
	return int(target)
}

func opEnd(vm *vm, c *callFrame, ip int, in *instr) int {
	if len(c.controlStack) > 0 {
		c.controlStack = c.controlStack[:len(c.controlStack)-1]
	}
	return ip + 1
}

func opBr(vm *vm, c *callFrame, ip int, in *instr) int { return c.brToLabel(vm, int(in.a)) }

func opBrTable(vm *vm, c *callFrame, ip int, in *instr) int {
	labels := vm.brTables[in.a]
	index := uint32(vm.stack.popInt32())
	if index < uint32(len(labels)) {
		return c.brToLabel(vm, int(labels[index]))
	}
	return c.brToLabel(vm, int(in.b))
}

func opBrIf(vm *vm, c *callFrame, ip int, in *instr) int {
	if vm.stack.popInt32() != 0 {
		return c.brToLabel(vm, int(in.a))
	}
	return ip + 1
}

func opReturn(vm *vm, c *callFrame, ip int, in *instr) int {
	return c.brToLabel(vm, len(c.controlStack)-1)
}

func opCall(vm *vm, c *callFrame, ip int, in *instr) int {
	function := vm.store.funcs[c.module.funcAddrs[in.a]]
	if err := vm.invokeFunction(function); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opCallIndirect(vm *vm, c *callFrame, ip int, in *instr) int {
	expectedType := c.module.types[in.a]
	table := vm.getTable(c, in.b)
	elementIndex := vm.stack.popInt32()
	tableElement, err := table.Get(elementIndex)
	if err != nil {
		c.trap = err
		return halt
	}
	if tableElement == NullReference {
		c.trap = fmt.Errorf("uninitialized element %d", elementIndex)
		return halt
	}
	function := vm.store.funcs[tableElement]
	if !function.GetType().Equal(expectedType) {
		c.trap = errIndirectCallTypeMismatch
		return halt
	}
	if err := vm.invokeFunction(function); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opSelect(vm *vm, c *callFrame, ip int, in *instr) int {
	data := vm.stack.data
	n := len(data)
	var top value
	if data[n-1].int32() != 0 {
		top = data[n-3]
	} else {
		top = data[n-2]
	}
	data[n-3] = top
	vm.stack.data = data[:n-2]
	return ip + 1
}

func opGlobalGet(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.push(vm.getGlobal(c, in.a).value)
	return ip + 1
}

func opGlobalSet(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.getGlobal(c, in.a).value = vm.stack.pop()
	return ip + 1
}

func opMemorySize(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(vm.getMemory(c, in.a).Size())
	return ip + 1
}

func opMemoryGrow(vm *vm, c *callFrame, ip int, in *instr) int {
	memory := vm.getMemory(c, in.a)
	vm.stack.pushInt32(memory.Grow(vm.stack.popInt32()))
	return ip + 1
}

func opRefIsNull(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(boolToInt32(vm.stack.popInt32() == NullReference))
	return ip + 1
}

func opRefNull(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(NullReference)
	return ip + 1
}

func opRefFunc(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(int32(c.module.funcAddrs[in.a]))
	return ip + 1
}

func opMemoryInit(vm *vm, c *callFrame, ip int, in *instr) int {
	data := vm.getData(c, in.a)
	memory := vm.getMemory(c, in.b)
	n, s, d := vm.stack.pop3Int32()
	if err := memory.Init(uint32(n), uint32(s), uint32(d), data.content); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opMemoryCopy(vm *vm, c *callFrame, ip int, in *instr) int {
	destMemory := vm.getMemory(c, in.a)
	srcMemory := vm.getMemory(c, in.b)
	n, s, d := vm.stack.pop3Int32()
	if err := srcMemory.Copy(destMemory, uint32(n), uint32(s), uint32(d)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opMemoryFill(vm *vm, c *callFrame, ip int, in *instr) int {
	memory := vm.getMemory(c, in.a)
	n, val, offset := vm.stack.pop3Int32()
	if err := memory.Fill(uint32(n), uint32(offset), byte(val)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opDataDrop(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.getData(c, in.a).content = nil
	return ip + 1
}

func opTableInit(vm *vm, c *callFrame, ip int, in *instr) int {
	element := vm.getElement(c, in.a)
	table := vm.getTable(c, in.b)
	n, s, d := vm.stack.pop3Int32()
	if err := table.Init(n, d, s, element.functionIndexes); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opElemDrop(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.getElement(c, in.a).functionIndexes = nil
	return ip + 1
}

func opTableCopy(vm *vm, c *callFrame, ip int, in *instr) int {
	destTable := vm.getTable(c, in.a)
	srcTable := vm.getTable(c, in.b)
	n, s, d := vm.stack.pop3Int32()
	if err := srcTable.Copy(destTable, n, s, d); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opTableGrow(vm *vm, c *callFrame, ip int, in *instr) int {
	table := vm.getTable(c, in.a)
	n := vm.stack.popInt32()
	val := vm.stack.popInt32()
	vm.stack.pushInt32(table.Grow(n, val))
	return ip + 1
}

func opTableSize(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(int32(vm.getTable(c, in.a).Size()))
	return ip + 1
}

func opTableFill(vm *vm, c *callFrame, ip int, in *instr) int {
	table := vm.getTable(c, in.a)
	n, val, i := vm.stack.pop3Int32()
	if err := table.Fill(n, i, val); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opTableGet(vm *vm, c *callFrame, ip int, in *instr) int {
	table := vm.getTable(c, in.a)
	element, err := table.Get(vm.stack.popInt32())
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushInt32(element)
	return ip + 1
}

func opTableSet(vm *vm, c *callFrame, ip int, in *instr) int {
	table := vm.getTable(c, in.a)
	reference := vm.stack.popInt32()
	index := vm.stack.popInt32()
	if err := table.Set(index, reference); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32Eqz(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(boolToInt32(vm.stack.popInt32() == 0))
	return ip + 1
}

func opI32Eq(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32Eq(); return ip + 1 }

func opI32Ne(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32Ne(); return ip + 1 }

func opI32LtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32LtS(); return ip + 1 }

func opI32LtU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32LtU(); return ip + 1 }

func opI32GtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32GtS(); return ip + 1 }

func opI32GtU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32GtU(); return ip + 1 }

func opI32LeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32LeS(); return ip + 1 }

func opI32LeU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32LeU(); return ip + 1 }

func opI32GeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32GeS(); return ip + 1 }

func opI32GeU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32GeU(); return ip + 1 }

func opI64Eqz(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(boolToInt32(vm.stack.popInt64() == 0))
	return ip + 1
}

func opI64Eq(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64Eq(); return ip + 1 }

func opI64Ne(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64Ne(); return ip + 1 }

func opI64LtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64LtS(); return ip + 1 }

func opI64LtU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64LtU(); return ip + 1 }

func opI64GtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64GtS(); return ip + 1 }

func opI64GtU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64GtU(); return ip + 1 }

func opI64LeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64LeS(); return ip + 1 }

func opI64LeU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64LeU(); return ip + 1 }

func opI64GeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64GeS(); return ip + 1 }

func opI64GeU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64GeU(); return ip + 1 }

func opF32Eq(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Eq(); return ip + 1 }

func opF32Ne(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Ne(); return ip + 1 }

func opF32Lt(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Lt(); return ip + 1 }

func opF32Gt(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Gt(); return ip + 1 }

func opF32Le(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Le(); return ip + 1 }

func opF32Ge(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Ge(); return ip + 1 }

func opF64Eq(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Eq(); return ip + 1 }

func opF64Ne(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Ne(); return ip + 1 }

func opF64Lt(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Lt(); return ip + 1 }

func opF64Gt(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Gt(); return ip + 1 }

func opF64Le(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Le(); return ip + 1 }

func opF64Ge(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Ge(); return ip + 1 }

func opI32Clz(vm *vm, c *callFrame, ip int, in *instr) int {
	a := vm.stack.popInt32()
	vm.stack.pushInt32(int32(bits.LeadingZeros32(uint32(a))))
	return ip + 1
}

func opI32Ctz(vm *vm, c *callFrame, ip int, in *instr) int {
	a := vm.stack.popInt32()
	vm.stack.pushInt32(int32(bits.TrailingZeros32(uint32(a))))
	return ip + 1
}

func opI32Popcnt(vm *vm, c *callFrame, ip int, in *instr) int {
	a := vm.stack.popInt32()
	vm.stack.pushInt32(int32(bits.OnesCount32(uint32(a))))
	return ip + 1
}

func opI32Add(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	res := vm.stack.data[len(vm.stack.data)-1].int32() + b
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI32Sub(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	res := vm.stack.data[len(vm.stack.data)-1].int32() - b
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI32Mul(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	res := vm.stack.data[len(vm.stack.data)-1].int32() * b
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI32DivS(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32DivS(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32DivU(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32DivU(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32RemS(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32RemS(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32RemU(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32RemU(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32And(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	res := vm.stack.data[len(vm.stack.data)-1].int32() & b
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI32Or(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	res := vm.stack.data[len(vm.stack.data)-1].int32() | b
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI32Xor(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	res := vm.stack.data[len(vm.stack.data)-1].int32() ^ b
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI32Shl(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	res := vm.stack.data[len(vm.stack.data)-1].int32() << (uint32(b) % 32)
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI32ShrS(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	res := vm.stack.data[len(vm.stack.data)-1].int32() >> (uint32(b) % 32)
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI32ShrU(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	a := vm.stack.data[len(vm.stack.data)-1].int32()
	res := int32(uint32(a) >> (uint32(b) % 32))
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI32Rotl(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	a := vm.stack.data[len(vm.stack.data)-1].int32()
	res := int32(bits.RotateLeft32(uint32(a), int(b)))
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI32Rotr(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt32()
	a := vm.stack.data[len(vm.stack.data)-1].int32()
	res := int32(bits.RotateLeft32(uint32(a), -int(b)))
	vm.stack.data[len(vm.stack.data)-1] = i32(res)
	return ip + 1
}

func opI64Clz(vm *vm, c *callFrame, ip int, in *instr) int {
	a := vm.stack.popInt64()
	vm.stack.pushInt64(int64(bits.LeadingZeros64(uint64(a))))
	return ip + 1
}

func opI64Ctz(vm *vm, c *callFrame, ip int, in *instr) int {
	a := vm.stack.popInt64()
	vm.stack.pushInt64(int64(bits.TrailingZeros64(uint64(a))))
	return ip + 1
}

func opI64Popcnt(vm *vm, c *callFrame, ip int, in *instr) int {
	a := vm.stack.popInt64()
	vm.stack.pushInt64(int64(bits.OnesCount64(uint64(a))))
	return ip + 1
}

func opI64Add(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	res := vm.stack.data[len(vm.stack.data)-1].int64() + b
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opI64Sub(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	res := vm.stack.data[len(vm.stack.data)-1].int64() - b
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opI64Mul(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	res := vm.stack.data[len(vm.stack.data)-1].int64() * b
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opI64DivS(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64DivS(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64DivU(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64DivU(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64RemS(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64RemS(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64RemU(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64RemU(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64And(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	res := vm.stack.data[len(vm.stack.data)-1].int64() & b
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opI64Or(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	res := vm.stack.data[len(vm.stack.data)-1].int64() | b
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opI64Xor(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	res := vm.stack.data[len(vm.stack.data)-1].int64() ^ b
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opI64Shl(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	res := vm.stack.data[len(vm.stack.data)-1].int64() << (uint64(b) % 64)
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opI64ShrS(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	res := vm.stack.data[len(vm.stack.data)-1].int64() >> (uint64(b) % 64)
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opI64ShrU(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	a := vm.stack.data[len(vm.stack.data)-1].int64()
	res := int64(uint64(a) >> (uint64(b) % 64))
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opI64Rotl(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	a := vm.stack.data[len(vm.stack.data)-1].int64()
	res := int64(bits.RotateLeft64(uint64(a), int(b)))
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opI64Rotr(vm *vm, c *callFrame, ip int, in *instr) int {
	b := vm.stack.popInt64()
	a := vm.stack.data[len(vm.stack.data)-1].int64()
	res := int64(bits.RotateLeft64(uint64(a), -int(b)))
	vm.stack.data[len(vm.stack.data)-1] = i64(res)
	return ip + 1
}

func opF32Abs(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(abs(vm.stack.popFloat32()))
	return ip + 1
}

func opF32Neg(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(-vm.stack.popFloat32())
	return ip + 1
}

func opF32Ceil(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(ceil(vm.stack.popFloat32()))
	return ip + 1
}

func opF32Floor(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(floor(vm.stack.popFloat32()))
	return ip + 1
}

func opF32Trunc(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(trunc(vm.stack.popFloat32()))
	return ip + 1
}

func opF32Nearest(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(nearest(vm.stack.popFloat32()))
	return ip + 1
}

func opF32Sqrt(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(sqrt(vm.stack.popFloat32()))
	return ip + 1
}

func opF32Add(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Add(); return ip + 1 }

func opF32Sub(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Sub(); return ip + 1 }

func opF32Mul(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Mul(); return ip + 1 }

func opF32Div(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Div(); return ip + 1 }

func opF32Min(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Min(); return ip + 1 }

func opF32Max(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32Max(); return ip + 1 }

func opF32Copysign(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleF32Copysign()
	return ip + 1
}

func opF64Abs(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(abs(vm.stack.popFloat64()))
	return ip + 1
}

func opF64Neg(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(-vm.stack.popFloat64())
	return ip + 1
}

func opF64Ceil(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(ceil(vm.stack.popFloat64()))
	return ip + 1
}

func opF64Floor(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(floor(vm.stack.popFloat64()))
	return ip + 1
}

func opF64Trunc(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(trunc(vm.stack.popFloat64()))
	return ip + 1
}

func opF64Nearest(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(nearest(vm.stack.popFloat64()))
	return ip + 1
}

func opF64Sqrt(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(sqrt(vm.stack.popFloat64()))
	return ip + 1
}

func opF64Add(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Add(); return ip + 1 }

func opF64Sub(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Sub(); return ip + 1 }

func opF64Mul(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Mul(); return ip + 1 }

func opF64Div(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Div(); return ip + 1 }

func opF64Min(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Min(); return ip + 1 }

func opF64Max(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64Max(); return ip + 1 }

func opF64Copysign(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleF64Copysign()
	return ip + 1
}

func opI32WrapI64(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(int32(vm.stack.popInt64()))
	return ip + 1
}

func opI32TruncF32S(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32TruncF32S(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32TruncF32U(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32TruncF32U(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32TruncF64S(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32TruncF64S(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32TruncF64U(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32TruncF64U(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64ExtendI32S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(int64(vm.stack.popInt32()))
	return ip + 1
}

func opI64ExtendI32U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(int64(uint32(vm.stack.popInt32())))
	return ip + 1
}

func opI64TruncF32S(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64TruncF32S(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64TruncF32U(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64TruncF32U(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64TruncF64S(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64TruncF64S(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64TruncF64U(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64TruncF64U(); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opF32ConvertI32S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(float32(vm.stack.popInt32()))
	return ip + 1
}

func opF32ConvertI32U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(float32(uint32(vm.stack.popInt32())))
	return ip + 1
}

func opF32ConvertI64S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(float32(vm.stack.popInt64()))
	return ip + 1
}

func opF32ConvertI64U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(float32(uint64(vm.stack.popInt64())))
	return ip + 1
}

func opF32DemoteF64(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(float32(vm.stack.popFloat64()))
	return ip + 1
}

func opF64ConvertI32S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(float64(vm.stack.popInt32()))
	return ip + 1
}

func opF64ConvertI32U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(float64(uint32(vm.stack.popInt32())))
	return ip + 1
}

func opF64ConvertI64S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(float64(vm.stack.popInt64()))
	return ip + 1
}

func opF64ConvertI64U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(float64(uint64(vm.stack.popInt64())))
	return ip + 1
}

func opF64PromoteF32(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(float64(vm.stack.popFloat32()))
	return ip + 1
}

func opI32ReinterpretF32(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(int32(math.Float32bits(vm.stack.popFloat32())))
	return ip + 1
}

func opI64ReinterpretF64(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(int64(math.Float64bits(vm.stack.popFloat64())))
	return ip + 1
}

func opF32ReinterpretI32(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat32(math.Float32frombits(uint32(vm.stack.popInt32())))
	return ip + 1
}

func opF64ReinterpretI64(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushFloat64(math.Float64frombits(uint64(vm.stack.popInt64())))
	return ip + 1
}

func opI32Extend8S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(int32(int8(vm.stack.popInt32())))
	return ip + 1
}

func opI32Extend16S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(int32(int16(vm.stack.popInt32())))
	return ip + 1
}

func opI64Extend8S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(int64(int8(vm.stack.popInt64())))
	return ip + 1
}

func opI64Extend16S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(int64(int16(vm.stack.popInt64())))
	return ip + 1
}

func opI64Extend32S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(int64(int32(vm.stack.popInt64())))
	return ip + 1
}

func opI32TruncSatF32S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(truncSatF32SToI32(vm.stack.popFloat32()))
	return ip + 1
}

func opI32TruncSatF32U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(truncSatF32UToI32(vm.stack.popFloat32()))
	return ip + 1
}

func opI32TruncSatF64S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(truncSatF64SToI32(vm.stack.popFloat64()))
	return ip + 1
}

func opI32TruncSatF64U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(truncSatF64UToI32(vm.stack.popFloat64()))
	return ip + 1
}

func opI64TruncSatF32S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(truncSatF32SToI64(vm.stack.popFloat32()))
	return ip + 1
}

func opI64TruncSatF32U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(truncSatF32UToI64(vm.stack.popFloat32()))
	return ip + 1
}

func opI64TruncSatF64S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(truncSatF64SToI64(vm.stack.popFloat64()))
	return ip + 1
}

func opI64TruncSatF64U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt64(truncSatF64UToI64(vm.stack.popFloat64()))
	return ip + 1
}

func opI8x16Swizzle(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI8x16Swizzle()
	return ip + 1
}

func opI8x16Splat(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI8x16Splat(vm.stack.popInt32()))
	return ip + 1
}

func opI16x8Splat(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI16x8Splat(vm.stack.popInt32()))
	return ip + 1
}

func opI32x4Splat(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4Splat(vm.stack.popInt32()))
	return ip + 1
}

func opI64x2Splat(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI64x2Splat(vm.stack.popInt64()))
	return ip + 1
}

func opF32x4Splat(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4Splat(vm.stack.popFloat32()))
	return ip + 1
}

func opF64x2Splat(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2Splat(vm.stack.popFloat64()))
	return ip + 1
}

func opI8x16Eq(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16Eq(); return ip + 1 }

func opI8x16Ne(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16Ne(); return ip + 1 }

func opI8x16LtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16LtS(); return ip + 1 }

func opI8x16LtU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16LtU(); return ip + 1 }

func opI8x16GtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16GtS(); return ip + 1 }

func opI8x16GtU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16GtU(); return ip + 1 }

func opI8x16LeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16LeS(); return ip + 1 }

func opI8x16LeU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16LeU(); return ip + 1 }

func opI8x16GeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16GeS(); return ip + 1 }

func opI8x16GeU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16GeU(); return ip + 1 }

func opI16x8Eq(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8Eq(); return ip + 1 }

func opI16x8Ne(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8Ne(); return ip + 1 }

func opI16x8LtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8LtS(); return ip + 1 }

func opI16x8LtU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8LtU(); return ip + 1 }

func opI16x8GtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8GtS(); return ip + 1 }

func opI16x8GtU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8GtU(); return ip + 1 }

func opI16x8LeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8LeS(); return ip + 1 }

func opI16x8LeU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8LeU(); return ip + 1 }

func opI16x8GeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8GeS(); return ip + 1 }

func opI16x8GeU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8GeU(); return ip + 1 }

func opI32x4Eq(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4Eq(); return ip + 1 }

func opI32x4Ne(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4Ne(); return ip + 1 }

func opI32x4LtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4LtS(); return ip + 1 }

func opI32x4LtU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4LtU(); return ip + 1 }

func opI32x4GtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4GtS(); return ip + 1 }

func opI32x4GtU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4GtU(); return ip + 1 }

func opI32x4LeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4LeS(); return ip + 1 }

func opI32x4LeU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4LeU(); return ip + 1 }

func opI32x4GeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4GeS(); return ip + 1 }

func opI32x4GeU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4GeU(); return ip + 1 }

func opF32x4Eq(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Eq(); return ip + 1 }

func opF32x4Ne(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Ne(); return ip + 1 }

func opF32x4Lt(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Lt(); return ip + 1 }

func opF32x4Gt(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Gt(); return ip + 1 }

func opF32x4Le(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Le(); return ip + 1 }

func opF32x4Ge(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Ge(); return ip + 1 }

func opF64x2Eq(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Eq(); return ip + 1 }

func opF64x2Ne(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Ne(); return ip + 1 }

func opF64x2Lt(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Lt(); return ip + 1 }

func opF64x2Gt(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Gt(); return ip + 1 }

func opF64x2Le(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Le(); return ip + 1 }

func opF64x2Ge(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Ge(); return ip + 1 }

func opV128Not(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdV128Not(vm.stack.popV128()))
	return ip + 1
}

func opV128And(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleV128And(); return ip + 1 }

func opV128Andnot(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleV128Andnot(); return ip + 1 }

func opV128Or(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleV128Or(); return ip + 1 }

func opV128Xor(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleV128Xor(); return ip + 1 }

func opV128Bitselect(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleV128Bitselect()
	return ip + 1
}

func opV128AnyTrue(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(boolToInt32(simdV128AnyTrue(vm.stack.popV128())))
	return ip + 1
}

func opF32x4DemoteF64x2Zero(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4DemoteF64x2Zero(vm.stack.popV128()))
	return ip + 1
}

func opF64x2PromoteLowF32x4(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2PromoteLowF32x4(vm.stack.popV128()))
	return ip + 1
}

func opI8x16Abs(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI8x16Abs(vm.stack.popV128()))
	return ip + 1
}

func opI8x16Neg(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI8x16Neg(vm.stack.popV128()))
	return ip + 1
}

func opI8x16Popcnt(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI8x16Popcnt(vm.stack.popV128()))
	return ip + 1
}

func opI8x16AllTrue(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(boolToInt32(simdI8x16AllTrue(vm.stack.popV128())))
	return ip + 1
}

func opI8x16Bitmask(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(simdI8x16Bitmask(vm.stack.popV128()))
	return ip + 1
}

func opI8x16NarrowI16x8S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI8x16NarrowI16x8S()
	return ip + 1
}

func opI8x16NarrowI16x8U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI8x16NarrowI16x8U()
	return ip + 1
}

func opF32x4Ceil(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4Ceil(vm.stack.popV128()))
	return ip + 1
}

func opF32x4Floor(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4Floor(vm.stack.popV128()))
	return ip + 1
}

func opF32x4Trunc(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4Trunc(vm.stack.popV128()))
	return ip + 1
}

func opF32x4Nearest(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4Nearest(vm.stack.popV128()))
	return ip + 1
}

func opI8x16Shl(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16Shl(); return ip + 1 }

func opI8x16ShrU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16ShrU(); return ip + 1 }

func opI8x16ShrS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16ShrS(); return ip + 1 }

func opI8x16Add(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16Add(); return ip + 1 }

func opI8x16AddSatS(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI8x16AddSatS()
	return ip + 1
}

func opI8x16AddSatU(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI8x16AddSatU()
	return ip + 1
}

func opI8x16Sub(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16Sub(); return ip + 1 }

func opI8x16SubSatS(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI8x16SubSatS()
	return ip + 1
}

func opI8x16SubSatU(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI8x16SubSatU()
	return ip + 1
}

func opF64x2Ceil(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2Ceil(vm.stack.popV128()))
	return ip + 1
}

func opF64x2Floor(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2Floor(vm.stack.popV128()))
	return ip + 1
}

func opI8x16MinS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16MinS(); return ip + 1 }

func opI8x16MinU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16MinU(); return ip + 1 }

func opI8x16MaxS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16MaxS(); return ip + 1 }

func opI8x16MaxU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16MaxU(); return ip + 1 }

func opF64x2Trunc(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2Trunc(vm.stack.popV128()))
	return ip + 1
}

func opI8x16AvgrU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI8x16AvgrU(); return ip + 1 }

func opI16x8ExtaddPairwiseI8x16S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI16x8ExtaddPairwiseI8x16S(vm.stack.popV128()))
	return ip + 1
}

func opI16x8ExtaddPairwiseI8x16U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI16x8ExtaddPairwiseI8x16U(vm.stack.popV128()))
	return ip + 1
}

func opI32x4ExtaddPairwiseI16x8S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4ExtaddPairwiseI16x8S(vm.stack.popV128()))
	return ip + 1
}

func opI32x4ExtaddPairwiseI16x8U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4ExtaddPairwiseI16x8U(vm.stack.popV128()))
	return ip + 1
}

func opI16x8Abs(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI16x8Abs(vm.stack.popV128()))
	return ip + 1
}

func opI16x8Neg(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI16x8Neg(vm.stack.popV128()))
	return ip + 1
}

func opI16x8Q15mulrSatS(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8Q15mulrSatS()
	return ip + 1
}

func opI16x8AllTrue(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(boolToInt32(simdI16x8AllTrue(vm.stack.popV128())))
	return ip + 1
}

func opI16x8Bitmask(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(simdI16x8Bitmask(vm.stack.popV128()))
	return ip + 1
}

func opI16x8NarrowI32x4S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8NarrowI32x4S()
	return ip + 1
}

func opI16x8NarrowI32x4U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8NarrowI32x4U()
	return ip + 1
}

func opI16x8ExtendLowI8x16S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI16x8ExtendLowI8x16S(vm.stack.popV128()))
	return ip + 1
}

func opI16x8ExtendHighI8x16S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI16x8ExtendHighI8x16S(vm.stack.popV128()))
	return ip + 1
}

func opI16x8ExtendLowI8x16U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI16x8ExtendLowI8x16U(vm.stack.popV128()))
	return ip + 1
}

func opI16x8ExtendHighI8x16U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI16x8ExtendHighI8x16U(vm.stack.popV128()))
	return ip + 1
}

func opI16x8Shl(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8Shl(); return ip + 1 }

func opI16x8ShrS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8ShrS(); return ip + 1 }

func opI16x8ShrU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8ShrU(); return ip + 1 }

func opI16x8Add(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8Add(); return ip + 1 }

func opI16x8AddSatS(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8AddSatS()
	return ip + 1
}

func opI16x8AddSatU(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8AddSatU()
	return ip + 1
}

func opI16x8Sub(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8Sub(); return ip + 1 }

func opI16x8SubSatS(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8SubSatS()
	return ip + 1
}

func opI16x8SubSatU(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8SubSatU()
	return ip + 1
}

func opF64x2Nearest(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2Nearest(vm.stack.popV128()))
	return ip + 1
}

func opI16x8Mul(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8Mul(); return ip + 1 }

func opI16x8MinS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8MinS(); return ip + 1 }

func opI16x8MinU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8MinU(); return ip + 1 }

func opI16x8MaxS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8MaxS(); return ip + 1 }

func opI16x8MaxU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8MaxU(); return ip + 1 }

func opI16x8AvgrU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI16x8AvgrU(); return ip + 1 }

func opI16x8ExtmulLowI8x16S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8ExtmulLowI8x16S()
	return ip + 1
}

func opI16x8ExtmulHighI8x16S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8ExtmulHighI8x16S()
	return ip + 1
}

func opI16x8ExtmulLowI8x16U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8ExtmulLowI8x16U()
	return ip + 1
}

func opI16x8ExtmulHighI8x16U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8ExtmulHighI8x16U()
	return ip + 1
}

func opI32x4Abs(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4Abs(vm.stack.popV128()))
	return ip + 1
}

func opI32x4Neg(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4Neg(vm.stack.popV128()))
	return ip + 1
}

func opI32x4AllTrue(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(boolToInt32(simdI32x4AllTrue(vm.stack.popV128())))
	return ip + 1
}

func opI32x4Bitmask(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(simdI32x4Bitmask(vm.stack.popV128()))
	return ip + 1
}

func opI32x4ExtendLowI16x8S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4ExtendLowI16x8S(vm.stack.popV128()))
	return ip + 1
}

func opI32x4ExtendHighI16x8S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4ExtendHighI16x8S(vm.stack.popV128()))
	return ip + 1
}

func opI32x4ExtendLowI16x8U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4ExtendLowI16x8U(vm.stack.popV128()))
	return ip + 1
}

func opI32x4ExtendHighI16x8U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4ExtendHighI16x8U(vm.stack.popV128()))
	return ip + 1
}

func opI32x4Shl(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4Shl(); return ip + 1 }

func opI32x4ShrS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4ShrS(); return ip + 1 }

func opI32x4ShrU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4ShrU(); return ip + 1 }

func opI32x4Add(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4Add(); return ip + 1 }

func opI32x4Sub(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4Sub(); return ip + 1 }

func opI32x4Mul(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4Mul(); return ip + 1 }

func opI32x4MinS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4MinS(); return ip + 1 }

func opI32x4MinU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4MinU(); return ip + 1 }

func opI32x4MaxS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4MaxS(); return ip + 1 }

func opI32x4MaxU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI32x4MaxU(); return ip + 1 }

func opI32x4DotI16x8S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI32x4DotI16x8S()
	return ip + 1
}

func opI32x4ExtmulLowI16x8S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI32x4ExtmulLowI16x8S()
	return ip + 1
}

func opI32x4ExtmulHighI16x8S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI32x4ExtmulHighI16x8S()
	return ip + 1
}

func opI32x4ExtmulLowI16x8U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI32x4ExtmulLowI16x8U()
	return ip + 1
}

func opI32x4ExtmulHighI16x8U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI32x4ExtmulHighI16x8U()
	return ip + 1
}

func opI64x2Abs(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI64x2Abs(vm.stack.popV128()))
	return ip + 1
}

func opI64x2Neg(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI64x2Neg(vm.stack.popV128()))
	return ip + 1
}

func opI64x2AllTrue(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(boolToInt32(simdI64x2AllTrue(vm.stack.popV128())))
	return ip + 1
}

func opI64x2Bitmask(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushInt32(simdI64x2Bitmask(vm.stack.popV128()))
	return ip + 1
}

func opI64x2ExtendLowI32x4S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI64x2ExtendLowI32x4S(vm.stack.popV128()))
	return ip + 1
}

func opI64x2ExtendHighI32x4S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI64x2ExtendHighI32x4S(vm.stack.popV128()))
	return ip + 1
}

func opI64x2ExtendLowI32x4U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI64x2ExtendLowI32x4U(vm.stack.popV128()))
	return ip + 1
}

func opI64x2ExtendHighI32x4U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI64x2ExtendHighI32x4U(vm.stack.popV128()))
	return ip + 1
}

func opI64x2Shl(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2Shl(); return ip + 1 }

func opI64x2ShrS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2ShrS(); return ip + 1 }

func opI64x2ShrU(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2ShrU(); return ip + 1 }

func opI64x2Add(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2Add(); return ip + 1 }

func opI64x2Sub(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2Sub(); return ip + 1 }

func opI64x2Mul(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2Mul(); return ip + 1 }

func opI64x2Eq(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2Eq(); return ip + 1 }

func opI64x2Ne(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2Ne(); return ip + 1 }

func opI64x2LtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2LtS(); return ip + 1 }

func opI64x2GtS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2GtS(); return ip + 1 }

func opI64x2LeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2LeS(); return ip + 1 }

func opI64x2GeS(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleI64x2GeS(); return ip + 1 }

func opI64x2ExtmulLowI32x4S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI64x2ExtmulLowI32x4S()
	return ip + 1
}

func opI64x2ExtmulHighI32x4S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI64x2ExtmulHighI32x4S()
	return ip + 1
}

func opI64x2ExtmulLowI32x4U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI64x2ExtmulLowI32x4U()
	return ip + 1
}

func opI64x2ExtmulHighI32x4U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI64x2ExtmulHighI32x4U()
	return ip + 1
}

func opF32x4Abs(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4Abs(vm.stack.popV128()))
	return ip + 1
}

func opF32x4Neg(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4Neg(vm.stack.popV128()))
	return ip + 1
}

func opF32x4Sqrt(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4Sqrt(vm.stack.popV128()))
	return ip + 1
}

func opF32x4Add(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Add(); return ip + 1 }

func opF32x4Sub(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Sub(); return ip + 1 }

func opF32x4Mul(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Mul(); return ip + 1 }

func opF32x4Div(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Div(); return ip + 1 }

func opF32x4Min(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Min(); return ip + 1 }

func opF32x4Max(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Max(); return ip + 1 }

func opF32x4Pmin(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Pmin(); return ip + 1 }

func opF32x4Pmax(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF32x4Pmax(); return ip + 1 }

func opF64x2Abs(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2Abs(vm.stack.popV128()))
	return ip + 1
}

func opF64x2Neg(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2Neg(vm.stack.popV128()))
	return ip + 1
}

func opF64x2Sqrt(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2Sqrt(vm.stack.popV128()))
	return ip + 1
}

func opF64x2Add(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Add(); return ip + 1 }

func opF64x2Sub(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Sub(); return ip + 1 }

func opF64x2Mul(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Mul(); return ip + 1 }

func opF64x2Div(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Div(); return ip + 1 }

func opF64x2Min(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Min(); return ip + 1 }

func opF64x2Max(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Max(); return ip + 1 }

func opF64x2Pmin(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Pmin(); return ip + 1 }

func opF64x2Pmax(vm *vm, c *callFrame, ip int, in *instr) int { vm.handleF64x2Pmax(); return ip + 1 }

func opI32x4TruncSatF32x4S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4TruncSatF32x4S(vm.stack.popV128()))
	return ip + 1
}

func opI32x4TruncSatF32x4U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4TruncSatF32x4U(vm.stack.popV128()))
	return ip + 1
}

func opF32x4ConvertI32x4S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4ConvertI32x4S(vm.stack.popV128()))
	return ip + 1
}

func opF32x4ConvertI32x4U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF32x4ConvertI32x4U(vm.stack.popV128()))
	return ip + 1
}

func opI32x4TruncSatF64x2SZero(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4TruncSatF64x2SZero(vm.stack.popV128()))
	return ip + 1
}

func opI32x4TruncSatF64x2UZero(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdI32x4TruncSatF64x2UZero(vm.stack.popV128()))
	return ip + 1
}

func opF64x2ConvertLowI32x4S(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2ConvertLowI32x4S(vm.stack.popV128()))
	return ip + 1
}

func opF64x2ConvertLowI32x4U(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(simdF64x2ConvertLowI32x4U(vm.stack.popV128()))
	return ip + 1
}

func opI32Load(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32Load(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Load(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Load(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opF32Load(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleF32Load(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opF64Load(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleF64Load(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32Load8S(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32Load8S(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32Load8U(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32Load8U(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32Load16S(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32Load16S(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32Load16U(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32Load16U(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Load8S(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Load8S(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Load8U(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Load8U(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Load16S(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Load16S(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Load16U(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Load16U(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Load32S(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Load32S(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Load32U(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Load32U(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32Store(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32Store(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Store(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Store(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opF32Store(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleF32Store(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opF64Store(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleF64Store(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32Store8(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32Store8(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI32Store16(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI32Store16(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Store8(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Store8(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Store16(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Store16(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI64Store32(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleI64Store32(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opV128Load(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleV128Load(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opV128Store(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleV128Store(c, uint32(in.a), uint32(in.b)); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opI8x16ExtractLaneS(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI8x16ExtractLaneS(uint32(in.a))
	return ip + 1
}

func opI8x16ExtractLaneU(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI8x16ExtractLaneU(uint32(in.a))
	return ip + 1
}

func opI16x8ExtractLaneS(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8ExtractLaneS(uint32(in.a))
	return ip + 1
}

func opI16x8ExtractLaneU(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8ExtractLaneU(uint32(in.a))
	return ip + 1
}

func opI32x4ExtractLane(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI32x4ExtractLane(uint32(in.a))
	return ip + 1
}

func opI64x2ExtractLane(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI64x2ExtractLane(uint32(in.a))
	return ip + 1
}

func opF32x4ExtractLane(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleF32x4ExtractLane(uint32(in.a))
	return ip + 1
}

func opF64x2ExtractLane(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleF64x2ExtractLane(uint32(in.a))
	return ip + 1
}

func opI8x16ReplaceLane(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI8x16ReplaceLane(uint32(in.a))
	return ip + 1
}

func opI16x8ReplaceLane(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI16x8ReplaceLane(uint32(in.a))
	return ip + 1
}

func opI32x4ReplaceLane(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI32x4ReplaceLane(uint32(in.a))
	return ip + 1
}

func opI64x2ReplaceLane(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleI64x2ReplaceLane(uint32(in.a))
	return ip + 1
}

func opF32x4ReplaceLane(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleF32x4ReplaceLane(uint32(in.a))
	return ip + 1
}

func opF64x2ReplaceLane(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.handleF64x2ReplaceLane(uint32(in.a))
	return ip + 1
}

func opV128Load8Lane(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleSimdLoadLane(c, uint32(in.a), uint32(in.b>>32), uint32(in.b), 8); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opV128Load16Lane(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleSimdLoadLane(c, uint32(in.a), uint32(in.b>>32), uint32(in.b), 16); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opV128Load32Lane(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleSimdLoadLane(c, uint32(in.a), uint32(in.b>>32), uint32(in.b), 32); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opV128Load64Lane(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleSimdLoadLane(c, uint32(in.a), uint32(in.b>>32), uint32(in.b), 64); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opV128Store8Lane(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleSimdStoreLane(c, uint32(in.a), uint32(in.b>>32), uint32(in.b), 8); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opV128Store16Lane(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleSimdStoreLane(c, uint32(in.a), uint32(in.b>>32), uint32(in.b), 16); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opV128Store32Lane(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleSimdStoreLane(c, uint32(in.a), uint32(in.b>>32), uint32(in.b), 32); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opV128Store64Lane(vm *vm, c *callFrame, ip int, in *instr) int {
	if err := vm.handleSimdStoreLane(c, uint32(in.a), uint32(in.b>>32), uint32(in.b), 64); err != nil {
		c.trap = err
		return halt
	}
	return ip + 1
}

func opV128Load8x8S(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 8)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdV128Load8x8S(data))
	return ip + 1
}

func opV128Load8x8U(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 8)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdV128Load8x8U(data))
	return ip + 1
}

func opV128Load16x4S(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 8)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdV128Load16x4S(data))
	return ip + 1
}

func opV128Load16x4U(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 8)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdV128Load16x4U(data))
	return ip + 1
}

func opV128Load32x2S(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 8)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdV128Load32x2S(data))
	return ip + 1
}

func opV128Load32x2U(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 8)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdV128Load32x2U(data))
	return ip + 1
}

func opV128Load8Splat(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 1)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdI8x16SplatFromBytes(data))
	return ip + 1
}

func opV128Load16Splat(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 2)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdI16x8SplatFromBytes(data))
	return ip + 1
}

func opV128Load32Splat(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 4)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdI32x4SplatFromBytes(data))
	return ip + 1
}

func opV128Load64Splat(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 8)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdI64x2SplatFromBytes(data))
	return ip + 1
}

func opV128Load32Zero(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 4)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdV128Load32Zero(data))
	return ip + 1
}

func opV128Load64Zero(vm *vm, c *callFrame, ip int, in *instr) int {
	data, err := vm.memGet(c, uint32(in.a), uint32(in.b), 8)
	if err != nil {
		c.trap = err
		return halt
	}
	vm.stack.pushV128(simdV128Load64Zero(data))
	return ip + 1
}

func opV128Const(vm *vm, c *callFrame, ip int, in *instr) int {
	vm.stack.pushV128(V128Value{Low: in.a, High: in.b})
	return ip + 1
}

func opI8x16Shuffle(vm *vm, c *callFrame, ip int, in *instr) int {
	v2 := vm.stack.popV128()
	v1 := vm.stack.popV128()
	vm.stack.pushV128(simdI8x16Shuffle(v1, v2, byte(in.a>>0), byte(in.a>>8), byte(in.a>>16), byte(in.a>>24), byte(in.a>>32), byte(in.a>>40), byte(in.a>>48), byte(in.a>>56), byte(in.b>>0), byte(in.b>>8), byte(in.b>>16), byte(in.b>>24), byte(in.b>>32), byte(in.b>>40), byte(in.b>>48), byte(in.b>>56)))
	return ip + 1
}

func (vm *vm) getInputCount(module *ModuleInstance, blockType int32) uint32 {
	// Empty (-0x40) and value-type block types both consume no inputs.
	if blockType < 0 {
		return 0
	}
	return uint32(len(module.types[blockType].ParamTypes))
}

func (vm *vm) getOutputCount(module *ModuleInstance, blockType int32) uint32 {
	if blockType == -0x40 { // empty block type.
		return 0
	}

	if blockType >= 0 { // type index.
		funcType := module.types[blockType]
		return uint32(len(funcType.ResultTypes))
	}

	return 1 // value type.
}

func (vm *vm) resolveExports(
	module *moduleDefinition,
	instance *ModuleInstance,
) []exportInstance {
	exports := []exportInstance{}
	for _, export := range module.exports {
		var value any
		switch export.indexType {
		case functionExportKind:
			storeIndex := instance.funcAddrs[export.index]
			value = vm.store.funcs[storeIndex]
		case globalExportKind:
			storeIndex := instance.globalAddrs[export.index]
			value = vm.store.globals[storeIndex]
		case memoryExportKind:
			storeIndex := instance.memAddrs[export.index]
			value = vm.store.memories[storeIndex]
		case tableExportKind:
			storeIndex := instance.tableAddrs[export.index]
			value = vm.store.tables[storeIndex]
		}
		exports = append(exports, exportInstance{name: export.name, value: value})
	}
	return exports
}

func (vm *vm) invokeHostFunction(fun *hostFunction) (err error) {
	defer func() {
		if r := recover(); r != nil {
			var panicErr error
			switch v := r.(type) {
			case error:
				panicErr = v
			default:
				panicErr = fmt.Errorf("panic: %v", v)
			}
			err = panicErr
		}
	}()

	// popValueTypes returns a fresh slice per call, so the host may retain args.
	args := vm.stack.popValueTypes(fun.GetType().ParamTypes)
	res := fun.hostCode(fun.module, args...)
	if len(res) != len(fun.GetType().ResultTypes) {
		return errHostResultCountMismatch
	}

	vm.stack.pushAll(res)
	return err
}

func (vm *vm) invokeExpression(
	expression []uint64,
	resultType ValueType,
	moduleInstance *ModuleInstance,
) (value, error) {
	// We create a fake function to execute the expression. The expression is
	// expected to return a single value.
	function := wasmFunction{
		functionType: FunctionType{
			ParamTypes:  []ValueType{},
			ResultTypes: []ValueType{resultType},
		},
		code: function{
			body:      expression,
			typeIndex: 0xffffffff, // Represents -1. See getOutputCount.
		},
		module: moduleInstance,
	}
	if err := vm.compile(&function); err != nil {
		return value{}, err
	}
	if err := vm.invokeWasmFunction(&function); err != nil {
		return value{}, err
	}
	return vm.stack.pop(), nil
}

func (vm *vm) applyActiveElementSegment(
	segment elementSegment,
	elem *elementInstance,
	moduleInstance *ModuleInstance,
) error {
	offsetExpression := segment.offsetExpression
	offsetVal, err := vm.invokeExpression(offsetExpression, I32, moduleInstance)
	if err != nil {
		return err
	}
	table := vm.store.tables[moduleInstance.tableAddrs[segment.tableIndex]]
	return table.InitFromSlice(offsetVal.int32(), elem.functionIndexes)
}

func (vm *vm) applyActiveDataSegment(
	segment dataSegment,
	data *dataInstance,
	moduleInstance *ModuleInstance,
) error {
	offsetExpression := segment.offsetExpression
	offsetVal, err := vm.invokeExpression(offsetExpression, I32, moduleInstance)
	if err != nil {
		return err
	}
	memory := vm.store.memories[moduleInstance.memAddrs[segment.memoryIndex]]
	return memory.Set(uint32(offsetVal.int32()), 0, data.content)
}

func (vm *vm) newElementInstance(
	segment elementSegment,
	moduleInstance *ModuleInstance,
) (elementInstance, error) {
	var indexes []int32
	switch {
	case len(segment.functionIndexes) > 0:
		indexes = toStoreFuncIndexes(moduleInstance, segment.functionIndexes)
	case len(segment.functionIndexesExpressions) > 0:
		indexes = make([]int32, len(segment.functionIndexesExpressions))
		for i, expr := range segment.functionIndexesExpressions {
			refVal, err := vm.invokeExpression(expr, segment.kind, moduleInstance)
			if err != nil {
				return elementInstance{}, err
			}
			indexes[i] = refVal.int32()
		}
	}
	return elementInstance{functionIndexes: indexes}, nil
}

func toStoreFuncIndexes(
	moduleInstance *ModuleInstance,
	localIndexes []int32,
) []int32 {
	storeIndices := make([]int32, len(localIndexes))
	for i, localIndex := range localIndexes {
		storeIndices[i] = int32(moduleInstance.funcAddrs[localIndex])
	}
	return storeIndices
}

func (vm *vm) getTable(frame *callFrame, localIndex uint64) *Table {
	tableIndex := frame.module.tableAddrs[localIndex]
	return vm.store.tables[tableIndex]
}

func (vm *vm) getMemory(frame *callFrame, localIndex uint64) *Memory {
	memoryIndex := frame.module.memAddrs[localIndex]
	return vm.store.memories[memoryIndex]
}

func (vm *vm) getGlobal(frame *callFrame, localIndex uint64) *Global {
	globalIndex := frame.module.globalAddrs[localIndex]
	return vm.store.globals[globalIndex]
}

func (vm *vm) getElement(frame *callFrame, localIndex uint64) *elementInstance {
	elementIndex := frame.module.elemAddrs[localIndex]
	return &vm.store.elements[elementIndex]
}

func (vm *vm) getData(frame *callFrame, localIndex uint64) *dataInstance {
	dataIndex := frame.module.dataAddrs[localIndex]
	return &vm.store.datas[dataIndex]
}
