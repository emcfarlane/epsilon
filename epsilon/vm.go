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
)

var (
	errUnreachable              = errors.New("unreachable")
	errCallStackExhausted       = errors.New("call stack exhausted")
	errFuelExhausted            = errors.New("fuel exhausted")
	errUnknownFunctionType      = errors.New("unknown function type")
	errIndirectCallTypeMismatch = errors.New("indirect call type mismatch")
	errHostResultCountMismatch  = errors.New("host func result count mismatch")
	errUnsupportedOpcode        = errors.New("function uses an opcode the compiler does not support")
)

const (
	controlStackCacheSlotSize = 14 // Control stack slot size per call frame.
	localsCacheSlotSize       = 12 // Locals slot size per call frame.
)

// store represents all global state that can be manipulated by the vm. It
// consists of the runtime representation of all instances of functions,
// tables, memories, globals, element segments, and data segments that have
// been allocated during the vm life time.
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

// callFrame tracks the per-call state the closure interpreter needs. Call depth
// (len(callStack)) drives the locals/control/context caches and the recursion
// limit; runClosureLoop reads locals and module from the top frame.
type callFrame struct {
	locals []value
	module *ModuleInstance
}

// vm is the WebAssembly Virtual Machine.
type vm struct {
	store            *store
	stack            *valueStack
	callStack        []callFrame
	closureCtrlCache []closureControl
	closureCtxCache  []closureCtx
	localsCache      []value
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
		store:            &store{},
		stack:            newValueStack(),
		callStack:        make([]callFrame, 0, config.CallStackPreallocationSize),
		closureCtrlCache: make([]closureControl, ctrlCacheSize),
		closureCtxCache:  make([]closureCtx, config.CallStackPreallocationSize),
		localsCache:      make([]value, localsCacheSize),
		config:           config,
		fuel:             config.Fuel,
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

	// Allocate runtime data instances. The byte slice is shared with the
	// parsed dataSegment: the runtime only ever reads it (memory.init copies
	// out) and data.drop nils the instance's slice header, never the backing
	// array, so the moduleDefinition is left untouched and can be reinstantiated.
	for _, segment := range module.dataSegments {
		storeIndex := uint32(len(vm.store.datas))
		moduleInstance.dataAddrs = append(moduleInstance.dataAddrs, storeIndex)
		data := dataInstance{content: segment.content}
		vm.store.datas = append(vm.store.datas, data)
	}

	// Apply active data segments to their target memories, then drop them. After
	// this, memory.init against an active segment sees an empty content and
	// traps for any non-zero size.
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
	if len(vm.callStack) >= vm.config.MaxCallStackDepth {
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
		// Cache slots may hold stale values from earlier calls, and WASM allows
		// reading uninitialized locals, so zero the non-parameter locals. The
		// parameter slots are filled from the operand stack afterward.
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

	vm.callStack = append(vm.callStack, callFrame{
		locals: locals,
		module: function.module,
	})

	vm.ensureCompiled(function)
	if !function.closuresOK {
		vm.callStack = vm.callStack[:len(vm.callStack)-1]
		vm.localsTop = localsMark
		return errUnsupportedOpcode
	}

	// Two loop variants so the no-fuel path pays nothing for the fuel check.
	arity := uint32(len(function.functionType.ResultTypes))
	var err error
	if vm.config.EnableFuel {
		err = vm.runClosureLoopWithFuel(function.closures, arity)
	} else {
		err = vm.runClosureLoop(function.closures, arity)
	}
	// The loop popped this frame; rewind the locals cursor to free its slots.
	vm.localsTop = localsMark
	return err
}

func (vm *vm) handleSelect() {
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
}

func (vm *vm) handleBinaryFloat32(op func(a, b float32) float32) {
	b := vm.stack.popFloat32()
	a := vm.stack.popFloat32()
	vm.stack.pushFloat32(op(a, b))
}

func (vm *vm) handleBinaryFloat64(op func(a, b float64) float64) {
	b := vm.stack.popFloat64()
	a := vm.stack.popFloat64()
	vm.stack.pushFloat64(op(a, b))
}

func (vm *vm) handleBinaryV128(op func(a, b V128Value) V128Value) {
	b := vm.stack.popV128()
	a := vm.stack.popV128()
	vm.stack.pushV128(op(a, b))
}

func (vm *vm) handleBinarySafeInt32(op func(a, b int32) (int32, error)) error {
	b := vm.stack.popInt32()
	a := vm.stack.popInt32()
	result, err := op(a, b)
	if err != nil {
		return err
	}
	vm.stack.pushInt32(result)
	return nil
}

func (vm *vm) handleBinarySafeInt64(op func(a, b int64) (int64, error)) error {
	b := vm.stack.popInt64()
	a := vm.stack.popInt64()
	result, err := op(a, b)
	if err != nil {
		return err
	}
	vm.stack.pushInt64(result)
	return nil
}

func (vm *vm) handleBinaryBoolInt32(op func(a, b int32) bool) {
	b := vm.stack.popInt32()
	a := vm.stack.popInt32()
	vm.stack.pushInt32(boolToInt32(op(a, b)))
}

func (vm *vm) handleBinaryBoolInt64(op func(a, b int64) bool) {
	b := vm.stack.popInt64()
	a := vm.stack.popInt64()
	vm.stack.pushInt32(boolToInt32(op(a, b)))
}

func (vm *vm) handleBinaryBoolFloat32(op func(a, b float32) bool) {
	b := vm.stack.popFloat32()
	a := vm.stack.popFloat32()
	vm.stack.pushInt32(boolToInt32(op(a, b)))
}

func (vm *vm) handleBinaryBoolFloat64(op func(a, b float64) bool) {
	b := vm.stack.popFloat64()
	a := vm.stack.popFloat64()
	vm.stack.pushInt32(boolToInt32(op(a, b)))
}

func (vm *vm) handleUnarySafeFloat32(op func(a float32) (int32, error)) error {
	a := vm.stack.popFloat32()
	result, err := op(a)
	if err != nil {
		return err
	}
	vm.stack.pushInt32(result)
	return nil
}

func (vm *vm) handleUnarySafeFloat64(op func(a float64) (int32, error)) error {
	a := vm.stack.popFloat64()
	result, err := op(a)
	if err != nil {
		return err
	}
	vm.stack.pushInt32(result)
	return nil
}

func (vm *vm) handleTruncFloat32Int64(op func(a float32) (int64, error)) error {
	a := vm.stack.popFloat32()
	result, err := op(a)
	if err != nil {
		return err
	}
	vm.stack.pushInt64(result)
	return nil
}

func (vm *vm) handleTruncFloat64Int64(op func(a float64) (int64, error)) error {
	a := vm.stack.popFloat64()
	result, err := op(a)
	if err != nil {
		return err
	}
	vm.stack.pushInt64(result)
	return nil
}

func (vm *vm) handleSimdShift(op func(v V128Value, shift int32) V128Value) {
	shift := vm.stack.popInt32()
	v := vm.stack.popV128()
	vm.stack.pushV128(op(v, shift))
}

func (vm *vm) handleSimdTernary(op func(v1, v2, v3 V128Value) V128Value) {
	v3 := vm.stack.popV128()
	v2 := vm.stack.popV128()
	v1 := vm.stack.popV128()
	vm.stack.pushV128(op(v1, v2, v3))
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
	// We create a fake function to execute the expression.
	// The expression is expected to return a single value.
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

func (vm *vm) getTable(module *ModuleInstance, localIndex uint64) *Table {
	tableIndex := module.tableAddrs[localIndex]
	return vm.store.tables[tableIndex]
}

func (vm *vm) getMemory(module *ModuleInstance, localIndex uint64) *Memory {
	memoryIndex := module.memAddrs[localIndex]
	return vm.store.memories[memoryIndex]
}

func (vm *vm) getGlobal(module *ModuleInstance, localIndex uint64) *Global {
	globalIndex := module.globalAddrs[localIndex]
	return vm.store.globals[globalIndex]
}

func (vm *vm) getElement(module *ModuleInstance, localIndex uint64) *elementInstance {
	elementIndex := module.elemAddrs[localIndex]
	return &vm.store.elements[elementIndex]
}

func (vm *vm) getData(module *ModuleInstance, localIndex uint64) *dataInstance {
	dataIndex := module.dataAddrs[localIndex]
	return &vm.store.datas[dataIndex]
}
