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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ziggy42/epsilon/internal/wabt"
)

// dispatchTestModules exercise the closure path alongside the control flow
// (loops, branches, if/else, br_table, calls) whose instruction-index remapping
// the compiler must get right. want is a pure-Go oracle for the exported
// function, used as an independent correctness reference.
var dispatchTestModules = []struct {
	name   string
	wat    string
	export string
	args   []any
	want   func(int32) int32
}{
	{
		name: "sum_loop",
		wat: `(module (func (export "f") (param $n i32) (result i32)
			(local $i i32) (local $acc i32)
			(block $exit
			  (loop $top
			    (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
			    (local.set $acc (i32.add (local.get $acc) (local.get $i)))
			    (local.set $i (i32.add (local.get $i) (i32.const 1)))
			    (br $top)))
			(local.get $acc)))`,
		export: "f",
		args:   []any{int32(1000)},
		want: func(n int32) int32 {
			var acc int32
			for i := int32(0); i < n; i++ {
				acc += i
			}
			return acc
		},
	},
	{
		name: "branchy",
		wat: `(module (func (export "f") (param $x i32) (result i32)
			(local $y i32)
			(if (result i32) (i32.gt_s (local.get $x) (i32.const 10))
			  (then (local.set $y (i32.add (local.get $x) (i32.const 5)))
			        (i32.add (local.get $y) (local.get $x)))
			  (else (i32.add (local.get $x) (i32.const 100))))))`,
		export: "f",
		args:   []any{int32(42)},
		want: func(x int32) int32 {
			if x > 10 {
				return (x + 5) + x
			}
			return x + 100
		},
	},
	{
		name: "br_table",
		wat: `(module (func (export "f") (param $x i32) (result i32)
			(local $r i32)
			(block $b2 (block $b1 (block $b0
			  (br_table $b0 $b1 $b2 (local.get $x)))
			  (local.set $r (i32.add (local.get $x) (i32.const 1))) (br $b2))
			  (local.set $r (i32.add (local.get $x) (i32.const 2))) (br $b2))
			(i32.add (local.get $r) (i32.const 1000))))`,
		export: "f",
		args:   []any{int32(1)},
		want: func(x int32) int32 {
			var r int32
			switch uint32(x) {
			case 0:
				r = x + 1
			case 1:
				r = x + 2
			default: // label $b2: r stays 0
			}
			return r + 1000
		},
	},
	{
		name: "calls",
		wat: `(module
			(func $add (param i32 i32) (result i32)
			  (i32.add (local.get 0) (local.get 1)))
			(func (export "f") (param $n i32) (result i32)
			  (local $i i32) (local $acc i32)
			  (block $exit (loop $top
			    (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
			    (local.set $acc (call $add (local.get $acc) (local.get $i)))
			    (local.set $i (i32.add (local.get $i) (i32.const 1)))
			    (br $top)))
			  (local.get $acc)))`,
		export: "f",
		args:   []any{int32(1000)},
		want: func(n int32) int32 {
			var acc int32
			for i := int32(0); i < n; i++ {
				acc += i
			}
			return acc
		},
	},
}

// TestDispatchCorrectness checks closure-dispatch results against an independent
// Go oracle across an argument sweep that covers both branch directions.
func TestDispatchCorrectness(t *testing.T) {
	for _, tc := range dispatchTestModules {
		t.Run(tc.name, func(t *testing.T) {
			wasm, err := wabt.Wat2Wasm(tc.wat)
			if err != nil {
				t.Fatalf("wat2wasm: %v", err)
			}
			instance, err := NewRuntime().InstantiateModuleFromBytes(wasm)
			if err != nil {
				t.Fatalf("instantiate: %v", err)
			}
			for _, x := range []int32{-5, 0, 1, 2, 11, 42, 257} {
				res, err := instance.Invoke(tc.export, x)
				if err != nil {
					t.Fatalf("x=%d: invoke: %v", x, err)
				}
				if got := res[0].(int32); got != tc.want(x) {
					t.Errorf("x=%d: got %d, want %d", x, got, tc.want(x))
				}
			}
		})
	}
}

// TestClosureCompilesSupported proves the workloads compile to closures.
func TestClosureCompilesSupported(t *testing.T) {
	for _, tc := range dispatchTestModules {
		t.Run(tc.name, func(t *testing.T) {
			wasm, err := wabt.Wat2Wasm(tc.wat)
			if err != nil {
				t.Fatalf("wat2wasm: %v", err)
			}
			def, err := newParser(strings.NewReader(string(wasm)), DefaultConfig()).parse()
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			vm := newVm(DefaultConfig())
			module := &ModuleInstance{types: def.types}
			for i := range def.funcs {
				if _, ok := vm.compileClosures(&def.funcs[i], module); !ok {
					t.Fatalf("func %d did not compile to closures", i)
				}
			}
		})
	}
}

// TestClosureFullCoverage walks the real WASI test corpus and asserts that every
// function in every parseable module compiles fully to closures (no opcode is
// unsupported).
func TestClosureFullCoverage(t *testing.T) {
	var files []string
	_ = filepath.Walk("../wasip1/wasi-testsuite", func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && strings.HasSuffix(path, ".wasm") {
			files = append(files, path)
		}
		return nil
	})
	if len(files) == 0 {
		t.Skip("no corpus .wasm files found")
	}

	vm := newVm(DefaultConfig())
	var parsed, functions int
	for _, path := range files {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		def, err := newParser(strings.NewReader(string(data)), DefaultConfig()).parse()
		if err != nil {
			continue // modules using features the parser rejects are out of scope
		}
		parsed++
		module := &ModuleInstance{types: def.types}
		for i := range def.funcs {
			functions++
			if _, ok := vm.compileClosures(&def.funcs[i], module); !ok {
				t.Fatalf("%s func %d uses an unsupported opcode", path, i)
			}
		}
	}
	t.Logf("full coverage: %d modules, %d functions all compiled to closures", parsed, functions)
}

// Benchmarks for the closure dispatch loop. Compare against the switch
// interpreter by running the same benchmark on the `main` branch.
func benchDispatch(b *testing.B, wat, export string, args ...any) {
	wasm, err := wabt.Wat2Wasm(wat)
	if err != nil {
		b.Fatalf("wat2wasm: %v", err)
	}
	instance, err := NewRuntime().InstantiateModuleFromBytes(wasm)
	if err != nil {
		b.Fatalf("instantiate: %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := instance.Invoke(export, args...); err != nil {
			b.Fatalf("invoke: %v", err)
		}
	}
}

const benchFibWat = `(module (func $fib (export "fib") (param $n i32) (result i32)
	(if (result i32) (i32.lt_s (local.get $n) (i32.const 2))
	  (then (local.get $n))
	  (else (i32.add
	    (call $fib (i32.sub (local.get $n) (i32.const 1)))
	    (call $fib (i32.sub (local.get $n) (i32.const 2))))))))`

func BenchmarkDispatchLoop(b *testing.B) {
	benchDispatch(b, dispatchTestModules[0].wat, "f", int32(100000))
}
func BenchmarkDispatchFib(b *testing.B) {
	benchDispatch(b, benchFibWat, "fib", int32(30))
}
