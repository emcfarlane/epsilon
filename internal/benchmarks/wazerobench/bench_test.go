// Copyright 2026 Google LLC
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

// Opt-in comparison of epsilon against wazero's compiler and interpreter on the
// shared benchmark corpus. Each workload runs the same .wasm with the same
// arguments under all three runtimes, emitting sub-benchmarks
// (".../epsilon", ".../wazero-compiler", ".../wazero-interpreter") so a single
// run shows how close epsilon is to wazero.
//
// This lives in its own Go module so wazero never enters the root module. Run:
//
//	make bench-wazero                 # from the repo root
//	go test -bench=. -benchmem .      # from this directory
package wazerobench

import (
	"bytes"
	"context"
	"os"
	"testing"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/ziggy42/epsilon/epsilon"
)

const wasmDir = "../wasm/"

// invokeBenchmarks mirrors the epsilon benchmark suite (internal/benchmarks):
// same .wasm files, exports, and arguments, so the numbers are comparable.
var invokeBenchmarks = []struct {
	name   string
	wasm   string
	fn     string
	epArgs []any
	wzArgs []uint64
}{
	{"FactorialRecursive", "factorial.wasm", "fac_recursive",
		[]any{int32(1000), int64(25)},
		[]uint64{api.EncodeI32(1000), api.EncodeI64(25)}},
	{"FactorialIterative", "factorial.wasm", "fac_iterative",
		[]any{int32(10000), int64(25)},
		[]uint64{api.EncodeI32(10000), api.EncodeI64(25)}},
	{"FibonacciRecursive", "fibonacci.wasm", "fib_recursive",
		[]any{int32(1), int32(25)},
		[]uint64{api.EncodeI32(1), api.EncodeI32(25)}},
	{"FibonacciIterative", "fibonacci.wasm", "fib_iterative",
		[]any{int32(10000), int32(25)},
		[]uint64{api.EncodeI32(10000), api.EncodeI32(25)}},
	{"Indirect", "indirect.wasm", "run_indirect_calls",
		[]any{int32(10000)}, []uint64{api.EncodeI32(10000)}},
	{"MatrixMultiplication", "matrix_multiplication.wasm",
		"run_matrix_multiplication",
		[]any{int32(100)}, []uint64{api.EncodeI32(100)}},
	{"VectorMath", "vector_math.wasm", "compute_vector_math",
		[]any{int32(100)}, []uint64{api.EncodeI32(100)}},
	{"IntegerVectorMath", "vector_math.wasm", "compute_integer_vector_math",
		[]any{int32(100)}, []uint64{api.EncodeI32(100)}},
	{"MemoryAccess", "memory_access.wasm", "run_memcpy",
		[]any{int32(100)}, []uint64{api.EncodeI32(100)}},
	{"TrigonometrySin", "trigonometry.wasm", "compute_sin",
		[]any{float32(42.7)}, []uint64{api.EncodeF32(42.7)}},
	{"SortingBubbleSort", "sorting.wasm", "bubble_sort", nil, nil},
	{"SortingMergeSort", "sorting.wasm", "merge_sort", nil, nil},
	{"SortingQuickSort", "sorting.wasm", "quick_sort", nil, nil},
	{"SHA256", "sha256.wasm", "run_sha256",
		[]any{int32(64)}, []uint64{api.EncodeI32(64)}},
}

func BenchmarkInvoke(b *testing.B) {
	for _, w := range invokeBenchmarks {
		wasm := readWasm(b, w.wasm)
		b.Run(w.name, func(b *testing.B) {
			b.Run("epsilon", func(b *testing.B) {
				inst, err := epsilon.NewRuntime().InstantiateModuleFromBytes(wasm)
				if err != nil {
					b.Fatalf("epsilon instantiate: %v", err)
				}
				for b.Loop() {
					if _, err := inst.Invoke(w.fn, w.epArgs...); err != nil {
						b.Fatalf("epsilon invoke: %v", err)
					}
				}
			})
			benchWazero(b, "wazero-compiler",
				wazero.NewRuntimeConfigCompiler(), wasm, w.fn, w.wzArgs, nil)
			benchWazero(b, "wazero-interpreter",
				wazero.NewRuntimeConfigInterpreter(), wasm, w.fn, w.wzArgs, nil)
		})
	}
}

func BenchmarkHostCall(b *testing.B) {
	wasm := readWasm(b, "host_call.wasm")
	const fn = "run_host_calls"
	args := []any{int32(10000)}
	wzArgs := []uint64{api.EncodeI32(10000)}

	b.Run("epsilon", func(b *testing.B) {
		imports := epsilon.NewModuleImports("env").
			AddHostFunc("noop",
				func(_ *epsilon.ModuleInstance, a ...any) []any {
					return []any{a[0]}
				})
		inst, err := epsilon.NewRuntime().
			InstantiateModuleWithImports(bytes.NewReader(wasm), imports)
		if err != nil {
			b.Fatalf("epsilon instantiate: %v", err)
		}
		for b.Loop() {
			if _, err := inst.Invoke(fn, args...); err != nil {
				b.Fatalf("epsilon invoke: %v", err)
			}
		}
	})

	registerNoop := func(ctx context.Context, r wazero.Runtime, b *testing.B) {
		_, err := r.NewHostModuleBuilder("env").
			NewFunctionBuilder().
			WithFunc(func(x uint32) uint32 { return x }).
			Export("noop").
			Instantiate(ctx)
		if err != nil {
			b.Fatalf("wazero host module: %v", err)
		}
	}
	benchWazero(b, "wazero-compiler",
		wazero.NewRuntimeConfigCompiler(), wasm, fn, wzArgs, registerNoop)
	benchWazero(b, "wazero-interpreter",
		wazero.NewRuntimeConfigInterpreter(), wasm, fn, wzArgs, registerNoop)
}

// benchWazero runs fn under a wazero runtime built from cfg, optionally
// registering host imports first.
func benchWazero(
	b *testing.B,
	name string,
	cfg wazero.RuntimeConfig,
	wasm []byte,
	fn string,
	args []uint64,
	host func(context.Context, wazero.Runtime, *testing.B),
) {
	b.Run(name, func(b *testing.B) {
		ctx := context.Background()
		runtime := wazero.NewRuntimeWithConfig(ctx, cfg)
		defer runtime.Close(ctx)
		if host != nil {
			host(ctx, runtime, b)
		}
		mod, err := runtime.Instantiate(ctx, wasm)
		if err != nil {
			b.Fatalf("wazero instantiate: %v", err)
		}
		call := mod.ExportedFunction(fn)
		if call == nil {
			b.Fatalf("wazero export %q not found", fn)
		}
		for b.Loop() {
			if _, err := call.Call(ctx, args...); err != nil {
				b.Fatalf("wazero call: %v", err)
			}
		}
	})
}

func readWasm(b *testing.B, name string) []byte {
	b.Helper()
	wasm, err := os.ReadFile(wasmDir + name)
	if err != nil {
		b.Fatalf("read %s: %v", name, err)
	}
	return wasm
}
