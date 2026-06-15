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

//go:build fusion

// Superinstruction (opcode fusion) corpus analysis.
//
// This is a spike tool, not a regular test. It walks a corpus of .wasm files,
// decodes each function body into its opcode sequence (skipping operand words),
// and counts how often each opcode n-gram appears STATICALLY across the corpus.
// The output ranks candidate sequences to fuse into single handlers.
//
// Run with:
//
//	go test -tags fusion -run TestFusionAnalysis -v ./epsilon \
//	    -corpus ./wasip1/wasi-testsuite,./internal/benchmarks/wasm
//
// Static frequency is a first-cut proxy. It counts occurrences in the code, not
// executions; a sequence inside a hot loop executes far more than its static
// count. Dynamic profiling is the natural follow-up (see investigation doc).
package epsilon

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

var corpusFlag = flag.String("corpus", "../wasip1/wasi-testsuite,../internal/benchmarks/wasm",
	"comma-separated list of directories to scan recursively for .wasm files")

// operandWordCount lives in fusion.go (non-test build) and is reused here.

// decodeOpcodes walks a function body and returns its opcode sequence with all
// operand words removed.
func decodeOpcodes(body []uint64) []opcode {
	ops := make([]opcode, 0, len(body))
	for i := 0; i < len(body); {
		op := opcode(body[i])
		ops = append(ops, op)
		i += 1 + operandWordCount(op, body, i+1)
	}
	return ops
}

func TestFusionAnalysis(t *testing.T) {
	const maxN = 4 // analyze bigrams..4-grams
	const topPerN = 25

	var files []string
	for _, dir := range strings.Split(*corpusFlag, ",") {
		dir = strings.TrimSpace(dir)
		if dir == "" {
			continue
		}
		_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() && strings.HasSuffix(path, ".wasm") {
				files = append(files, path)
			}
			return nil
		})
	}
	if len(files) == 0 {
		t.Fatalf("no .wasm files found under %q", *corpusFlag)
	}

	counts := make([]map[string]int, maxN+1) // counts[n] keyed by n-gram
	for n := 2; n <= maxN; n++ {
		counts[n] = map[string]int{}
	}
	var (
		parsedFiles  int
		failedFiles  int
		totalFuncs   int
		totalOpcodes int
		unigram      = map[opcode]int{}
	)

	for _, path := range files {
		data, err := os.ReadFile(path)
		if err != nil {
			failedFiles++
			continue
		}
		module, err := newParser(strings.NewReader(string(data)), DefaultConfig()).parse()
		if err != nil {
			failedFiles++
			continue
		}
		parsedFiles++
		for fi := range module.funcs {
			ops := decodeOpcodes(module.funcs[fi].body)
			totalFuncs++
			totalOpcodes += len(ops)
			for _, op := range ops {
				unigram[op]++
			}
			// n-grams do not span function boundaries.
			for n := 2; n <= maxN; n++ {
				for i := 0; i+n <= len(ops); i++ {
					counts[n][ngramKey(ops[i:i+n])]++
				}
			}
		}
	}

	fmt.Printf("\n=== Fusion corpus analysis ===\n")
	fmt.Printf("files: %d parsed, %d skipped | functions: %d | opcodes: %d\n\n",
		parsedFiles, failedFiles, totalFuncs, totalOpcodes)

	// Top single opcodes for baseline context.
	fmt.Printf("--- top 20 single opcodes (dispatch hot spots) ---\n")
	printTopUnigram(unigram, totalOpcodes, 20)

	for n := 2; n <= maxN; n++ {
		fmt.Printf("\n--- top %d %d-gram sequences ---\n", topPerN, n)
		printTopNgram(counts[n], topPerN)
	}
	fmt.Println()
}

func ngramKey(ops []opcode) string {
	var b strings.Builder
	for i, op := range ops {
		if i > 0 {
			b.WriteByte('\x00')
		}
		fmt.Fprintf(&b, "%d", uint32(op))
	}
	return b.String()
}

func keyToNames(key string) string {
	parts := strings.Split(key, "\x00")
	names := make([]string, len(parts))
	for i, p := range parts {
		var v uint32
		fmt.Sscanf(p, "%d", &v)
		names[i] = opcodeName(opcode(v))
	}
	return strings.Join(names, " ")
}

type ngramStat struct {
	key   string
	count int
}

func printTopNgram(m map[string]int, top int) {
	stats := make([]ngramStat, 0, len(m))
	for k, c := range m {
		stats = append(stats, ngramStat{k, c})
	}
	sort.Slice(stats, func(i, j int) bool {
		if stats[i].count != stats[j].count {
			return stats[i].count > stats[j].count
		}
		return stats[i].key < stats[j].key
	})
	if len(stats) > top {
		stats = stats[:top]
	}
	for _, s := range stats {
		fmt.Printf("%8d  %s\n", s.count, keyToNames(s.key))
	}
}

func printTopUnigram(m map[opcode]int, total, top int) {
	type st struct {
		op opcode
		c  int
	}
	stats := make([]st, 0, len(m))
	for op, c := range m {
		stats = append(stats, st{op, c})
	}
	sort.Slice(stats, func(i, j int) bool { return stats[i].c > stats[j].c })
	if len(stats) > top {
		stats = stats[:top]
	}
	for _, s := range stats {
		pct := 100 * float64(s.c) / float64(total)
		fmt.Printf("%8d  %5.1f%%  %s\n", s.c, pct, opcodeName(s.op))
	}
}

func opcodeName(op opcode) string {
	if n, ok := opcodeNames[uint32(op)]; ok {
		return n
	}
	return fmt.Sprintf("op_0x%X", uint32(op))
}

var opcodeNames = map[uint32]string{
	0x00:   "unreachable",
	0x01:   "nop",
	0x02:   "block",
	0x03:   "loop",
	0x04:   "ifOp",
	0x05:   "elseOp",
	0x0B:   "end",
	0x0C:   "br",
	0x0D:   "brIf",
	0x0E:   "brTable",
	0x0F:   "returnOp",
	0x10:   "call",
	0x11:   "callIndirect",
	0x1A:   "drop",
	0x1B:   "selectOp",
	0x1C:   "selectT",
	0x20:   "localGet",
	0x21:   "localSet",
	0x22:   "localTee",
	0x23:   "globalGet",
	0x24:   "globalSet",
	0x25:   "tableGet",
	0x26:   "tableSet",
	0x28:   "i32Load",
	0x29:   "i64Load",
	0x2A:   "f32Load",
	0x2B:   "f64Load",
	0x2C:   "i32Load8S",
	0x2D:   "i32Load8U",
	0x2E:   "i32Load16S",
	0x2F:   "i32Load16U",
	0x30:   "i64Load8S",
	0x31:   "i64Load8U",
	0x32:   "i64Load16S",
	0x33:   "i64Load16U",
	0x34:   "i64Load32S",
	0x35:   "i64Load32U",
	0x36:   "i32Store",
	0x37:   "i64Store",
	0x38:   "f32Store",
	0x39:   "f64Store",
	0x3A:   "i32Store8",
	0x3B:   "i32Store16",
	0x3C:   "i64Store8",
	0x3D:   "i64Store16",
	0x3E:   "i64Store32",
	0x3F:   "memorySize",
	0x40:   "memoryGrow",
	0x41:   "i32Const",
	0x42:   "i64Const",
	0x43:   "f32Const",
	0x44:   "f64Const",
	0x45:   "i32Eqz",
	0x46:   "i32Eq",
	0x47:   "i32Ne",
	0x48:   "i32LtS",
	0x49:   "i32LtU",
	0x4A:   "i32GtS",
	0x4B:   "i32GtU",
	0x4C:   "i32LeS",
	0x4D:   "i32LeU",
	0x4E:   "i32GeS",
	0x4F:   "i32GeU",
	0x50:   "i64Eqz",
	0x51:   "i64Eq",
	0x52:   "i64Ne",
	0x53:   "i64LtS",
	0x54:   "i64LtU",
	0x55:   "i64GtS",
	0x56:   "i64GtU",
	0x57:   "i64LeS",
	0x58:   "i64LeU",
	0x59:   "i64GeS",
	0x5A:   "i64GeU",
	0x5B:   "f32Eq",
	0x5C:   "f32Ne",
	0x5D:   "f32Lt",
	0x5E:   "f32Gt",
	0x5F:   "f32Le",
	0x60:   "f32Ge",
	0x61:   "f64Eq",
	0x62:   "f64Ne",
	0x63:   "f64Lt",
	0x64:   "f64Gt",
	0x65:   "f64Le",
	0x66:   "f64Ge",
	0x67:   "i32Clz",
	0x68:   "i32Ctz",
	0x69:   "i32Popcnt",
	0x6A:   "i32Add",
	0x6B:   "i32Sub",
	0x6C:   "i32Mul",
	0x6D:   "i32DivS",
	0x6E:   "i32DivU",
	0x6F:   "i32RemS",
	0x70:   "i32RemU",
	0x71:   "i32And",
	0x72:   "i32Or",
	0x73:   "i32Xor",
	0x74:   "i32Shl",
	0x75:   "i32ShrS",
	0x76:   "i32ShrU",
	0x77:   "i32Rotl",
	0x78:   "i32Rotr",
	0x79:   "i64Clz",
	0x7A:   "i64Ctz",
	0x7B:   "i64Popcnt",
	0x7C:   "i64Add",
	0x7D:   "i64Sub",
	0x7E:   "i64Mul",
	0x7F:   "i64DivS",
	0x80:   "i64DivU",
	0x81:   "i64RemS",
	0x82:   "i64RemU",
	0x83:   "i64And",
	0x84:   "i64Or",
	0x85:   "i64Xor",
	0x86:   "i64Shl",
	0x87:   "i64ShrS",
	0x88:   "i64ShrU",
	0x89:   "i64Rotl",
	0x8A:   "i64Rotr",
	0x8B:   "f32Abs",
	0x8C:   "f32Neg",
	0x8D:   "f32Ceil",
	0x8E:   "f32Floor",
	0x8F:   "f32Trunc",
	0x90:   "f32Nearest",
	0x91:   "f32Sqrt",
	0x92:   "f32Add",
	0x93:   "f32Sub",
	0x94:   "f32Mul",
	0x95:   "f32Div",
	0x96:   "f32Min",
	0x97:   "f32Max",
	0x98:   "f32Copysign",
	0x99:   "f64Abs",
	0x9A:   "f64Neg",
	0x9B:   "f64Ceil",
	0x9C:   "f64Floor",
	0x9D:   "f64Trunc",
	0x9E:   "f64Nearest",
	0x9F:   "f64Sqrt",
	0xA0:   "f64Add",
	0xA1:   "f64Sub",
	0xA2:   "f64Mul",
	0xA3:   "f64Div",
	0xA4:   "f64Min",
	0xA5:   "f64Max",
	0xA6:   "f64Copysign",
	0xA7:   "i32WrapI64",
	0xA8:   "i32TruncF32S",
	0xA9:   "i32TruncF32U",
	0xAA:   "i32TruncF64S",
	0xAB:   "i32TruncF64U",
	0xAC:   "i64ExtendI32S",
	0xAD:   "i64ExtendI32U",
	0xAE:   "i64TruncF32S",
	0xAF:   "i64TruncF32U",
	0xB0:   "i64TruncF64S",
	0xB1:   "i64TruncF64U",
	0xB2:   "f32ConvertI32S",
	0xB3:   "f32ConvertI32U",
	0xB4:   "f32ConvertI64S",
	0xB5:   "f32ConvertI64U",
	0xB6:   "f32DemoteF64",
	0xB7:   "f64ConvertI32S",
	0xB8:   "f64ConvertI32U",
	0xB9:   "f64ConvertI64S",
	0xBA:   "f64ConvertI64U",
	0xBB:   "f64PromoteF32",
	0xBC:   "i32ReinterpretF32",
	0xBD:   "i64ReinterpretF64",
	0xBE:   "f32ReinterpretI32",
	0xBF:   "f64ReinterpretI64",
	0xC0:   "i32Extend8S",
	0xC1:   "i32Extend16S",
	0xC2:   "i64Extend8S",
	0xC3:   "i64Extend16S",
	0xC4:   "i64Extend32S",
	0xD0:   "refNull",
	0xD1:   "refIsNull",
	0xD2:   "refFunc",
	0xFC00: "i32TruncSatF32S",
	0xFC01: "i32TruncSatF32U",
	0xFC02: "i32TruncSatF64S",
	0xFC03: "i32TruncSatF64U",
	0xFC04: "i64TruncSatF32S",
	0xFC05: "i64TruncSatF32U",
	0xFC06: "i64TruncSatF64S",
	0xFC07: "i64TruncSatF64U",
	0xFC08: "memoryInit",
	0xFC09: "dataDrop",
	0xFC0A: "memoryCopy",
	0xFC0B: "memoryFill",
	0xFC0C: "tableInit",
	0xFC0D: "elemDrop",
	0xFC0E: "tableCopy",
	0xFC0F: "tableGrow",
	0xFC10: "tableSize",
	0xFC11: "tableFill",
	0xFD00: "v128Load",
	0xFD01: "v128Load8x8S",
	0xFD02: "v128Load8x8U",
	0xFD03: "v128Load16x4S",
	0xFD04: "v128Load16x4U",
	0xFD05: "v128Load32x2S",
	0xFD06: "v128Load32x2U",
	0xFD07: "v128Load8Splat",
	0xFD08: "v128Load16Splat",
	0xFD09: "v128Load32Splat",
	0xFD0A: "v128Load64Splat",
	0xFD0B: "v128Store",
	0xFD0C: "v128Const",
	0xFD0D: "i8x16Shuffle",
	0xFD0E: "i8x16Swizzle",
	0xFD0F: "i8x16Splat",
	0xFD10: "i16x8Splat",
	0xFD11: "i32x4Splat",
	0xFD12: "i64x2Splat",
	0xFD13: "f32x4Splat",
	0xFD14: "f64x2Splat",
	0xFD15: "i8x16ExtractLaneS",
	0xFD16: "i8x16ExtractLaneU",
	0xFD17: "i8x16ReplaceLane",
	0xFD18: "i16x8ExtractLaneS",
	0xFD19: "i16x8ExtractLaneU",
	0xFD1A: "i16x8ReplaceLane",
	0xFD1B: "i32x4ExtractLane",
	0xFD1C: "i32x4ReplaceLane",
	0xFD1D: "i64x2ExtractLane",
	0xFD1E: "i64x2ReplaceLane",
	0xFD1F: "f32x4ExtractLane",
	0xFD20: "f32x4ReplaceLane",
	0xFD21: "f64x2ExtractLane",
	0xFD22: "f64x2ReplaceLane",
	0xFD23: "i8x16Eq",
	0xFD24: "i8x16Ne",
	0xFD25: "i8x16LtS",
	0xFD26: "i8x16LtU",
	0xFD27: "i8x16GtS",
	0xFD28: "i8x16GtU",
	0xFD29: "i8x16LeS",
	0xFD2A: "i8x16LeU",
	0xFD2B: "i8x16GeS",
	0xFD2C: "i8x16GeU",
	0xFD2D: "i16x8Eq",
	0xFD2E: "i16x8Ne",
	0xFD2F: "i16x8LtS",
	0xFD30: "i16x8LtU",
	0xFD31: "i16x8GtS",
	0xFD32: "i16x8GtU",
	0xFD33: "i16x8LeS",
	0xFD34: "i16x8LeU",
	0xFD35: "i16x8GeS",
	0xFD36: "i16x8GeU",
	0xFD37: "i32x4Eq",
	0xFD38: "i32x4Ne",
	0xFD39: "i32x4LtS",
	0xFD3A: "i32x4LtU",
	0xFD3B: "i32x4GtS",
	0xFD3C: "i32x4GtU",
	0xFD3D: "i32x4LeS",
	0xFD3E: "i32x4LeU",
	0xFD3F: "i32x4GeS",
	0xFD40: "i32x4GeU",
	0xFD41: "f32x4Eq",
	0xFD42: "f32x4Ne",
	0xFD43: "f32x4Lt",
	0xFD44: "f32x4Gt",
	0xFD45: "f32x4Le",
	0xFD46: "f32x4Ge",
	0xFD47: "f64x2Eq",
	0xFD48: "f64x2Ne",
	0xFD49: "f64x2Lt",
	0xFD4A: "f64x2Gt",
	0xFD4B: "f64x2Le",
	0xFD4C: "f64x2Ge",
	0xFD4D: "v128Not",
	0xFD4E: "v128And",
	0xFD4F: "v128Andnot",
	0xFD50: "v128Or",
	0xFD51: "v128Xor",
	0xFD52: "v128Bitselect",
	0xFD53: "v128AnyTrue",
	0xFD54: "v128Load8Lane",
	0xFD55: "v128Load16Lane",
	0xFD56: "v128Load32Lane",
	0xFD57: "v128Load64Lane",
	0xFD58: "v128Store8Lane",
	0xFD59: "v128Store16Lane",
	0xFD5A: "v128Store32Lane",
	0xFD5B: "v128Store64Lane",
	0xFD5C: "v128Load32Zero",
	0xFD5D: "v128Load64Zero",
	0xFD5E: "f32x4DemoteF64x2Zero",
	0xFD5F: "f64x2PromoteLowF32x4",
	0xFD60: "i8x16Abs",
	0xFD61: "i8x16Neg",
	0xFD62: "i8x16Popcnt",
	0xFD63: "i8x16AllTrue",
	0xFD64: "i8x16Bitmask",
	0xFD65: "i8x16NarrowI16x8S",
	0xFD66: "i8x16NarrowI16x8U",
	0xFD67: "f32x4Ceil",
	0xFD68: "f32x4Floor",
	0xFD69: "f32x4Trunc",
	0xFD6A: "f32x4Nearest",
	0xFD6B: "i8x16Shl",
	0xFD6C: "i8x16ShrS",
	0xFD6D: "i8x16ShrU",
	0xFD6E: "i8x16Add",
	0xFD6F: "i8x16AddSatS",
	0xFD70: "i8x16AddSatU",
	0xFD71: "i8x16Sub",
	0xFD72: "i8x16SubSatS",
	0xFD73: "i8x16SubSatU",
	0xFD74: "f64x2Ceil",
	0xFD75: "f64x2Floor",
	0xFD76: "i8x16MinS",
	0xFD77: "i8x16MinU",
	0xFD78: "i8x16MaxS",
	0xFD79: "i8x16MaxU",
	0xFD7A: "f64x2Trunc",
	0xFD7B: "i8x16AvgrU",
	0xFD7C: "i16x8ExtaddPairwiseI8x16S",
	0xFD7D: "i16x8ExtaddPairwiseI8x16U",
	0xFD7E: "i32x4ExtaddPairwiseI16x8S",
	0xFD7F: "i32x4ExtaddPairwiseI16x8U",
	0xFD80: "i16x8Abs",
	0xFD81: "i16x8Neg",
	0xFD82: "i16x8Q15mulrSatS",
	0xFD83: "i16x8AllTrue",
	0xFD84: "i16x8Bitmask",
	0xFD85: "i16x8NarrowI32x4S",
	0xFD86: "i16x8NarrowI32x4U",
	0xFD87: "i16x8ExtendLowI8x16S",
	0xFD88: "i16x8ExtendHighI8x16S",
	0xFD89: "i16x8ExtendLowI8x16U",
	0xFD8A: "i16x8ExtendHighI8x16U",
	0xFD8B: "i16x8Shl",
	0xFD8C: "i16x8ShrS",
	0xFD8D: "i16x8ShrU",
	0xFD8E: "i16x8Add",
	0xFD8F: "i16x8AddSatS",
	0xFD90: "i16x8AddSatU",
	0xFD91: "i16x8Sub",
	0xFD92: "i16x8SubSatS",
	0xFD93: "i16x8SubSatU",
	0xFD94: "f64x2Nearest",
	0xFD95: "i16x8Mul",
	0xFD96: "i16x8MinS",
	0xFD97: "i16x8MinU",
	0xFD98: "i16x8MaxS",
	0xFD99: "i16x8MaxU",
	0xFD9B: "i16x8AvgrU",
	0xFD9C: "i16x8ExtmulLowI8x16S",
	0xFD9D: "i16x8ExtmulHighI8x16S",
	0xFD9E: "i16x8ExtmulLowI8x16U",
	0xFD9F: "i16x8ExtmulHighI8x16U",
	0xFDA0: "i32x4Abs",
	0xFDA1: "i32x4Neg",
	0xFDA3: "i32x4AllTrue",
	0xFDA4: "i32x4Bitmask",
	0xFDA7: "i32x4ExtendLowI16x8S",
	0xFDA8: "i32x4ExtendHighI16x8S",
	0xFDA9: "i32x4ExtendLowI16x8U",
	0xFDAA: "i32x4ExtendHighI16x8U",
	0xFDAB: "i32x4Shl",
	0xFDAC: "i32x4ShrS",
	0xFDAD: "i32x4ShrU",
	0xFDAE: "i32x4Add",
	0xFDB1: "i32x4Sub",
	0xFDB5: "i32x4Mul",
	0xFDB6: "i32x4MinS",
	0xFDB7: "i32x4MinU",
	0xFDB8: "i32x4MaxS",
	0xFDB9: "i32x4MaxU",
	0xFDBA: "i32x4DotI16x8S",
	0xFDBC: "i32x4ExtmulLowI16x8S",
	0xFDBD: "i32x4ExtmulHighI16x8S",
	0xFDBE: "i32x4ExtmulLowI16x8U",
	0xFDBF: "i32x4ExtmulHighI16x8U",
	0xFDC0: "i64x2Abs",
	0xFDC1: "i64x2Neg",
	0xFDC3: "i64x2AllTrue",
	0xFDC4: "i64x2Bitmask",
	0xFDC7: "i64x2ExtendLowI32x4S",
	0xFDC8: "i64x2ExtendHighI32x4S",
	0xFDC9: "i64x2ExtendLowI32x4U",
	0xFDCA: "i64x2ExtendHighI32x4U",
	0xFDCB: "i64x2Shl",
	0xFDCC: "i64x2ShrS",
	0xFDCD: "i64x2ShrU",
	0xFDCE: "i64x2Add",
	0xFDD1: "i64x2Sub",
	0xFDD5: "i64x2Mul",
	0xFDD6: "i64x2Eq",
	0xFDD7: "i64x2Ne",
	0xFDD8: "i64x2LtS",
	0xFDD9: "i64x2GtS",
	0xFDDA: "i64x2LeS",
	0xFDDB: "i64x2GeS",
	0xFDDC: "i64x2ExtmulLowI32x4S",
	0xFDDD: "i64x2ExtmulHighI32x4S",
	0xFDDE: "i64x2ExtmulLowI32x4U",
	0xFDDF: "i64x2ExtmulHighI32x4U",
	0xFDE0: "f32x4Abs",
	0xFDE1: "f32x4Neg",
	0xFDE3: "f32x4Sqrt",
	0xFDE4: "f32x4Add",
	0xFDE5: "f32x4Sub",
	0xFDE6: "f32x4Mul",
	0xFDE7: "f32x4Div",
	0xFDE8: "f32x4Min",
	0xFDE9: "f32x4Max",
	0xFDEA: "f32x4Pmin",
	0xFDEB: "f32x4Pmax",
	0xFDEC: "f64x2Abs",
	0xFDED: "f64x2Neg",
	0xFDEF: "f64x2Sqrt",
	0xFDF0: "f64x2Add",
	0xFDF1: "f64x2Sub",
	0xFDF2: "f64x2Mul",
	0xFDF3: "f64x2Div",
	0xFDF4: "f64x2Min",
	0xFDF5: "f64x2Max",
	0xFDF6: "f64x2Pmin",
	0xFDF7: "f64x2Pmax",
	0xFDF8: "i32x4TruncSatF32x4S",
	0xFDF9: "i32x4TruncSatF32x4U",
	0xFDFA: "f32x4ConvertI32x4S",
	0xFDFB: "f32x4ConvertI32x4U",
	0xFDFC: "i32x4TruncSatF64x2SZero",
	0xFDFD: "i32x4TruncSatF64x2UZero",
	0xFDFE: "f64x2ConvertLowI32x4S",
	0xFDFF: "f64x2ConvertLowI32x4U",
}
