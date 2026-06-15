// Separate module so wazero (a third-party dependency) never enters the root
// module's go.mod. This harness is opt-in: it is built and run only via
// `make bench-wazero`, never by the normal `go test ./...` / `make test`.
module github.com/ziggy42/epsilon/internal/benchmarks/wazerobench

go 1.25.0

require (
	github.com/tetratelabs/wazero v1.1.0
	github.com/ziggy42/epsilon v0.0.0
)

replace github.com/ziggy42/epsilon => ../../..
