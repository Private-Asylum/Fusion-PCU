# Strict SGD comparisons

Run from the workspace root:

```sh
cargo bench -p fusion-pcu-rocm --features tensor --bench strict_sgd
cargo bench -p fusion-pcu-rocm --features tensor --bench strict_sgd -- --test
cargo bench -p fusion-pcu-rocm --features tensor,allocation-census --bench strict_sgd -- --test
```

`source/source.rs` contains actual generic per-function `#[pcu]` compositions.
Each accepted update checks and rounds `0.5 * gradient`, then checks and rounds
`weight - product`, in the destination F32/F64 format. It never contracts these
steps. The IEEE-derived default reports tiny, inexact results after rounding;
exact subnormals remain legal. Tight and gradual policies are separate controls.
Default checked boundary SGD, portable determinism, and incompatible numerical
permissions are explicit cold rejections.

Thirty-six Criterion peers cover F32/F64, 65/65,536/1,048,576 elements, full-host
and resident boundaries, and three execution routes:

- Actual annotated source, with automatic input staging and owned output.
- The identical explicit graph, prepared and prewarmed before measurement.
- Native HIP launch of the identical public generated checker, bypassing PCU
  graph/source scheduling. This is a native execution control, not independently
  handwritten arithmetic or an unchecked vendor-throughput comparison.

All routes allocate a fresh output/status, launch once, wait for terminal
completion, inspect status and release output ownership. The full-host boundary
includes input refresh and output readback. Resident measurements alternate
preloaded distinct input banks and exclude transfers; a complete output oracle
still runs outside timing on every call. Two dyadic input patterns provide an
independent exact-output oracle. Seven exceptional profiles per width compare
structured core/source/graph faults with native packed status and verify retry.

Cold preparation and separate native submission/completion durations are
reported outside Criterion. Those phase witnesses are single observations, not
statistical estimates. Allocation census is a separate compile-time diagnostic
build and includes the untimed readback/oracle path; primary builds contain no
census hooks. It counts caller-thread Rust allocations, not GPU/runtime/internal
driver allocations. On the accepted smoke cut, source and native each allocate
five Rust objects/208 bytes per call, versus six/304 bytes for the explicit graph.
The latter retains a separate owned-output collection.

Setup, code generation, selection, numerical probes and oracle work stay outside
timed submission. `--test` checks correctness without establishing throughput.
The AMD activity guard runs before setup and each shape; other GPU users still
make any timing measurements provisional.
