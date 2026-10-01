# Checked F64 Neg host benchmark

`checked_f64_neg` compares the actual `#[pcu]` function shared with the integration
tests, explicit `PcuCpuPreparedHost` prepared from that same captured source IR,
and a native Rust control that performs exact core checked preflight before
publishing sign-bit inversion. Each detected Scalar/SSE2/AVX2/NEON instruction
family receives separate source/prepared entries. The native control is common.

```sh
cargo +stable bench -p fusion-pcu-cpu --features std --bench checked_f64_neg -- --test
```

Extents are 17, 4,096 and 1,048,576 elements. Every warm iteration changes input,
validates complete extents, checks selected numerical faults, computes and
finishes the synchronous host boundary, and observes output. Criterion measures
this full elapsed wall. Discovery, cold source capture/preparation, container
setup and complete output/fault/retry/tail oracles stay outside warm timing.
This is an explicit prepared-source comparison; global facade routing has its
own root-owned production-source benchmark.

Before timing, each source/prepared/native peer checks every output bit including
signed zero and exact subnormals, all untouched tails, changed input, first
nonfinite fault, complete rollback, retry and short input/output rejection.
Defaults use IEEE tininess after rounding and inexactness; Neg is exact, so
finite subnormals do not fault under the default policy. Unit tests separately
cover every explicit underflow policy and rejected numerical offers.

The support module installs a delegating System allocator and enables a census
only around the current thread's cold source/prepared construction and successful
or faulting warm source/prepared/native calls. Allocation, deallocation and
reallocation counts and bytes are printed and asserted zero for this admitted
profile. Output/input containers, Criterion, unrelated threads, operating system
allocation and foreign/native allocator APIs are outside this Rust census.
Measurement itself has tracking disabled. There is no foreign library call in
this executor.

The 2026-09-30 Ryzen 9 5950X / Rust 1.98.1 smoke passed all 21 available x86
cases, including the full oracles and zero-allocation assertions. The complete
Cargo invocation took 2.910906 seconds including compilation and harness work.
Smoke completion is correctness evidence, not a statistical performance result;
no throughput or instruction preference follows. NEON cross-compilation is
separate from pending AArch64 hardware execution evidence.
