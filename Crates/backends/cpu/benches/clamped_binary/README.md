# CPU Clamp semantic pairs

Run `cargo bench -p fusion-pcu-cpu --bench clamped_binary --features source-clamp -- --test` for192 genuine prepared-source, ordinary-source, explicit-graph diagnostic and independent native-control peers. All callers retain their input/output allocations, publish complete recovered maps, and return observable recovered faults. Cold proof checks both alternating input states, complete output/tails and fatal rollback; warmed256-call census asserts zero alloc/realloc/free and no candidate rescoring.

The F32/F64 native arithmetic controls are bounded normal/overflow controls, not general underflow oracles. Low-format controls manually decode and search destination midpoints independently of PCU arithmetic. Full underflow qualification is separate in `tests/clamped_binary`; no scan-based underflow claim or hidden fallback is made. `--test` collects no statistical estimates.
