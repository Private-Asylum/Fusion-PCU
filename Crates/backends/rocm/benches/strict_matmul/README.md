# Ordered Strict MatMul source, graph and native controls

The Criterion target performs genuine generic `#[pcu]` MatMul, the frozen
explicit graph, and the same independently launched integer-only ordered
checker. F32/F64 fold K in ascending order, separately rounding/checking each
multiply and add. Native compound and optimized precision permissions may use
this stronger implementation; their exact tuple is retained in cache identity.
Default checked Boundary and tensor Portable remain unsupported. The vendor
BLAS throughput control has explicitly different unchecked semantics and a
preallocated output, so it is not a matched checked implementation.

All four compound/precision combinations and all three underflow policies run
at the small shape, alongside original larger default-profile cases. Exceptional
core/source/graph/native step faults and retry run under all four permissions.
Every measured source, graph and checked-native call has a fresh output and
private status word, terminal completion, status inspection and output release.
Host boundaries include changing input upload and output readback; resident
measurements exclude transfers but validate complete output outside timing.

A separate census runs 64 alternating-bank calls per checked route, including
matched readback/oracle. It reports caller-thread Rust allocations and actual
backend SDK invoke counters without pretending those count vendor internals.
`--test` runs semantic checks and skips the separate elapsed-ratio experiment.
GPU activity guards remain required for statistical measurements. Device proof
and frozen source hashes are tracked in the backend plan, separately from this
executable definition.
