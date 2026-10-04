# Dense integer graph literals

This target compares `(input + constant) * uniform` for fourteen signed/unsigned
8/16/32/64/128/256/512-bit integer carriers at extents 65 and 4096. The eight
headers cross Boundary/Strict, Checked/BackendDefined compounds and
Preserve/BackendOptimized precision. Range is Reject, reproducibility is
Unspecified, and floating underflow is IEEE. Compact uniforms and PortableV1
integer tensor compounds remain refused.

The explicit graph owns dense Constant and Uniform producers. The closest
ordinary `#[pcu]` source accepts three dense input banks because the owned
frontend cannot author Constant/Uniform producers. The handwritten SDK control
also accepts three banks and uses the independently authored checked limb
arithmetic from the composed-integer target. These routes share values and
expression, but do not have identical graph producer or transfer topology.
There is no claim of source literal lowering.
The graph route constructs and uploads a fresh device input on each call; its
allocation census includes that ownership boundary. The ordinary source and
preallocated SDK control have different input and output ownership topology.

Before sampling, each row checks full high-bit output and two untouched tail
values, fatal Add and Mul overflow with exact earliest index and graph operation order before invocation order, host publication
rollback, rejected short spans and healthy retry. Each profile separately
checks returned literal storage, mutation followed by graph input consumption,
replay of the original prepared literal and escaped uniform validity after plan
and global cache drop. Literal storage only used inside a schedule stays in cold
scratch; selected outputs always get fresh initialized storage.

Set `PCU_INTEGER_TENSOR_LITERAL_SEMANTICS=1` for correctness without statistics.
Set `PCU_INTEGER_TENSOR_LITERAL_CPU_REFERENCE=1` alongside it for the closest
source's independent checked-Rust comparison on CPU, without GPU selection.
The separate `allocation-census` build records 64 changing calls per route.
PCU allocations are reported as observed, not asserted absent for owned graphs.
SDK Rust heap is asserted zero; its own counters require 192 uploads, 128
downloads, 64 resets, event records and waits, and 128 launches. Raw SDK allocations,
module creation/destruction and SDK internals are outside these counters.
Compilation, endpoint allocation and correctness probes are cold. Timing, when
explicitly run after activity guards, includes full host boundary transfers,
launch, completion, status and publication; verification is outside elapsed
samples. No timing is inferred from correctness or census runs.

The SDK uses separate ordered Add/Mul kernels and separate immutable per-phase
status words. A later-index Add fault wins over an earlier-index Mul fault,
matching the graph and ordinary source schedule. This is deliberately distinct
from a fused lane-level expression fault order.

Frozen source `f1d7465e` passes all224 closest-source CPU profiles, all672 native
semantic keys and a separate64-changing-call census over all672 keys. The earlier
`49ceea46` fused-control capture was never run on GPU. PCU source/graph scopes
report192/384 Rust allocation/free calls per64 calls respectively; SDK scopes
report zero. These are exact fixture observations, not an equivalent ownership
or latency comparison.
