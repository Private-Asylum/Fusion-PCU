# Native F64 SGD source / graph / identical native comparison

Actual per-function `#[pcu]` updates use explicit native compound arithmetic with
preserved precision or explicitly permitted F64 FMA. The finite frozen F32 rate
-0.25 widens exactly to F64. Strict, default checked Boundary, PortableV1 and
stronger native underflow policies remain rejected. No checked status is invented.

Canonical Criterion target `native_sgd_f64` runs 65 and 1,048,576 elements,
Preserve/BackendOptimized, full-host/resident, with complete independent integer
oracles outside every timed invocation and alternating physical input banks.
All routes allocate one fresh output, launch one kernel, complete one terminal
event and release output; inputs stay allocated across warm calls. Full-host
includes payload uploads/readback; resident excludes payload transfers from timing.
Both native/graph banks are restored before the resident boundary. Cold compilation
is separate. Twenty samples, 500 ms warmup, 2 s requested measurement, Criterion
95% intervals. Device activity guards apply; counting allocator is a separate run.

Authored hardware tests prove an independent exact cancellation witness:
rate=1+2^-23, gradient=1-2^-23+2^-46 gives preserved +0 and explicitly fused
-2^-69, plus signed-zero rates, maximal finite F32 rate widening, changing inputs,
permission/cache/shape/escaped ownership and finite retry. This native permission
is not portable exceptional arithmetic or blanket wider floating support.
