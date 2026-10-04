# CUDA flat scalar broadcast

This Criterion target compares genuine ordinary `#[pcu]` source, retained prepared
IR, and an independent retained CUDA Driver/NVRTC kernel. All 22 transport carriers
run Boundary and Strict, direct and grid geometry, at 65 and 4096 output elements:
176 cases and 528 route registrations. Explicit request metadata is asserted cold.
Each warm call changes the scalar bank, reads only element zero, verifies every
output byte and both untouched tail elements, and asserts no source reranking.

The independent kernel uses unsigned carriers or unsigned-limb structs and has no
PCU code-emitter dependency. Its input device allocation is exactly one scalar;
its output allocation is exactly the logical extent. Two pinned input banks and
one private pinned output remain owned through same-stream event completion.
Transfers, launch, event record, and event wait are inside the measured interval.
Verification and bank choice are outside. The PCU routes additionally pay their
normal staging/publication costs; the SDK route exposes its private pinned endpoint
directly. These different host boundaries are intentional and must accompany any
published timing comparison. This target does not claim optimal kernel throughput.

Build the primary target with `--features tensor`. Set
`PCU_FLAT_BROADCAST_SEMANTICS=1` and pass `--test` to run changing-input semantic
qualification without Criterion sampling. This mode includes one known-terminal
whole-owner retention witness; it does not simulate driver loss or qualify fault
cleanup. The normal timing mode checks device activity before each route.

Build a separate target with `--features tensor,allocation-census` for the census.
It measures 64 additional changing calls per route, prints Rust allocation counts,
and verifies that wrapped PCU symbol resolution, module load, allocation and free
counts stay unchanged. Independent SDK counters report only upload, download,
launch, event-record and event-wait calls plus transferred bytes. They do not count
raw SDK allocation/module/free calls. Primary builds contain neither allocation
instrumentation nor the independent SDK counter updates. Census results and
semantic results are separate evidence; neither is a performance sample.
