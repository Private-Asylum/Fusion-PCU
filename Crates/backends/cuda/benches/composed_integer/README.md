# CUDA composed integer controls

The canonical Criterion target has genuine deterministic `#[pcu]` source,
independently authored typed Dispatch IR, and independently handwritten raw CUDA
SDK arithmetic. Its 448 profiles and 1,344 route keys cover all fourteen integer
carriers, Boundary/Strict, Reject/Clamp, direct/grid, staged/dead expressions and
65/4096 elements. Every requested tuple explicitly retains PortableV1, Checked,
Preserve and IeeeAfterRounding. The earlier full24-tuple arithmetic qualification
is separate; this benchmark does not claim a larger header matrix.

Staged source reads an independent scalar seed, adds it to the saved input,
stores and reloads stage, multiplies by the original input, then subtracts the
original input into output. The dead companion discards a checked Add and
publishes the original input. Cold source/explicit-IR descriptor checks compare
actual resources, initial-read rules and logical extent. The SDK arithmetic
asset uses unsigned32 limbs and an explicit full product with signed magnitude
bounds; it imports no PCU IR, emitter or generated arithmetic helper. NVIDIA's
runtime compiler wrapper and cold selected-device facts are shared utility;
launch geometry is matched to the explicitly prepared dispatch. This is a
correctness control, not a claim of optimal CUDA arithmetic throughput.

Every route first checks the unchanged410 narrow and76 wide immutable goldens,
complete prefixes, earliest fault at logical index7, recovered Clamp notice,
whole-host Reject rollback, untouched unequal2/3 tails, healthy retry and short
input/stage/output sibling refusals. Signed MIN cases with two faulting lanes
verify earliest index3 and first checked-effect order. Offline exact .cu execution
through a C++ host adapter is additional evidence, not GPU qualification.

SDK owners retain pinned input/seed, private pinned output/stage/status shadows,
private device allocations, module, stream, event, context and Driver library.
Each call copies current Rust input/seed into pinned staging, uploads both,
resets status, launches, reads useful outputs and status privately, records and
waits on one explicit stream, then publishes all useful output prefixes only
when healthy or explicitly clamped. Reject never publishes caller bytes. Short
borrows refuse before enqueue. An uncertain SDK result retains the entire state.
The known-terminal protocol-retention witness does not simulate driver loss.

Warm calls alternate dense zero/one input banks and high-limb signed/unsigned
seeds with exact healthy results. Host staging, transfers, launch, wait, status
validation and publication are inside the interval; verification is outside.
PCU creates a distinct ordinary event per call; the SDK owner retains one event
through synchronous calls. Report those ownership/work boundaries alongside any
future timing. No statistics are valid before the semantic gates pass.

Use `PCU_COMPOSED_INTEGER_SEMANTICS=1` for native semantic qualification without
Criterion sampling. `PCU_COMPOSED_INTEGER_CPU_REFERENCE=1` runs actual source and
explicit IR against the opt-in scalar CPU reference without probing CUDA. Build
a separate `tensor,allocation-census` executable for64-changing-input Rust and
complete wrapped API deltas per route; independent SDK counters cover uploads,
status resets, downloads, launches, event records/waits and transfer bytes. Raw
SDK allocation/module/free calls and C++/Driver internal allocations are outside
those counters. Timing builds exclude census instrumentation and require the
activity guard before each route. No graph Portable or Clamp profile is added.

Set `PCU_COMPOSED_INTEGER_REPRESENTATIVE_TIMING=1` for a bounded timing subset:
u32/u512, 65/4096 elements, direct staged/dead forms, Boundary Reject with the
same explicit PortableV1/Checked/Preserve/IEEE options. Its eight profiles pair
all three routes for 24 timing rows. Full golden preflight still runs for each
selected profile. The switch does not filter semantic or census runs, which
retain their complete qualification matrix.
