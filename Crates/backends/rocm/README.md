# ROCm backend examples and benchmark suite

## Owned results and captured composition

`cargo run -p fusion-pcu --features rocm,tensor --example owned-tensor-borrows
--release` demonstrates
ordinary calls over RAM and resident borrows, individually marked helper
functions captured into
one graph, a mixed-input transform, exclusive resident mutation, explicit stack
readback and
ordinary early/scope Drop. Device ownership is opaque: no backend upload/binding
API is needed.
The current bounded owned-source profile supports homogeneous f32 or f64 slices,
fixed arrays and matrices, usize const generics, identity, ReLU, MatMul,
Add/Sub/Mul,
ordinary elementwise `+`, `-`, `*`, nested pure expressions and marked helper
calls, immutable
let bindings and `?`/`Ok(...)`; it is not arbitrary Rust compilation. Every
synchronous call
returns after terminal completion. Separate calls retain residency; calls
captured inside a
marked composition describe one graph and allow internal liveness planning.

For example, a marked function can return `Ok(pcu::relu((lhs + rhs) * rhs)?)`.
Grouping and operand order are preserved in the cold capture. Literal
broadcasting, division,
mutable expression borrows and arbitrary control flow are rejected; expression
nesting is bounded.
Scalar signatures may be concrete or use `T: PcuScalar`, including const-shaped
generic helpers.
`&PcuTensor<T>` parameters retain a read-only resident borrow and the owner's
runtime rank and extents.
`PcuTensor<T>` parameters may be moved into marked functions, including multiple
owners
and mixtures with RAM or read-only resident borrows. Their captured bodies
support immutable operation chains and marked helper calls: borrow each external
owner before its final bare use;
any use after that move is rejected. Immutable owner renames move the owner;
borrowed views
use real Rust reference loans and may be used before the final move. Returning a
borrowed view
as an owner or using it after its owner moves is rejected. Returning a bare
borrowed parameter
as an owner is also rejected; use an explicit operation such as
`pcu::identity(input)`.
Produced graph values are non-Copy
owners too: borrow them for fanout, or move them into their final operation or
consuming helper.
Helper signatures distinguish `&PcuTensor<T>` borrows from
`PcuTensor<T>` moves, including values produced earlier in the captured
composition.
Consumed identity returns the same owner without copying; terminal ReLU may
reuse
exclusive backing. Wider graphs and ineligible/shared mutation use fresh
storage.
ROCm executes homogeneous f32/f64 arithmetic in this owned profile. Input-only
transport also
preserves all 12 sealed `PcuScalar` representations, including integer extrema
and f16/bf16 bits.
Cold tensor construction failures preserve `TensorBuild(TensorError)` kinds;
unsupported execution
remains a separate typed backend error. Selected-away inputs are not staged or
bound and do not
force readiness or session-affinity checks. Declared source shape contracts
remain checked;
selected inputs retain access, readiness and session checks. Selection is
captured once per
cache miss and reused on warm calls.
Transport support does not imply arithmetic support: integer/half owned
arithmetic, mixed-dtype
arithmetic and f64 training operators return typed errors without CPU fallback
or conversion.

`cargo bench -p fusion-pcu-rocm --features tensor --bench owned_program`
compares raw owned PCU,
source-authored owned PCU and native HIP with fresh output allocation, launch,
completion and
release timed. Current input setup/refresh and verified readback are outside
timing. The retained
source-input control separates that boundary from repeated input ownership
setup. Cold preparation,
Rust heap counts and balanced paired diagnostics are reported separately from
Criterion intervals.

`cargo bench -p fusion-pcu-rocm --features tensor --bench owned_binary` compares
two-input Add through
the same three routes at 65 and 1,048,576 elements. Both inputs remain allocated
and receive
new values before every job. Cold source/refresh preparation finishes before
calibration;
CPU result checks and input refresh stay outside the matched fresh-output timing
boundary.

`cargo run -p fusion-pcu --features rocm,tensor --example
owned-tensor-composition --release` composes
MatMul, Add and ReLU using const-generic matrices and resident weights, then
reads the result into
a stack matrix. Its escaped matrix result moves directly into a marked consuming
activation
function with runtime shape preserved. Shape and rank survive capture, staging
and residency; incompatible resident
shapes and marked helper contracts return typed errors. Dynamic resident handles
require explicit
const arguments where Rust cannot infer extents.

`cargo bench -p fusion-pcu-rocm --features tensor --bench owned_matmul` compares
that shaped source path,
a raw owned MatMul graph and native rocBLAS SGEMM/DGEMM for f32/f64 at 4x2x4 and
256x256x256. Inputs remain resident and
receive new values before each job; an independent CPU oracle checks each
result. Fresh output
allocation, submission, completion and release are timed. Readback, refresh and
oracle work are
outside those intervals; total Criterion wall time is also reported. All three
routes receive the
same complete input-refresh sequence before each timed execution to control
setup cadence.
Both precisions call the same generic source `#[pcu]` identity and MatMul
functions; the shared
benchmark driver uses static dispatch, while each native profile retains its own
precision and
refresh policy.

## Typed kernel calls

`cargo run -p fusion-pcu --features rocm --example scalar-composition --release`
demonstrates
`transform(&seed, &input, &mut output)?` with lazy runtime defaults and
individually annotated `#[pcu]`
scalar helpers. The facade's `rocm` feature makes the provider available;
compatible device
selection happens at runtime. Optional `fusion_pcu::global::configure`
preferences select a
specific device or scoring policy. Prepared state is cached per thread and
specialization;
every host call still uses the current borrowed input and mutable output
contents. No CPU
fallback is enabled. Host-call staging reuse is distinct from device-resident
result chaining.

`cargo run -p fusion-pcu-rocm --example fusion-rocm-typed-kernel` prepares one
typed `#[pcu]` kernel and calls it repeatedly with ordinary Rust slices on the
explicitly selected
ROCm device. Set `FUSION_ROCM_DEVICE` to choose a runtime device index. Each
synchronous host call
uses the current input contents, transfers the initial contents of mutable
slices as needed, waits
for completion, and returns results through the same mutable borrow. The output
suffix beyond the
kernel extent remains intact, and short buffers return an error.

The `typed_kernel` Criterion target has distinct host-slice and resident-device
groups. The host
group compares prepared PCU, direct PCU, and native HIP calls including matched
input upload,
fault reset, launch, wait, terminal fault-status readback, and result download
at 65 and 1,048,576 elements. Native HIP compiles the same checked multiply/add
kernel as PCU; both fault on invalid operands, overflow and policy-rejected
underflow. This complete-writer kernel does
not read old output contents, so neither route uploads them. Other kernels still
receive current
mutable input contents whenever required, and untouched host tails remain
intact. Preparation and native
compilation are reported separately. Each measured job uses a new input and
checks the full result
against a CPU oracle. The resident group keeps typed device buffers across
invocations to measure
reusable device-side calls without introducing a host-transfer flag. Resident
timing includes checked launch, wait and fault handling; input refresh and
payload download/oracle stay outside the timer on both routes. Additional
alternating pairs
check order effects separately from Criterion intervals, and a warm Rust heap
census reports
allocations without claiming to count driver/device allocations. Run it with
`cargo bench -p fusion-pcu-rocm --features tensor --bench typed_kernel` on a
ROCm host. The ignored
host-borrow regressions run with
`cargo test -p fusion-pcu-rocm --features tensor --test typed_kernel_host --
--ignored --nocapture`.

The smoke executable verifies the installed ROCm runtime by launching a small
HIP kernel and
running rocBLAS SGEMM. `fusion-rocm-pcu` explicitly selects the ROCm backend and
a device, then
uses PCU memory admission, transfer, owned binding, prepared Dispatch submission
and completion
contracts to execute and verify a 2048-element f32 grid-stride program with 250
logical
invocations. It does not call HIP or rocBLAS directly. Backend-specific
compilation and launch
remain inside `fusion-pcu-rocm`. Both target the RX 6900 XT.

The PCU example enumerates visible ROCm devices at runtime. Its own example
policy chooses the
device with the most physical memory, breaking ties by lowest index; set
`FUSION_ROCM_DEVICE` to
an explicit device index to override that policy. It tries each ranked device
until the actual
Dispatch program prepares successfully. The tensor example similarly requires
the selected
graph to assess as supported on its rocBLAS and synthesized Dispatch routes. An
explicit device request never selects a
different device. The neutral PCU discovery contracts expose selection
information; these explicit
examples own their policies. The optional hosted facade separately supplies
configurable defaults
for ordinary direct calls.
The example imports the ROCm crate only to register/open the explicitly chosen
provider; the
workload uses PCU traits and submits twice from one prepared executable.
Prepared warm-loop timing
belongs in the Cargo `dispatch` benchmark.

```sh
cargo run -p fusion-pcu-rocm --example fusion-rocm-smoke --release
cargo run -p fusion-pcu-rocm --example fusion-rocm-pcu --release
cargo run -p fusion-pcu-rocm --example fusion-rocm-discover --release
cargo run -p fusion-pcu-rocm --features tensor --example fusion-rocm-tensor --release
cargo run -p fusion-pcu-rocm --example fusion-rocm-rtc --release
cargo run -p fusion-pcu-rocm --features tensor --example fusion-rocm-train-step --release
cargo bench -p fusion-pcu-rocm --features tensor --bench dispatch
cargo bench -p fusion-pcu-rocm --features tensor --bench dispatch_u32
cargo bench -p fusion-pcu-rocm --features tensor --bench dispatch_u32_alu
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_add
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_relu
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_mse
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_uniform
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_add_relu_fusion
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_add_sub_chain
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_add_sub_identity
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_mul_chain
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_sgd_contract
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_resident
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_train_step
cargo bench -p fusion-pcu-rocm --features tensor --bench half_transport
cargo bench -p fusion-pcu-rocm --features tensor --bench dispatch_widen
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_mlp_train
cargo bench -p fusion-pcu-rocm --features tensor --bench owned_binary
cargo bench -p fusion-pcu-rocm --features tensor --bench owned_matmul
cargo bench -p fusion-pcu-rocm --features tensor --bench owned_relu
cargo bench -p fusion-pcu-rocm --features tensor --bench owned_identity_transfer
cargo bench -p fusion-pcu-rocm --features tensor --bench owned_binary_donor
cargo bench -p fusion-pcu-rocm --features tensor --bench owned_multi_consuming
cargo bench -p fusion-pcu-rocm --features tensor --bench owned_selected_owners
```

The owned-source benchmarks execute actual per-function `#[pcu]` calls.
`owned_relu` separates
borrowed/raw/native fresh-output work from consumed-source/native in-place work
at 65 and
1,048,576 elements. A consuming source signature transfers the logical owner;
PCU reuses its
backing only after graph legality, physical exclusivity and completion checks.
Shared or
ineligible storage uses fresh output. Inputs change and every comparison route
refreshes outside
each measured sample; readback and independent result oracles also remain
outside timing.
Rust heap counts exclude HIP/driver allocations and do not prove backing reuse
on their own.

`owned_binary_donor` compares one moved owner plus a readonly peer with native
in-place
Add/Sub/Mul, including right-donor subtraction. `owned_multi_consuming`
separates fresh-output
borrowing inside an ordinary host consuming wrapper from two-owner donation;
each native
control matches that storage policy and release boundary.
`owned_selected_owners` authors three
owned parameters but uses only the last two, testing selected-input pruning and
donation.
Its separate mixed group borrows the unused owner and RHS, retaining and
verifying both across
the call. Each native control matches that ownership boundary. Non-donor release
is timed when
the signature consumes it; retained output release is timed separately after
verified readback.

`dispatch_u32_alu` pairs a macro-authored `wrapping_add` kernel with an
independent HIP
unsigned-add kernel at 65 and 1,048,576 elements. Both keep three buffers
resident, verify
overflow results, and report prepared PCU versus native whole-route
submission/completion and
benchmark-thread Rust allocations. Cold PCU compilation is reported separately.

`fusion-rocm-tensor` builds a small two-layer MLP forward graph
(`MatMul → Add → ReLU → MatMul`), asks an explicitly selected ROCm
session to assess it, then executes the selected output dependency chain through
PCU-managed memory, rocBLAS, and synthesized PCU Dispatch. It verifies the
result against the explicit CPU reference.
Input, constant, nonempty rank-two f32 MatMul, bounded same-shape f32
Add/Sub/Mul/ReLU,
ReLU backward, and scalar MSE nodes execute on ROCm; unsupported operations
return an error
without CPU fallback.
This MLP example remains forward inference.

The tensor execution API binds inputs and a memory provider once. Call
`update_input`,
`execute` (or `execute_steps` for explicit feedback), and `read_output`; PCU
prepares storage
and revalidates changed bindings automatically. Plain execution needs one output
bank;
feedback preserves prior-step values with two. Provider and execution failures
remain
`Result` errors, and a failed multi-step run exposes no partial output.

`fusion-rocm-train-step` builds a bounded linear-regression gradient with
`Graph::backward_mse` and
composes an SGD update graph. It keeps input allocations on the selected ROCm
device across two
steps, keeps each updated weight tensor resident for the next step, and checks
each result against
the CPU reverse-mode gradient oracle. Both steps must reduce the CPU-evaluated
MSE. The example
reads each output back for verification; optimizer state and a fully
device-resident training loop
remain open.

`fusion-rocm-discover` uses PCU's bounded registry facade to report backend
readiness,
targets, devices, location, and coarse capabilities. It can run without a GPU
and reports
`Unavailable` when the HIP runtime cannot see one. This is a discovery probe,
not a kernel
execution test.

`fusion-rocm-rtc` compiles a Rust-authored `#[pcu(invocations = 250)]` kernel
using
`pcu::context::global_invocation_id()` and `pcu::context::invocation_count()`
across a
2048-element grid-stride loop. It compiles through HIPRTC without an explicit
architecture,
loads on the selected device, and verifies
all output values. This is a backend compilation probe; the PCU-only example
above remains the
consumer-facing dispatch path.

The `dispatch` benchmark compares a prepared PCU dispatch with a direct launch
of an
independently handwritten HIP kernel with the same f32 add semantics, using
separately allocated
resident buffers. It runs a 65-element
orchestration-heavy case and a 1,048,576-element memory-traffic case with
identical launch
geometry and verified output. The generated and handwritten kernels may compile
to different
machine code; interpret total-time differences as whole-route results, not pure
PCU call overhead.
The `dispatch_u32` target applies the same paired setup to a concrete `#[pcu]`
u32 identity copy
and a handwritten native HIP copy. A third timed case compares a macro-authored
grid-stride copy
against an independent native HIP grid-stride loop: 250 logical invocations
cover 2048 elements.
Both paths verify the full output, and the PCU grid-stride path also runs a
correctness preflight.
U32 arithmetic remains outside ROCm's supported subset.
The Cargo `[[bench]]` targets use Criterion with a custom harness; the shared
configuration uses
30 samples, a 300 ms warmup, and a two-second measurement window per case. The
dispatch target
keeps paired, alternating PCU/native launches inside its Criterion measurement
and reports PCU binding
construction, host submit/enqueue return, host completion wait, and blocking
readback separately;
the direct HIP path reports launch return, completion wait, and readback.
Compilation, module
loading, allocation, input upload, and output verification are outside the
repeated dispatch
samples. Each repeated sample also reports Rust global-allocator calls and
requested bytes by
binding, submit, completion-wait, and total stage on the benchmark thread. This
excludes
allocations made inside the native HIP
runtime, ROCr, the kernel driver, and other threads, so it is a measure of
Rust-side heap activity,
not total process or GPU allocation activity. The wait number is host
synchronization latency, not
device kernel time. Readback uses 64 paired, alternating, prewarmed blocking
copies into separate
preallocated host buffers; these copies are outside dispatch timing and are not
a transfer
throughput benchmark. The example ranks capable discovered devices by physical
memory, then stable device index.
Set `FUSION_ROCM_DEVICE` to require a specific runtime device. The target
architecture comes
from runtime discovery; no GPU architecture override is needed. This command
requires a visible
ROCm device and does not fall back to CPU.

The `tensor` benchmark compares PCU tensor MatMul with direct rocBLAS SGEMM
through this
repository's ROCm FFI wrapper. Its 8×8 case emphasizes per-call orchestration;
its 1024×1024
case emphasizes GPU work. Both use the same row-major mathematical operation,
separate resident
buffers and verified output. Criterion reports host wall-time estimates
including rocBLAS and
device synchronization, excluding setup, transfers and readback. These
measurements do not
isolate GPU kernel duration or measure a third-party Rust ROCm crate.
The same target also compares end-to-end two-MatMul graphs at 8×8 and 256×256.
Those
samples include allocation, uploads from contiguous f32 byte views, both
synchronized
MatMuls, readback into f32 storage, and Tensor construction on each route;
output
validation occurs before Criterion sampling.
They reveal orchestration and memory-path costs that the resident-buffer
comparison excludes.
The direct native peer matches PCU's allocation and release order. The same
target also times a
prepared PCU graph, with structural assessment
outside the repeated calls; each execution still validates its inputs and
selected handle.

The `tensor_add` Criterion benchmark compares a PCU graph Add with a handwritten
HIP Add at 65 and
1,048,576 elements. Both routes include allocation, upload, submission,
completion, and readback,
with three device allocations per call and output checks before and after
sampling. The PCU Add path caches prepared
kernels by flattened element count, so these samples are warm repeated
executions. Preparation
is reported separately as a cold first execution; a new shape or cache eviction
can incur compilation.
On the RX 6900 XT, one Criterion run estimated 85.58 µs PCU versus 73.18 µs
direct HIP for 65
elements, and 1.39 ms versus 1.43 ms for 1,048,576 elements. These are
whole-route host-wall
samples, not isolated GPU kernel times.

The separate `tensor_relu` target pairs PCU graph `ReLU` with a handwritten HIP
kernel over
alternating negative and positive values at the same sizes. It reports
first-execution cost and
warm Criterion estimates with two device allocations per call; it checks both
outputs before and
after the sampled run.
The tensor executable also verifies ReLU's NaN and signed-zero behavior on the
device against
the CPU reference contract.
One RX 6900 XT Criterion run estimated 72.25 µs PCU versus 59.56 µs HIP at 65
elements, and
1.22 ms versus 1.19 ms at 1,048,576 elements. First PCU execution took about
0.75 seconds.

The `tensor_add_relu_fusion` target isolates the opt-in PCU `SingleUseAddRelu`
grouping at 65 and
1,048,576 f32 elements. It compares the prepared separate Add/ReLU schedule, the
grouped PCU
schedule, and a handwritten single-kernel HIP peer. Inputs and PCU
scratch/output banks are
resident across samples; the native peer also reuses its two input buffers and
one output buffer.
Graph preparation, compilation, uploads, device allocation, correctness
readback, and output
verification are outside Criterion sampling. Samples include submission and
completion, but no
readback. This measures the execution path with its selected schedule and
associated resident
storage footprint, rather than allocation or transfer cost. Both PCU variants
and HIP are checked
bitwise against the same CPU result before and after sampling.

The `tensor_add_sub_chain` target extends that comparison to a three-step
Add/Sub/Add chain
ending in ReLU. It compares separate PCU kernels, the opt-in bounded group, and
a native HIP
single kernel at 65 and 1,048,576 elements. All warm routes reuse resident
inputs and outputs;
the grouped route needs no intermediate scratch. The benchmark checks bitwise
results before and
after timing, reports cold graph preparation, explicit executable prewarm, and
first execution separately, and prints an
alternating-order paired grouped/native timing and Rust heap allocation
diagnostic. Cold scopes
are labeled by what they include: PCU executable prewarm includes compilation,
while PCU first
execution, native compile/setup, and native first launch are reported
separately.

The `tensor_add_sub_identity` target uses the same bounded three-step chain with
its final
arithmetic result as the output. It compares separate PCU kernels, the opt-in
Identity-epilogue
group, and a native HIP single kernel under the same resident, bitwise-check,
prewarm, and paired
timing conditions. This measures whether the grouping works without a terminal
activation.

The `tensor_mul_chain` target applies the same comparison to a three-step
Mul-only chain. Its
opt-in group preserves each separate f32 multiplication and excludes mixed
Mul/Add/Sub regions,
where contraction would require a distinct numerical contract.

The `tensor_mse` target compares a scalar mean-squared-error graph with a native
route that
launches a squared-difference HIP kernel and reduces the nonnegative squared
values with rocBLAS
SASUM. It reports both end-to-end host-input calls and resident-input calls;
both include temporary
and scalar output allocation, synchronization, and scalar readback, while the
resident pair keeps
inputs uploaded across samples. Neither route allocates or uploads a dense
vector of ones. Before
sampling, the target compares SASUM against the previous dot-with-ones reduction
for finite
nonnegative, signed-zero, infinity, and NaN fixtures. The CPU graph checks MSE
results before and
after Criterion sampling. The end-to-end group includes host allocation and
transfers; the
resident group excludes input uploads.

The `tensor_uniform` target compares a dense splat constant with a graph-level
Uniform consumed
through scalar-index Dispatch. It reports cold provider allocation/upload bytes
separately from
warm execution; requested outputs remain dense. The `tensor_sgd_contract` target
compares strict
separate-rounding SGD and explicitly opted-in FMA contraction against matching
native HIP routes,
with preparation, resident execution, and readback reported separately. The two
arithmetic modes
have different bitwise results for the included rounding witness.

The `tensor_resident` target measures prepared MatMul graphs with input buffers
uploaded once and
reused across calls. It includes the existing PCU host-input route as a control
and a direct
rocBLAS route with equally resident inputs. Each timed call allocates its
output, waits for SGEMM,
and reads back a dense host tensor; input uploads and graph preparation are
outside the repeated
samples. The benchmark verifies every route against the CPU graph before and
after sampling.

For hardware runs, use a release build and record the runtime/device context
alongside results:

```sh
rocminfo
cargo bench -p fusion-pcu-rocm --features tensor --bench dispatch
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_add
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_relu
cargo bench -p fusion-pcu-rocm --features tensor --bench tensor_train_step
```

The `tensor_train_step` benchmark compares two prepared PCU graph executions per
sample, with
persistent inputs and the first updated weight allocation fed directly into the
second step,
against a direct HIP plus rocBLAS route that reuses preallocated intermediate
and output buffers.
Both routes start each sample from immutable resident initial weights; native
does not perform a
timed reset copy.
Both perform the same forward MatMul, prediction-gradient elementwise
operations, transposed
sample-gradient MatMul, and SGD update. PCU prepares reusable graph constants
and intermediate
scratch once, then allocates only the two chained outputs each sample; native
reuses all buffers.
Both routes read back only the final output.
The 4×2 case emphasizes orchestration; 1024×64 exercises a moderate matrix
workload; and
8192×1024 tests whether that overhead is amortized by substantially more work.
Every result is
checked against two CPU `backward_mse` graph steps before and after measurement.

On an RX 6900 XT, one 20-sample Criterion run with matched rocBLAS matrix
orientations,
resident initial weights, and prepared scratch measured PCU/direct HIP+rocBLAS
at
232.45/144.32 µs for 4×2, 297.46/208.17 µs for 1024×64, and 1.4923/1.4154 ms for
8192×1024. The last shape has 128 times the sample elements of 1024×64; its
relative gap is
about 5%, while the absolute gap is roughly 77–89 µs across the three cases.
These are complete
two-step host-wall timings on one device, not isolated IR costs or portable
performance guarantees.
The native gradient MatMul uses the same rocBLAS transpose orientation as PCU:
another equivalent
orientation took about twice as long at 8192×1024 on this card and would distort
the comparison.
An untimed diagnostic execution of the PCU two-step route observed two output
allocation calls,
zero uploads, and one final download at every shape. The output allocations took
about 3–4 µs
in those samples; these durations do not include resource release or isolate the
remaining
kernel/synchronization costs. Scratch creation and constant uploads occur before
Criterion timing.

The `tensor_mlp_train` target exercises a 1024→2048→2048→1024 dense network with
batch 256,
ReLU after both hidden layers, scalar mean-squared-error loss, gradients for all
three weight
matrices, and SGD updates. Each sample runs two steps, consuming the first
step's device-resident
updated weights in the second. PCU prepares one union graph for the loss and
three updated
weights, then reuses scratch; the native peer uses HIP kernels and rocBLAS with
resident inputs
and weight banks. Both download the two losses and final weights, and the
benchmark checks
numerical parity before and after timing. An independent small CPU graph checks
the gradient
construction; the full-size CPU oracle would be impractical. Preparation is
reported separately.
Set `FUSION_PCU_MLP_BATCH=1024` to run the larger batch; the default is 256. The
target keeps
separate prepared PCU routes with fresh owned outputs and reusable output banks,
alongside the
native route. PCU's first-class `SgdUpdate` performs each weight update in one
HIP launch. Prepared
MSE scratch retains its squared-difference buffer, and both routes defer reading
the two
losses until after step two. The banked route uses two validated, non-aliasing
output banks to
ping-pong weights and makes zero allocations or uploads during the measured
two-step pass.

In one RX 6900 XT ten-sample Criterion run, batch 1024 measured 10.544 ms PCU
fresh-output,
8.942 ms PCU banked, and 8.878 ms native; batch 256 measured 6.675, 5.251, and
5.479 ms.
A separate 16-pair order-balanced host-wall diagnostic alternated banked PCU and
native call
order and found median paired ratios of 1.002× at batch 1024 and 0.997× at batch
256. Pairwise
ranges were 0.839–1.152× and 0.871–1.046× respectively, so these results support
parity on this
card and workload, not a universal speed claim. The older single-thread/1×1
SGEMM loss results
and an extra PCU host-copy result are superseded. Preparation, allocation,
transfer, and per-node
diagnostics run outside Criterion. Native prepared 28 resident allocations
(226.5 MB). ROCm did
not report process-memory usage through the selected pool snapshot, so PCU peak
live memory
remains unavailable; cumulative allocations are not a peak. Per-node
measurements are synchronous
host-wall spans, not GPU event durations or isolated IR overhead.

Requirements: `hipcc`, the ROCm HIP runtime and headers, rocBLAS
headers/library, and accessible
`/dev/kfd` plus a render node. Runtime architecture discovery uses the KFD
topology and DRM
render-node PCI identity to match each HIP device to its base AMD `gfx` target.
Optional target
feature suffixes are not reported by this lookup. This lookup currently requires
Linux KFD/DRM
topology. When that target or `hipcc` is unavailable, PCU Dispatch uses HIPRTC's
current-device
compilation if the runtime compiler is present; rocBLAS tensor work does not
require either
source compiler. Set `HIPCC` if the compiler is not on `PATH`. The execution
binaries return exit
code 1 on compile or runtime failure.

### Strict MatMul benchmark

`cargo +stable bench -p fusion-pcu-rocm --features tensor --bench strict_matmul`
runs executed
`#[pcu(flag(strict))]` source against an explicit prepared graph and a native
launch of the
same public generated ordered checker. F32/F64 profiles are 4×2×4, 32×32×32 and
64×256×64.
The selected provider and device ordinal 0 are explicit, with block size 256 on
every route.
Run only after the lab GPU ownership gate and backend hardware conformance tests
pass.
The benchmark also checks utilization before setup and each profile.

Every strict route uses increasing K, separate destination-precision
multiply/add checks,
fresh output, fresh private fault status, terminal completion/status and timed
output release.
Full-host timing includes input upload and completed readback; resident timing
alternates two
preloaded input banks and excludes readback. Explicit graph and native inputs
reuse allocated
storage. The source route owns its ordinary staging/cache behavior. These
compare complete API
boundaries and do not isolate scheduler cost. All output bits are checked
outside the summed
elapsed time. Thirty-six resident interleaved triples repeat all six strict
route orders equally;
printed ratios are medians of paired ratios, with no confidence interval.

Boundary mode is asserted Unsupported, without treating rejection latency as
compute throughput.
Vendor BLAS is a separate unchecked, different-semantics throughput control with
a preallocated
output. Its timing does not establish checked numerical equivalence. Nominal
dyadic inputs permit
bitwise comparison without granting the vendor route a fault contract.

Untimed F32/F64 preflights compare source, explicit graph and native checks
against the core
checked reference for intermediate/final underflow, multiply/add overflow, later
cancellation,
exact subnormal and zero controls, gradual/reject-subnormal policies, no-FMA
rounding and invalid
operands. A successful retry follows every case through the same prepared
graph/native status.
No failed output is read. `--features allocation-census` selects a separate
resident Rust allocation
census run and skips Criterion timings; device and driver internal allocations
are excluded.

The existing ROCm `owned_matmul` benchmark still needs an explicit
numerical-contract migration;
its widened/FMA tolerance oracle and vendor BLAS route cannot serve as a strict
comparison.

## Development dependency and publication

The actual-source tests and PCU/native Criterion comparisons run from this
repository checkout. They use a shared path-only facade link and a local
path-only macros link to avoid test-only publication cycles. The macros link
keeps its canonical crate name; workspace inheritance cannot replace the
production macros dependency's version policy. Cargo omits those development
links and their feature forwarding from normalized registry manifests; the
library's production dependencies remain versioned. The standalone backend
archive does not provide a self-contained harness for these development targets.

### Checked training primitives

Dense F32/F64 `pcu::relu_backward(input, upstream)` supports checked Boundary
and Strict calls. It validates both inputs as finite, selects the upstream bits
for positive input, and otherwise returns positive zero. Exact selected
subnormals succeed unless `reject_subnormal_result` is selected. Explicit
Boundary `native_compound` accepts documented special-value selection without a
checked scan. PortableV1 derivatives reject. Strict or non-IEEE underflow requests use checked selection even when native compound arithmetic is permitted.

Strict F32/F64 `pcu::mean_squared_error` executes prescribed ascending
subtraction, square, running addition, and final division with destination-width
checked rounding at each step. Its current ordered reduction uses one active
GPU lane; admission proves semantics, and does not imply optimized reduction
throughput. Default checked Boundary loss remains unsupported.

The `training_step` example authors forward matrix multiplication, activation,
checked loss, ReLU derivative, gradient matrix multiplication and SGD using
ordinary arrays and per-function `#[pcu]`. A checked discarded loss remains a
fault producer before output publication. The example's two-sample gradient
scale is `2/N = 1`; it is a bounded regression model, not a general autodiff API.

Canonical `relu_backward` and `strict_mse` Criterion targets compare actual
annotated source, frozen explicit graph, and matched native source/control.
See their benchmark guides for measured physical boundaries and independent
complete-output oracles. New hardware evidence lives separately from old
records; compiling a target is not hardware acceptance.

Native F64 SGD now shares the explicit Boundary/native-compound policy with F32.
A finite frozen F32 learning rate widens exactly, including signed zero. Preserve
uses separate destination-width product/subtraction; BackendOptimized explicitly
permits F64 FMA. Default checked Boundary, native Strict, PortableV1 and stronger
native underflow policies stay rejected. The `native_sgd_f64` Criterion target
pairs genuine annotated host/resident functions with graph and identical native
controls, independent complete-output oracles and separate allocation census.
`examples/native_sgd_f64/native_sgd_f64.rs` is a self-contained ordinary function
example.

Native F64 MSE now uses same-width squared scratch, separately rounded F64
difference/square, typed double ASUM and F64 reciprocal scaling. Preserve and
BackendOptimized are independently explicit permissions; either uses F64 storage
and arithmetic in this offer, with native reduction order and special-value
behavior. Checked Boundary, native Strict, PortableV1 and stronger underflow
profiles remain rejected. BLAS handles retain exact ASUM/SCAL/pointer-mode entries
at setup, while device selection, extent/alias/lease checks, pointer-mode restore
and terminal quarantine remain on the execution path.

`native_mse_f64` pairs an actual annotated source function with graph/native
controls at matched full-host/resident boundaries; each call freshly allocates
squared scratch and a scalar. Its independent integer/dyadic oracle verifies
every result outside timing. The ordinary projection/loss example is
`examples/native_mse_f64/native_mse_f64.rs`. The legacy borrowed scratch interface
remains explicitly F32-only; F64 owned source/graph execution is supported.
Current acceptance records and any activity-gated timing remain separate in
`.pcu-validation/2026-10-01/rocm-cuda-native-f64-mse/`.


### Current checked unary and storage parity

Four low formats (F16, BF16, named E4M3FN and E5M2) execute bounded single-op
Neg/ReLU through ordinary generic `#[pcu]`, explicit IR and native controls.
Exact integer encoding preserves sign/selection bytes, rejects nonfinite inputs
including inactive ReLU, and follows all three underflow policies. Reject keeps
host outputs unchanged; observable Clamp publishes exact subnormals and a
recovered fault. Fatal resident outputs are discarded and require a fresh owner.
Unary Portable remains rejected. The `low_unary` test, canonical benchmark and
example cover actual RX6900XT/RTX3080 proof, tails and warm physical census.

All22sealed byte carriers also support direct/grid source identity and input-only
resident ownership, including I/U128/256/512 and representation-only F128/F256.
Padding-free little-endian limb storage preserves every high limb and arbitrary
float payload. Transport admission does not offer wider float arithmetic,
conversions, uniform tensor materialization or Portable identity. The canonical
`wide_transport` benchmark pairs genuine source/IR/native host and resident
workloads; its512-bit example checks output tails. Exact source/binary archives
and current proof are in the ignored outer `plans/.pcu-validation` tree.

Both current cohorts have matched API/work/heap census on real AMD/NVIDIA.
No new statistical latency claim is made under unrelated user activity. Legacy
active benchmark migration to genuine ordinary-source peers remains pending;
earlier graph-only figures are historical evidence, not source-authoring parity.


Dense owned Input/Identity/Add/Sub/Mul now admits all14 sealed integer widths,
including signed/unsigned128/256/512, through ordinary generic `#[pcu]`.
Boundary/Strict and explicit compound/precision permissions retain exact
rejecting scalar arithmetic. Both real GPUs qualify changing host/resident/mixed
inputs, consuming identity, retained escaped owners, earliest fatal range error,
private failure preserving inputs/siblings and fresh retry. Integer Uniform,
tensor Clamp/Portable, low4 tensor math and wider floating arithmetic remain
separate unadmitted contracts.

The canonical `wide_tensor` target pairs genuine source, prepared single-output
graph and identical native kernels with matched fresh output/private8B status.
Each route's64changing-input warm census measures24SDK calls including12device
selectors, two device allocations/frees, one status upload/two readbacks, one
kernel/event lifecycle, zero warm lookup/module load and five Rust
allocations/frees totaling208bytes, zero reallocations. Untimed full readback and
cached-sentinel oracle are included in this census. Current status allocation
per owned call is an explicit retention optimization gap. No statistical
speedup is claimed; exact proof/provenance lives in the ignored outer
`plans/.pcu-validation/2026-10-01/rocm-cuda-wide-tensor/final-matched` archive.
