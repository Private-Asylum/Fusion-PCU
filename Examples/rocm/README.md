# Fusion ROCm host smoke test

The smoke executable verifies the installed ROCm runtime by launching a small HIP kernel and
running rocBLAS SGEMM. `fusion-rocm-pcu` explicitly selects the ROCm backend and a device, then
uses PCU memory admission, transfer, owned binding, prepared Dispatch submission and completion
contracts to execute and verify a 2048-element f32 grid-stride program with 250 logical
invocations. It does not call HIP or rocBLAS directly. Backend-specific compilation and launch
remain inside `fusion-pcu-rocm`. Both target the RX 6900 XT.

The PCU example enumerates visible ROCm devices at runtime. Its own example policy chooses the
device with the most physical memory, breaking ties by lowest index; set `FUSION_ROCM_DEVICE` to
an explicit device index to override that policy. It tries each ranked device until the actual
Dispatch program prepares successfully. The tensor example similarly requires the selected
graph to assess as supported on its rocBLAS and synthesized Dispatch routes. An explicit device request never selects a
different device. PCU itself does not select or rank devices.
The example imports the ROCm crate only to register/open the explicitly chosen provider; the
workload uses PCU traits and submits twice from one prepared executable. Prepared warm-loop timing
belongs in the Cargo `dispatch` benchmark.

```sh
cargo run -p fusion-example-hosted-compute-rocm --bin fusion-rocm-smoke --release
cargo run -p fusion-example-hosted-compute-rocm --bin fusion-rocm-pcu --release
cargo run -p fusion-example-hosted-compute-rocm --bin fusion-rocm-discover --release
cargo run -p fusion-example-hosted-compute-rocm --bin fusion-rocm-tensor --release
cargo run -p fusion-example-hosted-compute-rocm --bin fusion-rocm-rtc --release
cargo run -p fusion-example-hosted-compute-rocm --bin fusion-rocm-train-step --release
cargo bench -p fusion-example-hosted-compute-rocm --bench dispatch
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor_add
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor_relu
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor_mse
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor_resident
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor_train_step
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor_mlp_train
```

`fusion-rocm-tensor` builds a small two-layer MLP forward graph
(`MatMul → Add → ReLU → MatMul`), asks an explicitly selected ROCm
session to assess it, then executes the selected output dependency chain through
PCU-managed memory, rocBLAS, and synthesized PCU Dispatch. It verifies the result against the explicit CPU reference.
Input, constant, nonempty rank-two f32 MatMul, bounded same-shape f32 Add/Sub/Mul/ReLU,
ReLU backward, and scalar MSE nodes execute on ROCm; unsupported operations return an error
without CPU fallback.
This MLP example remains forward inference.

`fusion-rocm-train-step` builds a bounded linear-regression gradient with `Graph::backward_mse` and
composes an SGD update graph. It keeps input allocations on the selected ROCm device across two
steps, keeps each updated weight tensor resident for the next step, and checks each result against
the CPU reverse-mode gradient oracle. Both steps must reduce the CPU-evaluated MSE. The example
reads each output back for verification; optimizer state and a fully device-resident training loop
remain open.

`fusion-rocm-discover` uses PCU's bounded registry facade to report backend readiness,
targets, devices, location, and coarse capabilities. It can run without a GPU and reports
`Unavailable` when the HIP runtime cannot see one. This is a discovery probe, not a kernel
execution test.

`fusion-rocm-rtc` compiles a Rust-authored `#[pcu(invocations = 250)]` kernel using
`pcu::context::global_invocation_id()` and `pcu::context::invocation_count()` across a
2048-element grid-stride loop. It compiles through HIPRTC without an explicit architecture,
loads on the selected device, and verifies
all output values. This is a backend compilation probe; the PCU-only example above remains the
consumer-facing dispatch path.

The `dispatch` benchmark compares a prepared PCU dispatch with a direct launch of an
independently handwritten HIP kernel with the same f32 add semantics, using separately allocated
resident buffers. It runs a 65-element
orchestration-heavy case and a 1,048,576-element memory-traffic case with identical launch
geometry and verified output. The generated and handwritten kernels may compile to different
machine code; interpret total-time differences as whole-route results, not pure PCU call overhead.
The Cargo `[[bench]]` targets use Criterion with a custom harness; the shared configuration uses
30 samples, a 300 ms warmup, and a two-second measurement window per case. The dispatch target
keeps paired, alternating PCU/native launches inside its Criterion measurement and reports PCU binding
construction, host submit/enqueue return, host completion wait, and blocking readback separately;
the direct HIP path reports launch return, completion wait, and readback. Compilation, module
loading, allocation, input upload, and output verification are outside the repeated dispatch
samples. Each repeated sample also reports Rust global-allocator calls and requested bytes by
binding, submit, completion-wait, and total stage on the benchmark thread. This excludes
allocations made inside the native HIP
runtime, ROCr, the kernel driver, and other threads, so it is a measure of Rust-side heap activity,
not total process or GPU allocation activity. The wait number is host synchronization latency, not
device kernel time. Readback uses 64 paired, alternating, prewarmed blocking copies into separate
preallocated host buffers; these copies are outside dispatch timing and are not a transfer
throughput benchmark. The example ranks capable discovered devices by physical memory, then stable device index.
Set `FUSION_ROCM_DEVICE` to require a specific runtime device. The target architecture comes
from runtime discovery; no GPU architecture override is needed. This command requires a visible
ROCm device and does not fall back to CPU.

The `tensor` benchmark compares PCU tensor MatMul with direct rocBLAS SGEMM through this
repository's ROCm FFI wrapper. Its 8×8 case emphasizes per-call orchestration; its 1024×1024
case emphasizes GPU work. Both use the same row-major mathematical operation, separate resident
buffers and verified output. Criterion reports host wall-time estimates including rocBLAS and
device synchronization, excluding setup, transfers and readback. These measurements do not
isolate GPU kernel duration or measure a third-party Rust ROCm crate.
The same target also compares end-to-end two-MatMul graphs at 8×8 and 256×256. Those
samples include allocation, uploads from contiguous f32 byte views, both synchronized
MatMuls, readback into f32 storage, and Tensor construction on each route; output
validation occurs before Criterion sampling.
They reveal orchestration and memory-path costs that the resident-buffer comparison excludes.
The direct native peer matches PCU's allocation and release order. The same target also times a
prepared PCU graph, with structural assessment
outside the repeated calls; each execution still validates its inputs and selected handle.

The `tensor_add` Criterion benchmark compares a PCU graph Add with a handwritten HIP Add at 65 and
1,048,576 elements. Both routes include allocation, upload, submission, completion, and readback,
with three device allocations per call and output checks before and after sampling. The PCU Add path caches prepared
kernels by flattened element count, so these samples are warm repeated executions. Preparation
is reported separately as a cold first execution; a new shape or cache eviction can incur compilation.
On the RX 6900 XT, one Criterion run estimated 85.58 µs PCU versus 73.18 µs direct HIP for 65
elements, and 1.39 ms versus 1.43 ms for 1,048,576 elements. These are whole-route host-wall
samples, not isolated GPU kernel times.

The separate `tensor_relu` target pairs PCU graph `ReLU` with a handwritten HIP kernel over
alternating negative and positive values at the same sizes. It reports first-execution cost and
warm Criterion estimates with two device allocations per call; it checks both outputs before and
after the sampled run.
The tensor executable also verifies ReLU's NaN and signed-zero behavior on the device against
the CPU reference contract.
One RX 6900 XT Criterion run estimated 72.25 µs PCU versus 59.56 µs HIP at 65 elements, and
1.22 ms versus 1.19 ms at 1,048,576 elements. First PCU execution took about 0.75 seconds.

The `tensor_mse` target compares a scalar mean-squared-error graph with a native route that
launches a squared-difference HIP kernel and uses rocBLAS SGEMM for the final mean reduction.
Both routes allocate and upload inputs and a vector of ones, allocate temporary and scalar output
buffers, synchronize both GPU operations, and read back the scalar. Alternating inputs exercise
both zero and nonzero squared differences. The CPU graph evaluates the
same inputs as a correctness oracle before and after Criterion sampling. This measures the whole
host call, including allocation and transfers; it does not isolate GPU kernel duration.

The `tensor_resident` target measures prepared MatMul graphs with input buffers uploaded once and
reused across calls. It includes the existing PCU host-input route as a control and a direct
rocBLAS route with equally resident inputs. Each timed call allocates its output, waits for SGEMM,
and reads back a dense host tensor; input uploads and graph preparation are outside the repeated
samples. The benchmark verifies every route against the CPU graph before and after sampling.

For hardware runs, use a release build and record the runtime/device context alongside results:

```sh
rocminfo
cargo bench -p fusion-example-hosted-compute-rocm --bench dispatch
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor_add
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor_relu
cargo bench -p fusion-example-hosted-compute-rocm --bench tensor_train_step
```

The `tensor_train_step` benchmark compares two prepared PCU graph executions per sample, with
persistent inputs and the first updated weight allocation fed directly into the second step,
against a direct HIP plus rocBLAS route that reuses preallocated intermediate and output buffers.
Both routes start each sample from immutable resident initial weights; native does not perform a
timed reset copy.
Both perform the same forward MatMul, prediction-gradient elementwise operations, transposed
sample-gradient MatMul, and SGD update. PCU prepares reusable graph constants and intermediate
scratch once, then allocates only the two chained outputs each sample; native reuses all buffers.
Both routes read back only the final output.
The 4×2 case emphasizes orchestration; 1024×64 exercises a moderate matrix workload; and
8192×1024 tests whether that overhead is amortized by substantially more work. Every result is
checked against two CPU `backward_mse` graph steps before and after measurement.

On an RX 6900 XT, one 20-sample Criterion run with matched rocBLAS matrix orientations,
resident initial weights, and prepared scratch measured PCU/direct HIP+rocBLAS at
232.45/144.32 µs for 4×2, 297.46/208.17 µs for 1024×64, and 1.4923/1.4154 ms for
8192×1024. The last shape has 128 times the sample elements of 1024×64; its relative gap is
about 5%, while the absolute gap is roughly 77–89 µs across the three cases. These are complete
two-step host-wall timings on one device, not isolated IR costs or portable performance guarantees.
The native gradient MatMul uses the same rocBLAS transpose orientation as PCU: another equivalent
orientation took about twice as long at 8192×1024 on this card and would distort the comparison.
An untimed diagnostic execution of the PCU two-step route observed two output allocation calls,
zero uploads, and one final download at every shape. The output allocations took about 3–4 µs
in those samples; these durations do not include resource release or isolate the remaining
kernel/synchronization costs. Scratch creation and constant uploads occur before Criterion timing.

The `tensor_mlp_train` target exercises a 1024→2048→2048→1024 dense network with batch 256,
ReLU after both hidden layers, scalar mean-squared-error loss, gradients for all three weight
matrices, and SGD updates. Each sample runs two steps, consuming the first step's device-resident
updated weights in the second. PCU prepares one union graph for the loss and three updated
weights, then reuses scratch; the native peer uses HIP kernels and rocBLAS with resident inputs
and weight banks. Both download the two losses and final weights, and the benchmark checks
numerical parity before and after timing. An independent small CPU graph checks the gradient
construction; the full-size CPU oracle would be impractical. Preparation is reported separately.
Set `FUSION_PCU_MLP_BATCH=1024` to run the larger batch; the default is 256. The target keeps
separate prepared PCU routes with fresh owned outputs and reusable output banks, alongside the
native route. PCU's first-class `SgdUpdate` performs each weight update in one HIP launch. Prepared
MSE scratch retains its squared-difference and ones buffers, and both routes defer reading the two
losses until after step two. The banked route uses two validated, non-aliasing output banks to
ping-pong weights and makes zero allocations or uploads during the measured two-step pass.

In one RX 6900 XT ten-sample Criterion run, batch 1024 measured 10.544 ms PCU fresh-output,
8.942 ms PCU banked, and 8.878 ms native; batch 256 measured 6.675, 5.251, and 5.479 ms.
A separate 16-pair order-balanced host-wall diagnostic alternated banked PCU and native call
order and found median paired ratios of 1.002× at batch 1024 and 0.997× at batch 256. Pairwise
ranges were 0.839–1.152× and 0.871–1.046× respectively, so these results support parity on this
card and workload, not a universal speed claim. The older single-thread/1×1 SGEMM loss results
and an extra PCU host-copy result are superseded. Preparation, allocation, transfer, and per-node
diagnostics run outside Criterion. Native prepared 28 resident allocations (226.5 MB). ROCm did
not report process-memory usage through the selected pool snapshot, so PCU peak live memory
remains unavailable; cumulative allocations are not a peak. Per-node measurements are synchronous
host-wall spans, not GPU event durations or isolated IR overhead.

Requirements: `hipcc`, the ROCm HIP runtime and headers, rocBLAS headers/library, and accessible
`/dev/kfd` plus a render node. Runtime architecture discovery uses the KFD topology and DRM
render-node PCI identity to match each HIP device to its base AMD `gfx` target. Optional target
feature suffixes are not reported by this lookup. This lookup currently requires Linux KFD/DRM
topology. When that target or `hipcc` is unavailable, PCU Dispatch uses HIPRTC's current-device
compilation if the runtime compiler is present; rocBLAS tensor work does not require either
source compiler. Set `HIPCC` if the compiler is not on `PATH`. The execution binaries return exit
code 1 on compile or runtime failure.
