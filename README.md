# Fusion PCU - Universal Peripheral Compute Unit

Fusion PCU is a Rust framework for describing work through a backend-neutral
intermediate representation and executing it on capable devices. Per-function
`#[pcu]` annotations capture supported Rust expressions. Borrows, moves and
shapes describe resource access; PCU manages discovery, RAM staging, resident
values and prepared execution.

The ultimate goal above this is no trait objects, no dynamic dispatch,
nanoscopic overhead if any, while retaining comprehensive runtime backend and
device selection. I refuse compromises to any of the above.

This is heavily inspired by projects like [rust-gpu](https://rust-gpu.github.io)
and others from which I desired a rust-native semantic library in which native
rust could be used to perform work against coprocessors, gpus, etc. This project
started as part of a personal operating system project written in rust, as a way
to write IR against virtual machines (ACPI/AML), cortex-m PIO, and others. PIO
was never meant to be a compute unit, but two's complement and masking allowed
me to get decrement, and others, but that's a different subject and a different
project. I decided to pull this project out and turn it into its own super
library. I'm starting with generic GPU compute, tensor math, and I'm using this
to learn AI inference, training, writing model executors, and even experiment
with deterministic float processing and many adjacent fields. This project is
ambitious, but it has turned into something truly nasty, and my goal is to make
it competitive with modern libraries while allowing you to write GPU-native code
in native rust without ever having to worry about the backend. Broader shader
work through SPIR-V/Vulkan, WebGPU, and other APIs is still ahead. If you've
read this far, I hope this project interests you and thank you for checking it
out!

This project is experimental. ROCm and CUDA execute hosted invocation kernels
and owned tensor compositions, including checked arithmetic, opt-in strict
F32/F64 MatMul, MSE, SGD and bounded reverse-mode training. CPU and Vulkan also
execute bounded strict training graphs; SPIR-V supplies Vulkan's shader
lowering. Metal and MLX execute bounded strict F32/F64 training graphs through
ordinary annotated functions, including loss, backward and SGD, with resident
values and original-stage fault reporting. MLX also supports checked scalar
arithmetic, six-format owned binary compositions and an explicitly permitted
native F32 MatMul path through the global facade. Broader Apple graph shapes
and low-format graph operations are still being qualified. The remaining
backend crates are scaffolds.
Default checked boundary MatMul, MSE and SGD, and owned tensor clamp, remain
unsupported until their numerical and ownership contracts are implemented.
Explicit native permissions admit additional vendor operations. The example
below targets the current checkout; published versions may lag.

## Use it today

Compile the providers you want; selection happens at runtime. There is no
implicit CPU fallback. Enable just `rocm` or `cuda` if your application supports
only one.

```toml
# Your application's Cargo.toml; adjust the checkout path.
[dependencies.fusion-pcu]
path = "../Fusion-PCU/Crates/fusion-pcu"
features = ["rocm", "cuda", "tensor"]
```

**Strictness controls checking granularity, not whether arithmetic is safe.** By
default, compound primitives must report faults at their specified result
boundary. `#[pcu(flag(strict))]` instead checks each prescribed constituent
operation: strict MatMul visits K in order and separately rounds and checks
every multiplication and addition. Strict SGD separately rounds and checks the
learning-rate multiplication and weight subtraction. Neither substitutes FMA.
Default checked boundary MatMul is currently unsupported; turning strict off
does not enable unchecked BLAS.

In both modes, supported integer arithmetic rejects overflow and underflow, and
checked division rejects zero divisors. Supported floating-point arithmetic
rejects invalid operands and overflow; the default IEEE underflow policy accepts
exact subnormals but reports tiny, inexact results. Supported scalar arithmetic
and checked casts keep their individual checks in either mode. Faults are lifted
through the call's `Result`, and fatal faults prevent output publication.
Permitted behavior or observable clamp recovery requires an explicit supported
policy; it is never implied by disabling strict.

This example passes ordinary Rust borrows into strict SGD, then composes generic
strict matrix multiplication, checked bias addition and a consuming activation.
PCU stages the host inputs automatically. Updated weights and results stay
resident between calls; explicit readback copies the final matrix into stack
RAM. A second invocation kernel demonstrates a grid-stride loop and a scalar
helper. These compositions currently require ROCm or CUDA; enabling another
provider does not grant it the same operations.

```rust
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu(flag(strict))]
fn update<T: PcuScalar, const R: usize, const C: usize>(
    weights: &[[T; C]; R],
    gradients: &[[T; C]; R],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradients, 0.5_f32)
}

#[pcu(flag(strict))]
fn linear<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    input: &[[T; K]; R],
    weights: &[[T; C]; K],
    bias: &[[T; C]; R],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::matmul(input, weights)? + bias)
}

#[pcu]
fn activate(
    input: PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::relu(input)
}

#[pcu]
fn affine(value: f32) -> f32 {
    value * 2.0 + 1.0
}

#[pcu(invocations = 2)]
fn map_affine<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = affine(input[id]);
        id += stride;
    }
}

fn main() -> Result<(), PcuExecutionError> {
    // Defaults already discover compiled providers and select a capable device.
    // fusion_pcu::global::use_defaults()?;
    // Optional runtime override; omit this to keep automatic selection:
    // fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
    //     backend: fusion_pcu::global::PcuBackendChoice::Rocm,
    //     device: Some(0),
    //     ..Default::default()
    // })?;
    let input = [[1.0_f32, 2.0, 3.0], [-1.0, 0.0, 2.0]];
    let weights = [[1.0_f32, -2.0], [0.0, 1.0], [2.0, 1.0]];
    let gradients = [[2.0_f32, -2.0], [0.0, 2.0], [2.0, 0.0]];
    let bias = [[-8.0_f32, 2.0], [1.0, -5.0]];
    let mut matrix = [[0.0_f32; 2]; 2];
    {
        // The first useful call discovers a capable provider and automatically
        // stages both RAM borrows. No separate upload/staging call is needed.
        // The returned owner retains updated device bytes, without readback;
        // the original weights in RAM remain unchanged.
        let trained_weights = update::<f32, 3, 2>(&weights, &gradients)?;
        
        // Updated weights bind by borrow; input and bias are staged from RAM.
        // MatMul/Add intermediates stay inside the graph, governed by liveness.
        let result = linear::<f32, 2, 3, 2>(&input, &trained_weights, &bias)?;
        
        // Move the owner: ReLU may reuse exclusive backing when legal.
        let result = activate(result)?;
        
        // Completed calls no longer need the updated weights owner.
        drop(trained_weights);
        
        // Explicit device -> RAM boundary, into a matrix on the stack.
        result.read_into(matrix.as_flattened_mut())?;
        
        // Scope exit drops the result owner. Quiescent backing may be freed
        // or recycled; prepared caches have their own storage lifetime.
    }
    println!("Matrix in stack RAM: {matrix:?}"); // [[0.0, 4.0], [3.0, 0.0]]
    assert_eq!(matrix, [[0.0, 4.0], [3.0, 0.0]]);

    // Stack input is staged to the device. Completed output returns to RAM;
    // invocation staging can then be released or reused by the prepared cache.
    // Two invocations traverse four values; the helper expands on device.
    let mut mapped = [0.0_f32; 4];
    map_affine::<4>(matrix.as_flattened(), &mut mapped)?;
    println!("Mapped in stack RAM: {mapped:?}"); // [1.0, 9.0, 7.0, 1.0]
    assert_eq!(mapped, [1.0, 9.0, 7.0, 1.0]);
    Ok(())
}
```

The annotated calls actually execute. Successful synchronous calls have reached
terminal completion; device owners keep escaped results live until they are
consumed or dropped. Inside a captured graph, PCU manages intermediate liveness.
The second kernel deliberately starts from the stack readback, so it crosses a
new RAM/device boundary. Chaining resident values instead avoids that transfer.

`#[pcu]` is per function. On ROCm and CUDA, owned compositions currently support
bounded helpers, homogeneous F32/F64 identity, ReLU, Add/Sub/Mul/Div, strict
MatMul and strict SGD. Invocation kernels support symbolic invocation counts,
context intrinsics, bounded scalar helpers, grid-stride loops and checked
conversions. This is a supported Rust subset, not arbitrary Rust compilation.

Arithmetic faults return `Result` by default. Exact subnormals succeed; tiny and
inexact results error under the default IEEE underflow policy. `flag(strict)`
checks each prescribed compound arithmetic step, independently of underflow and
range policy; it does not promise exact dot products or whole-model cross-vendor
determinism. Function flags and `global::configure` can select explicit
policies. Invocation `flag(clamp_range)` writes completed recovered output and
returns its observable range fault; fatal errors discard staged output.

## Benchmarks

Release Criterion builds compare actual annotated calls with native controls.
Each section states its numerical contract and measured boundary. Historical
MatMul/SGD summaries and refreshed representative slices retain their capture
dates. CPU, resident GPU execution and full host calls are separate comparisons.

### Strict MatMul — resident sweeps

These summaries compare actual `#[pcu(flag(strict))]` calls with native launches
of the **identical ordered checked MatMul kernel**. For each precision,
best/worst means the lowest/highest observed paired source/native ratio across
the three tested shapes: 4×2×4, 32×32×32 and 64×256×64 (R×K×C). These are
measured cases within this sweep, not universal overhead bounds.

Times are complete calls with resident inputs and fresh outputs: allocation,
launch, terminal wait/status and output release are included. Preparation, input
refresh, payload transfers and correctness oracles are excluded. Inputs change,
and output bits and fault/retry behavior are verified. Brackets give Criterion
95% confidence intervals in microseconds. Paired ratios are separate medians of
36 interleaved triples covering all six route orders, with no paired confidence
interval; they are not ratios of the displayed estimates. Ratios above one mean
PCU took longer. Below-native estimates and noisy small cases do not establish
speedups.

#### ROCm — Radeon RX 6900 XT (September 29, 2026)

**Best observed cases**

| Scalar | Shape | PCU µs [95% CI] | Native µs [95% CI] | Paired PCU/native |
|---|---|---:|---:|---:|
| F32 | 64×256×64 | 379.33 [378.54–380.09] | 374.65 [373.99–375.14] | 1.0085× |
| F64 | 64×256×64 | 367.36 [366.25–368.18] | 363.62 [362.92–364.39] | 1.0101× |

**Worst observed cases**

| Scalar | Shape | PCU µs [95% CI] | Native µs [95% CI] | Paired PCU/native |
|---|---|---:|---:|---:|
| F32 | 32×32×32 | 101.95 [101.43–102.70] | 99.09 [98.43–100.02] | 1.0303× |
| F64 | 4×2×4 | 64.18 [64.01–64.39] | 67.15 [66.58–67.96] | 1.0209× |

The full 48-group sweep used 20 samples per group; process wall time was 329.32
seconds.

#### CUDA — GeForce RTX 3080 12 GiB (October 7, 2026)

The refreshed sweep measures all three shapes and both precisions, with source,
explicit IR and the same-checker native control: 18 resident routes. Displayed
times are Criterion slopes with individual 95% confidence intervals, 20 samples,
1 s warmup and 2 s requested measurement. Best/worst is selected using the
separate interleaved paired ratios described above.

**Best observed cases**

| Scalar | Shape | PCU µs [95% CI] | Native µs [95% CI] | Paired PCU/native |
|---|---|---:|---:|---:|
| F32 | 4×2×4 | 27.98 [27.81–28.19] | 39.02 [38.00–40.63] | 0.7704× |
| F64 | 4×2×4 | 28.22 [27.93–28.59] | 36.87 [36.35–37.54] | 0.7704× |

**Worst observed cases**

| Scalar | Shape | PCU µs [95% CI] | Native µs [95% CI] | Paired PCU/native |
|---|---|---:|---:|---:|
| F32 | 64×256×64 | 331.34 [328.68–335.51] | 335.76 [333.00–339.31] | 0.9781× |
| F64 | 64×256×64 | 379.00 [366.37–388.68] | 360.69 [355.16–367.69] | 0.9771× |

The large F64 paired ratio and fixed-order slope quotient fall on opposite
sides of one. They come from separate timing windows; neither should be
substituted for the other or presented as a controlled PCU speedup.

### Strict SGD — resident sweeps

Actual `#[pcu(flag(strict))]` calls update F32/F64 weights against native
HIP/CUDA launches of the same ordered arithmetic checker: separately checked
Multiply then Subtract, without FMA. Both use checked range policy and the
default IEEE underflow policy. Each call changes its inputs and verifies
complete outputs outside timing; fault and retry profiles are checked
separately.

These are resident-input calls with fresh output/status allocations, launch,
terminal wait/status and output release included. Input staging, output readback
and correctness oracles are excluded. Twenty Criterion samples per group give
95% confidence intervals in microseconds. Best/worst selects the lowest/highest
source/native **estimate quotient** per precision across the tested sizes;
unlike MatMul's paired medians, these quotients come from one fixed-order run
and have no paired confidence interval. They do not isolate wrapper cost, and
below-native or overlapping estimates do not establish speedups.

#### ROCm — Radeon RX 6900 XT (September 30, 2026)

The sweep covers 65, 65,536 and 1,048,576 elements, across source, explicit
graph and native routes at resident and full-host boundaries: 36 measured
groups.

**Best observed cases**

| Scalar | Elements | PCU µs [95% CI] | Native µs [95% CI] | Estimate quotient |
|---|---:|---:|---:|---:|
| F32 | 1,048,576 | 337.224 [333.525–341.807] | 329.763 [327.309–331.695] | 1.0226× |
| F64 | 1,048,576 | 379.015 [364.277–393.612] | 401.171 [386.144–412.250] | 0.9448× |

**Worst observed cases**

| Scalar | Elements | PCU µs [95% CI] | Native µs [95% CI] | Estimate quotient |
|---|---:|---:|---:|---:|
| F32 | 65 | 64.205 [63.766–64.734] | 61.404 [61.050–61.837] | 1.0456× |
| F64 | 65,536 | 64.960 [63.012–67.163] | 60.669 [60.460–60.905] | 1.0707× |

#### CUDA — GeForce RTX 3080 12 GiB (October 7, 2026)

The fresh sweep covers 65 and 1,048,576 elements, both precisions and both
resident/full-host boundaries: 24 measured routes. The tables below select the
lowest/highest resident source/native slope quotient per precision. Values are
Criterion slopes with individual 95% confidence intervals, 20 samples, 500 ms
warmup and 2 s requested measurement. Fixed-order quotients have no paired
confidence interval. Native and PCU use the same strict ordered arithmetic;
NVCC and NVRTC compilation can still produce different machine code.

**Best observed resident cases**

| Scalar | Elements | PCU µs [95% CI] | Native µs [95% CI] | Estimate quotient |
|---|---:|---:|---:|---:|
| F32 | 65 | 24.296 [23.739–25.141] | 30.751 [30.287–31.339] | 0.7901× |
| F64 | 65 | 23.371 [23.127–23.673] | 30.405 [30.140–30.748] | 0.7687× |

**Worst observed resident cases**

| Scalar | Elements | PCU µs [95% CI] | Native µs [95% CI] | Estimate quotient |
|---|---:|---:|---:|---:|
| F32 | 1,048,576 | 392.612 [350.028–421.284] | 387.843 [378.336–399.288] | 1.0123× |
| F64 | 1,048,576 | 825.453 [793.443–856.093] | 887.858 [864.770–916.495] | 0.9297× |

### Representative ROCm slices — October 7, 2026

Radeon RX 6900 XT, stable Rust 1.98.1, with no foreign compute owner and GPU
activity at or below 10% before each benchmark process. Values are sample
medians in microseconds with individual 95% confidence intervals. MatMul/SGD
use 20 samples per route; two-step linear training uses 30. These measurements
use fixed route order, rather than interleaved paired ratios.

Strict MatMul and SGD retain the matched ordered native checker. MatMul uses
resident inputs and fresh outputs. SGD includes both that resident boundary
and the full host call, including staging and readback. Two-step linear
training starts with resident inputs and ends with final weight readback:
source/IR publish fresh owners; native retains output banks and uses unchecked
HIP/rocBLAS compounds. Measured differences include those numerical and storage
choices. Cold preparation and correctness oracles are excluded from each
workload timer. The tiny source MatMul interval is wide; its point estimate
should not be treated as a stable overhead bound.

| Workload / boundary | PCU µs [95% CI] | Explicit IR µs [95% CI] | Native µs [95% CI] |
|---|---:|---:|---:|
| Strict MatMul 4×2×4 | 43.69 [42.65–62.56] | 39.87 [39.84–40.03] | 59.48 [59.23–60.06] |
| Strict MatMul 64×256×64 | 334.90 [333.35–335.33] | 331.32 [330.18–332.30] | 363.21 [361.01–365.11] |
| Strict SGD 65, resident | 39.53 [39.41–39.63] | 39.43 [39.37–39.65] | 58.92 [58.76–59.25] |
| Strict SGD 65, full host | 77.84 [77.42–78.31] | 77.79 [76.91–78.07] | 97.38 [97.19–98.31] |
| Strict SGD 1,048,576, resident | 262.54 [255.45–267.41] | 270.56 [266.04–276.20] | 290.29 [288.64–294.77] |
| Strict SGD 1,048,576, full host | 1,513.58 [1,480.28–1,591.40] | 1,552.09 [1,465.00–1,652.47] | 1,624.43 [1,597.54–1,653.50] |
| Two-step training 4×2 | 337.77 [337.20–338.93] | 350.17 [345.77–358.14] | 246.13 [245.91–246.71] |
| Two-step training 1024×64 | 423.16 [420.88–425.36] | 416.60 [415.22–416.92] | 315.11 [314.62–316.26] |

### Representative CPU slices — October 7, 2026

Ryzen 9 5950X, stable Rust 1.98.1, pinned to logical CPU 7 with SMT sibling 23
uncontrolled. Browser and .NET compiler activity was present: these remain
**provisional workstation-load results**. Each row compares an actual `#[pcu]`
invocation, its explicit prepared-source path, and an independently checked
native Rust implementation over host memory. Warm calls and their host outputs
are timed; preparation and correctness oracles are excluded. Values are sample
medians in microseconds with individual 95% confidence intervals, 20 samples per
route, 500 ms warmup and 2 s requested measurement. No GPU work is included.

The U64 helper rows below precede the integer lane-block optimization. Its
latest measurements appear in the separate update immediately after this table.

| Operation | Elements | PCU µs [95% CI] | Prepared µs [95% CI] | Native µs [95% CI] |
|---|---:|---:|---:|---:|
| F32 Neg | 17 | 0.057 [0.057–0.057] | 0.029 [0.029–0.029] | 0.030 [0.030–0.030] |
| F32 Neg | 1,048,576 | 555.761 [555.572–556.759] | 560.054 [557.046–585.612] | 1,574.648 [1,568.210–1,594.052] |
| U64 Add | 17 | 0.078 [0.076–0.088] | 0.039 [0.039–0.041] | 0.012 [0.012–0.012] |
| U64 Add | 1,048,576 | 641.881 [561.817–881.026] | 546.557 [508.837–577.416] | 877.156 [844.034–902.141] |
| F32 helper composition | 17 | 0.506 [0.503–0.508] | 0.459 [0.458–0.460] | 0.323 [0.323–0.324] |
| F32 helper composition | 1,048,576 | 25,128.368 [25,040.578–25,323.108] | 25,452.413 [25,237.008–25,745.783] | 20,017.065 [19,745.722–20,180.219] |
| U64 helper composition | 17 | 0.210 [0.210–0.211] | 0.170 [0.170–0.172] | 0.014 [0.014–0.014] |
| U64 helper composition | 1,048,576 | 7,350.379 [7,177.721–7,613.251] | 7,518.066 [7,193.064–7,768.983] | 809.393 [787.588–898.736] |

The pre-optimization million-element U64 helper took roughly 9.1× the native
median in this capture. The preceding October 3 capture was 14.6×. Different
workstation load and native-control drift prevent attributing that earlier
change entirely to PCU; these measurements do not isolate orchestration cost.

### CPU integer lane-block update — October 7, 2026

Generic checked integer composition now uses cold-prepared, bounded lane blocks
for eligible programs across all 14 signed/unsigned widths: 8, 16, 32, 64, 128,
256 and 512 bits. Programs must have readonly loads and store-only outputs;
small or unproven programs retain scalar execution. Any arithmetic exception,
including Clamp, replays its block in original lane/program order. Exact fault
selection and whole-call transactional publication remain unchanged.

The U64 helper computes `(input + seed) * input - input`, retaining all three
checked operations. Ordinary `#[pcu]`, prepared and independently checked native
routes use private output storage and publish only after fatal-free completion.
Inputs change per call; full-result, fault, retry and tail oracles run outside
timing. Times are complete calls over the listed elements, not per-element time.

These shorter qualification runs are separate from the median-based sweep above.
Normal release builds use 20 Criterion samples, 200 ms warmup and 500 ms
requested measurement, pinned to CPU 7. Frozen baseline/candidate executables
run in A/B/B/A order; each process keeps its source/prepared/native route order.
SMT, clocks and external desktop activity remain uncontrolled. Values below are
**ranges of point estimates across two runs**, not confidence intervals:
Criterion slopes for linear sampling, means for flat sampling in the baseline
million-element PCU routes. Timing builds contain no insights or allocation
census instrumentation.

| Elements | Before PCU µs | After PCU µs | After prepared µs | After native µs |
|---|---:|---:|---:|---:|
| 17 | 0.2034–0.2114 | 0.2165–0.2789 | 0.1737–0.1750 | 0.0131–0.0135 |
| 4,096 | 26.544–26.598 | 7.686–7.706 | 7.564–7.647 | 2.645–2.665 |
| 1,048,576 | 7,163.634–7,337.365 | 1,992.224–2,002.240 | 1,970.982–2,180.615 | 717.549–755.535 |

Ordinary PCU improves about 3.45× at 4,096 elements and 3.6× at 1,048,576; the
million-element source remains 2.65–2.78× its contemporaneous native control.
Tiny calls are a tradeoff: prepared 17-element calls move from 163.3–164.3 ns to
173.7–175.0 ns, roughly 6–7% slower. Ordinary tiny calls also vary, including a
278.9 ns cohort that did not repeat. These results do not establish a universal
overhead bound or a regression-free improvement.

Separate matched counters show 112.764 → 35.703 retired instructions per element
(68.3% less) and 32.62–34.19 → 8.71–8.77 cycles per element. Counters include
startup, warmup and final verification; they are aggregate diagnostics. A
separately gated census observes zero warm Rust heap allocations, reallocations,
frees or device rescoring across 64 changing-input calls per route and shape.
All 388 CPU tests pass, including block-boundary overflow and retry for every
integer width. This is a CPU implementation improvement, not new GPU coverage.

### CUDA low-format host slices — October 7, 2026

GeForce RTX 3080 12 GiB. An actual `#[pcu]` composition computes
`(input + constant) * scale` beside explicit IR and independently handwritten
CUDA over F16, BF16 and both FP8 formats. The profile is Strict / Checked /
Preserve, default IEEE underflow, unspecified reproducibility and Reject range.
Healthy calls include host upload, compute, completion, download and result
publication; verification and fault/lifetime probes remain outside timing.

The sweep covers 65 and 4,096 elements with scales 2 and 0.5: 16 cohorts,
48 measured routes and 960 samples. Values are Criterion **slopes**, with
individual 95% confidence intervals in microseconds, 20 samples per route.
Process wall time was 234.48 seconds. Best/worst selects the lowest/highest
PCU/native point-estimate quotient per format across those four cohorts.
These fixed-order quotients have no paired confidence interval.

Source publishes fresh owners with retained immutable producers; explicit IR
uses synchronous public staging; native uses preallocated dense input banks.
Matching results does not erase those ownership and staging costs. Allocation
census and insights instrumentation are disabled in these timing builds.

**Best observed cases**

| Scalar | Elements | Scale | PCU µs [95% CI] | Explicit IR µs [95% CI] | Native µs [95% CI] | Estimate quotient |
|---|---:|---:|---:|---:|---:|---:|
| F16 | 4,096 | 0.5 | 45.17 [43.69–46.37] | 78.60 [76.67–79.72] | 51.78 [51.39–52.15] | 0.8723× |
| BF16 | 4,096 | 0.5 | 49.87 [49.44–50.26] | 79.37 [79.00–79.76] | 51.03 [50.76–51.38] | 0.9773× |
| FP8 E4M3FN | 4,096 | 2 | 43.85 [43.54–44.23] | 69.78 [68.87–70.84] | 39.06 [38.78–39.38] | 1.1225× |
| FP8 E5M2 | 4,096 | 0.5 | 42.59 [42.16–43.04] | 68.88 [67.96–69.92] | 39.06 [38.68–39.53] | 1.0901× |

**Worst observed cases**

| Scalar | Elements | Scale | PCU µs [95% CI] | Explicit IR µs [95% CI] | Native µs [95% CI] | Estimate quotient |
|---|---:|---:|---:|---:|---:|---:|
| F16 | 65 | 2 | 37.87 [37.60–38.11] | 56.31 [55.83–56.77] | 28.57 [28.53–28.60] | 1.3259× |
| BF16 | 65 | 2 | 45.52 [45.29–45.76] | 73.78 [73.24–74.33] | 28.07 [27.96–28.20] | 1.6216× |
| FP8 E4M3FN | 65 | 0.5 | 46.81 [46.18–47.49] | 66.68 [65.86–67.60] | 24.56 [24.31–24.84] | 1.9062× |
| FP8 E5M2 | 65 | 2 | 40.57 [40.05–41.27] | 76.09 [75.36–76.69] | 24.58 [24.37–24.80] | 1.6501× |

### ROCm pageable transfer slices — October 7, 2026

Radeon RX 6900 XT. A borrowed `#[pcu]` byte identity kernel is compared with
its source-prepared IR and an independently handwritten HIP kernel. Each call
includes pageable H2D, kernel launch, terminal completion, D2H and publication.
Input refresh, output sentinel reset and complete byte/tail verification are
outside the accumulated workload duration. Values are Criterion **slopes** in
microseconds with individual 95% confidence intervals, 30 samples per route.
All six routes completed in 38.31 seconds of process wall time. The activity
preflight passed with no foreign compute owner.

| Payload | PCU µs [95% CI] | Source-prepared IR µs [95% CI] | Native HIP µs [95% CI] |
|---|---:|---:|---:|
| 4 KiB | 30.855 [30.763–30.964] | 30.916 [30.725–31.117] | 37.934 [37.698–38.227] |
| 4 MiB | 776.823 [765.729–789.956] | 742.733 [734.605–752.204] | 767.675 [747.317–796.523] |

The native kernel uses HIP runtime wrappers with retained private pageable
staging and extra host copies. This is a full call comparison, not pinned-copy
bandwidth. Source-prepared IR comes from the same annotated source; it is not
claimed as an independently handwritten graph.

The large 1024→2048→2048→1024, batch-256 two-step MLP has a genuine `#[pcu]`
entry and a bitwise-checked small CPU counterpart. Its large GPU correctness
and timing remain pending: the activity guard refused before workload launch.
Native array-output and raw compact-status correctness tests on other backends
are separate from statistical benchmarks; no latency is inferred from them.

### Guarded U32 composition — October 7, 2026

A genuine `#[pcu]` two-stage checked U32 composition uses changing host inputs,
fresh owned output, one terminal status wait and a separate payload readback.
The native control uses preallocated output and combined status/payload
readback. These differences make this a complete-call comparison, **not isolated
wrapper cost**. Both paths validate results, faults, retry and output tails
outside timing.

Values are Criterion means with individual 95% confidence intervals, 20 samples,
500 ms warmup and 2 s requested measurement per route. Forward and reverse are
separate sequential route orders, not interleaved paired measurements. The
quotients below have no paired confidence interval. ROCm's reverse cohort was
refused by the activity guard and is excluded in full.

| Backend | Order | Elements | PCU µs [95% CI] | Native µs [95% CI] | Estimate quotient |
|---|---|---:|---:|---:|---:|
| ROCm | forward | 65 | 70.86 [69.96–72.03] | 43.94 [43.42–44.56] | 1.613× |
| ROCm | forward | 4,096 | 75.73 [75.41–76.12] | 47.98 [47.71–48.29] | 1.578× |
| CUDA | forward | 65 | 45.91 [44.24–47.77] | 23.08 [22.87–23.30] | 1.989× |
| CUDA | forward | 4,096 | 45.76 [45.27–46.30] | 51.43 [50.95–51.86] | 0.890× |
| CUDA | reverse | 65 | 42.92 [39.84–46.04] | 21.66 [21.50–21.84] | 1.982× |
| CUDA | reverse | 4,096 | 46.64 [45.67–47.82] | 51.02 [50.61–51.36] | 0.914× |

Separate diagnostic builds counted one Rust allocation (96 bytes), one device
allocation, four H2D copies, two D2H copies, two kernel launches and one
terminal event wait per source call on both GPUs. Insights and census
instrumentation were disabled in the timing builds. Counts describe this
composition only.

## Run the examples

Ordinary PCU examples belong to the facade and use its public API:

```sh
cargo run -p fusion-pcu --example scalar-composition \
    --features rocm --release
cargo run -p fusion-pcu --example owned-tensor-borrows \
    --features rocm,tensor --release
cargo run -p fusion-pcu --example owned-tensor-composition \
    --features rocm,tensor --release
cargo run -p fusion-pcu --example cuda-transform \
    --features cuda --release
```

Backend-specific discovery, compilation and native utilities belong to their
backend packages. See the [example guide](Crates/fusion-pcu/examples/README.md).
The independent Snake application remains under `Examples/ai/`.

## Run the comparisons

From the repository root, with the appropriate toolkit installed and the GPU
available:

```sh
cargo bench -p fusion-pcu-rocm --bench strict_matmul --features tensor
cargo bench -p fusion-pcu-cuda --bench strict_matmul --features tensor
cargo bench -p fusion-pcu-rocm --bench strict_sgd --features tensor
cargo bench -p fusion-pcu-cuda --bench strict_sgd --features tensor

# Compact host-memory CPU composition:
cargo bench -p fusion-pcu --bench cpu_source --features cpu \
    -- '^cpu_source/(F32Neg|U64Add|F32HelperLocals|U64HelperLocals)/.*/(17|4096|1048576)$'
# Pageable host/device boundary:
cargo bench -p fusion-pcu-rocm --bench host_transfer
# Guarded U32 full-call physical-work controls:
PCU_PHYSICAL_WORK_WITNESS=1 \
    cargo bench -p fusion-pcu-rocm --bench integer_tensor_literals --features tensor
PCU_PHYSICAL_WORK_WITNESS=1 \
    cargo bench -p fusion-pcu-cuda --bench integer_tensor_literals --features tensor
# CUDA low-format retained host staging:
PCU_LOW_TENSOR_PRODUCER_STAGING_SUBSET=1 \
    cargo bench -p fusion-pcu-cuda --bench low_tensor_producers --features tensor
```

The canonical Cargo `benches/` targets use Criterion, with workload composition
separated from setup, native controls, numerical oracles and allocation
diagnostics. See the [ROCm benchmarks](Crates/backends/rocm/README.md) and [CUDA
benchmarks](Crates/backends/cuda/README.md) for supporting workloads and
execution controls.

Run these comparisons from the repository checkout. Backend development targets
use an in-tree facade dependency that is omitted from registry manifests to
avoid a publication cycle; ordinary consumer dependencies remain versioned.

Licensed under Apache-2.0 © Private Asylum LLC.
