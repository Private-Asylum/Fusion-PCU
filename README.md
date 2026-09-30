# Fusion PCU - Universal Peripheral Compute Unit

Fusion PCU is a Rust framework for describing work through a backend-neutral
intermediate representation and executing it on capable devices. Per-function
`#[pcu]` annotations capture supported Rust expressions. Borrows, moves and
shapes describe resource access; PCU manages discovery, RAM staging, resident
values and prepared execution. 

The ultimate goal above this is no trait objects, no dynamic dispatch, nanoscopic
overhead if any, while retaining comprehensive runtime backend and device selection.
I refuse compromises to any of the above.

This is heavily inspired by projects like 
[rust-gpu](https://rust-gpu.github.io) and others from which I desired a
rust-native semantic library in which native rust could be used to perform work
against coprocessors, gpus, etc. This project started as part of a personal
operating system project written in rust, as a way to write IR against virtual
machines (ACPI/AML), cortex-m PIO, and others. PIO was never meant to be a compute 
unit, but twos-compliment and masking allowed me to get decrement, and others, 
but that's a different subject and a different project. I decided to pull this 
project out and turn it into it's own super library. I'm starting with generic
GPU compute, tensor math, and I'm using this to learn AI inference, training,
writing model executors, and even experiment with deterministic float processing
and many adjacent fields. This project is ambitious, but it has turned into
something truly nasty, and my goal is to make it competitive with modern libraries
while allowing you to write GPU-native code in native rust without ever having to 
worry about the backend. This will eventually include shader compilation to spir-v,
web-gpu, and others. If you've read this far, I hope this project interests you
and thank you for checking it out!

To be clear, this project is experimental. ROCm and CUDA execute hosted invocation 
kernels and owned tensor compositions, including checked arithmetic and opt-in strict
F32/F64 MatMul. Other GPU backends are scaffolds. Default-boundary MatMul, MSE,
SGD, ReLU backward and owned tensor clamp remain unsupported until their
numerical contracts are implemented. The example below targets the current
checkout; published versions may lag.

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
every multiplication and addition. It cannot substitute FMA or reorder those
steps. Default-boundary MatMul is currently unsupported; turning strict off does
not enable unchecked BLAS.

In both modes, supported integer arithmetic rejects overflow and underflow, and
checked division rejects zero divisors. Supported floating-point arithmetic
rejects invalid operands and overflow; the default IEEE underflow policy accepts
exact subnormals but reports tiny, inexact results. Supported scalar arithmetic
and checked casts keep their individual checks in either mode. Faults are lifted
through the call's `Result`, and fatal faults prevent output publication.
Permitted behavior or observable clamp recovery requires an explicit supported
policy; it is never implied by disabling strict.

This example composes generic strict matrix multiplication with checked bias
addition and a consuming activation. Weights stay resident across the borrow;
only the explicit readback crosses into stack RAM. A second invocation kernel
demonstrates a grid-stride loop and a scalar helper.

```rust
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu]
fn stage_matrix<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
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
    input: PcuTensor<f32>
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
    let bias = [[-8.0_f32, 2.0], [1.0, -5.0]];
    let mut matrix = [[0.0_f32; 2]; 2];
    {
        // Ordinary calls discover a capable compiled provider and stage RAM
        // inputs.
        let device_weights = stage_matrix(&weights)?;
        // Explicit const dimensions are checked against the resident weights'
        // shape.
        // Weights bind by borrow; input and bias are staged from RAM
        // automatically.
        let result = linear::<f32, 2, 3, 2>(&input, &device_weights, &bias)?;
        // Move the owner: exclusive backing may be reused for ReLU when
        // eligible.
        let result = activate(result)?;
        // The successful calls have completed; weights can be dropped
        // independently.
        drop(device_weights);
        // Explicit device -> RAM boundary, into a matrix on the stack.
        result.read_into(matrix.as_flattened_mut())?;
        // Scope exit releases the result's device storage.
    }
    println!("Matrix in stack RAM: {matrix:?}"); // [[0.0, 5.0], [4.0, 0.0]]
    assert_eq!(matrix, [[0.0, 5.0], [4.0, 0.0]]);

    // Invocation work stages the readback and writes its completed result to
    // RAM.
    // Two invocations traverse four values; the marked helper expands on
    // device.
    let mut mapped = [0.0_f32; 4];
    map_affine::<4>(matrix.as_flattened(), &mut mapped)?;
    println!("Mapped in stack RAM: {mapped:?}"); // [1.0, 11.0, 9.0, 1.0]
    assert_eq!(mapped, [1.0, 11.0, 9.0, 1.0]);
    Ok(())
}
```

The annotated calls actually execute. Successful synchronous calls have reached
terminal completion; device owners keep escaped results live until they are
consumed or dropped. Inside a captured graph, PCU manages intermediate liveness.
The second kernel deliberately starts from the stack readback, so it crosses a
new RAM/device boundary. Chaining resident values instead avoids that transfer.

`#[pcu]` is per function. Owned compositions currently support bounded helpers,
homogeneous F32/F64 identity, ReLU, Add/Sub/Mul/Div and strict MatMul.
Invocation kernels support symbolic invocation counts, context intrinsics,
bounded scalar helpers, grid-stride loops and checked conversions. This is a
supported Rust subset, not arbitrary Rust compilation.

Arithmetic faults return `Result` by default. Exact subnormals succeed; tiny and
inexact results error under the default IEEE underflow policy. `flag(strict)`
checks each prescribed compound arithmetic step, independently of underflow and
range policy; it does not promise exact dot products or whole-model cross-vendor
determinism. Function flags and `global::configure` can select explicit
policies. Invocation `flag(clamp_range)` writes completed recovered output and
returns its observable range fault; fatal errors discard staged output.

## Benchmarks

September 29, 2026; release Criterion builds. These summaries compare actual
`#[pcu(flag(strict))]` calls with native launches of the **identical ordered
checked MatMul kernel**. For each precision, best/worst means the lowest/highest
observed paired source/native ratio across the three tested shapes: 4×2×4,
32×32×32 and 64×256×64 (R×K×C). These are measured cases within this sweep, not
universal overhead bounds.

Times are complete calls with resident inputs and fresh outputs: allocation,
launch, terminal wait/status and output release are included. Preparation, input
refresh, payload transfers and correctness oracles are excluded. Inputs change,
and output bits and fault/retry behavior are verified. Brackets give Criterion
95% confidence intervals in microseconds. Paired ratios are separate medians of
36 interleaved triples covering all six route orders, with no paired confidence
interval; they are not ratios of the displayed estimates. Ratios above one mean
PCU took longer. Below-native estimates and noisy small cases do not establish
speedups.

### ROCm — Radeon RX 6900 XT

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

### CUDA — RTX 3080 12 GiB

**Best observed cases**

| Scalar | Shape | PCU µs [95% CI] | Native µs [95% CI] | Paired PCU/native |
|---|---|---:|---:|---:|
| F32 | 64×256×64 | 339.11 [336.80–341.99] | 333.84 [332.76–334.92] | 1.0120× |
| F64 | 64×256×64 | 367.21 [364.74–369.67] | 363.12 [360.83–365.80] | 1.0101× |

**Worst observed cases**

| Scalar | Shape | PCU µs [95% CI] | Native µs [95% CI] | Paired PCU/native |
|---|---|---:|---:|---:|
| F32 | 4×2×4 | 35.79 [34.77–37.06] | 30.35 [29.90–30.94] | 1.1218× |
| F64 | 4×2×4 | 40.22 [37.10–44.27] | 30.93 [30.50–31.45] | 1.1168× |

The full 48-group sweep used 20 samples per group; process wall time was 214.27
seconds.

Both source and same-checker native use five caller-thread Rust allocations /
208 requested bytes per fresh call. Driver/device allocations are excluded; the
allocation probe is compiled out of primary timing builds. Unchecked
rocBLAS/cuBLAS, historical training, explicit graph diagnostics and
transfer-only comparisons are omitted here because they describe different or
superseded boundaries. The two GPUs were measured separately; these tables do
not establish cross-vendor throughput rankings.

## Run the comparisons

From the repository root, with the appropriate toolkit installed and the GPU
available:

```sh
cargo bench -p fusion-pcu-example-rocm --bench strict_matmul
cargo bench -p fusion-pcu-example-cuda --bench strict_matmul
```

The canonical Cargo `benches/` targets use Criterion, with workload composition
separated from setup, native controls, numerical oracles and allocation
diagnostics. See the [ROCm example](Examples/rocm/README.md) and [CUDA
example](Examples/cuda/README.md) for supporting workloads and execution
controls.

Licensed under Apache-2.0 © Private Asylum LLC.
