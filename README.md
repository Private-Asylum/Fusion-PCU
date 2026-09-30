# Fusion PCU - Universal Peripheral Compute Unit

Fusion PCU is a Rust framework for describing work through a backend-neutral intermediate representation and executing it on capable devices. Per-function `#[pcu]` annotations capture supported Rust expressions; ordinary borrows and moves describe resource access and ownership. PCU handles device discovery, RAM staging, resident values and prepared execution.

The project is experimental. ROCm and CUDA support hosted `#[pcu]` calls and resident tensor ownership, including ordinary borrows, moves and mixed RAM/device inputs. Both support opt-in strict F32/F64 MatMul. MatMul in default Boundary mode, MSE, SGD, ReLU backward and owned tensor clamp remain unsupported until their numerical contracts are implemented. The remaining GPU backends are scaffolds. The examples below use the current source checkout; published releases may lag its frontend.

## Use it today

Enable the providers you want to compile. Device selection happens at runtime; there is no implicit CPU fallback. This example requires a working ROCm installation and a compatible GPU.

For custom selection, discovery exposes executable capabilities separately from optional cold `device_facts` queries. Facts include namespaced stable identity, native processor/warp counts and launch/storage limits; unknown fields remain `None`. Consumers choose their own ranking. These queries open no device sessions or streams and add no work to warm execution.

```toml
# Your application's Cargo.toml; adjust the path to your Fusion-PCU checkout.
[dependencies]
fusion-pcu = { path = "../Fusion-PCU/Crates/fusion-pcu", features = ["rocm", "tensor"] }
```

```rust
use fusion_pcu::pcu;
use fusion_pcu::{
    PcuExecutionError,
    PcuTensor,
};

#[pcu]
fn transform(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::relu(input)
}

#[pcu]
fn add_bias(
    input: PcuTensor<f32>,
    bias: &[f32; 4],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(input + bias)
}

fn main() -> Result<(), PcuExecutionError> {
    // Defaults already apply: discover compiled providers and select a capable device.
    // fusion_pcu::global::use_defaults()?;
    // Runtime preferences can instead be set through fusion_pcu::global::configure(...).

    let input = [-2.0_f32, 1.0, -3.0, 4.0];
    let bias = [1.0_f32; 4];
    let mut output = [0.0_f32; 4];
    {
        // Executes transform: input is staged from RAM automatically.
        // The returned owner retains its initialized device-resident result.
        let result = transform(&input)?;

        // Moves that resident owner into add_bias; it does not round-trip through RAM.
        // Bias is staged automatically. Checked Add currently uses fresh result backing;
        // the consumed input remains retained until device use has completed.
        let result = add_bias(result, &bias)?;

        // Explicit device -> RAM boundary, into the caller's stack array.
        result.read_into(&mut output)?;
        // Scope exit drops the device result. Prepared cache storage has its own lifetime.
    }
    println!("Result in stack RAM: {output:?}"); // [1.0, 2.0, 1.0, 5.0]
    Ok(())
}
```

These calls are synchronous: a successful result has reached terminal completion. Supported owned compositions currently include homogeneous `f32`/`f64` identity, ReLU, strict MatMul and Add/Sub/Mul/Div, with bounded marked helpers and shape-aware inputs. This is a bounded frontend, rather than an arbitrary Rust compiler. See [the ROCm examples](Examples/rocm/README.md) for larger compositions and execution controls.

Checked faults are the required default contract. Integer Add/Sub/Mul and finite F32/F64 Add/Sub/Mul/Div return normalized arithmetic faults. Floating underflow follows IEEE tininess after rounding: exact subnormals succeed; tiny, inexact results error. Invocation `f64 as f32` casts use checked conversion: finite values round to nearest, ties to even; nonfinite operands, overflow and policy-rejected underflow return faults. `f32 as f64` widens finite values exactly and reports nonfinite operands. Remaining conversions and library operations still need the same retrofit; broader numeric compliance is not claimed.

Owned F32/F64 functions can select `#[pcu(flag(ieee_underflow))]`, `#[pcu(flag(reject_subnormal_result))]` or `#[pcu(flag(allow_gradual_underflow))]`. The latter permits gradual rounded underflow results; it does not permit overflow or invalid operands. Unannotated helpers use the global default independently of their caller's flag. Set that default with `global::configure(PcuExecutionPolicy { float_underflow: PcuFloatUnderflowPolicy::AllowGradualUnderflow, ..Default::default() })`. Policies are captured coherently during preparation and participate in kernel cache identity. Explicit backend `_prepare`/`_prepare_device` and manual `_ir` construction are pinned to the function flag or IEEE default; global configuration applies to direct automatically dispatched calls. Invocation F32/F64 kernels use checked arithmetic too: their selected policy applies throughout the compiled kernel, including scalar helper expansion. Owned helper policies retain the independent lexical behavior described above.

Invocation F32/F64 kernels can opt into `#[pcu(invocations = N, flag(clamp_range))]`. A recovered overflow continues with signed maximum finite; a recovered underflow continues with its rounded gradual result. After terminal completion, the call writes the completed result into `&mut output` and returns an arithmetic error whose `recovered_range_fault()` reports the recovery. Fatal faults still discard staging output. `PcuExecutionPolicy::range_policy` supplies the global default for automatic calls. Integer invocation clamp and owned tensor recovery are not supported yet and reject this policy explicitly; scalar core clamped integer arithmetic is available separately.

### Strict compound arithmetic

Checking granularity is independent of underflow and range policy. The default `PcuNumericalMode::Boundary` requires checking at the declared compound operation's result boundary; it does not permit unchecked arithmetic. MatMul's Boundary implementation is not yet admitted on either provider. Existing explicit scalar arithmetic stays individually checked in both modes.

```rust
use fusion_pcu::pcu;
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu(flag(strict))]
fn product<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    lhs: &[[T; K]; R],
    rhs: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(lhs, rhs)
}

fn main() -> Result<(), PcuExecutionError> {
    let lhs = [[1.0_f32, 2.0], [3.0, 4.0]];
    let rhs = [[5.0_f32, 6.0], [7.0, 8.0]];
    let mut output = [0.0_f32; 4];
    {
        // RAM inputs are staged automatically; the result remains device-resident.
        let result = product(&lhs, &rhs)?;
        result.read_into(&mut output)?;
        // The device owner drops here, after terminal completion and readback.
    }
    println!("{output:?}"); // [19.0, 22.0, 43.0, 50.0]
    Ok(())
}
```

The admitted strict route supports F32/F64, positive rank-two shapes and both transpose flags. Each output cell traverses K in increasing order, separately rounds and checks multiplication then addition, and reports the output index, reduction index and failing step. It uses generated device work with terminal fault delivery; it does not synchronize with the host after each arithmetic operation. FMA, reassociation and vendor BLAS are not substituted for this sequence. A fault prevents output publication and dependent work. This is an ordered arithmetic contract, not an exact-dot guarantee.

For automatic calls, `global::configure(PcuExecutionPolicy { numerical_mode: PcuNumericalMode::Strict, ..Default::default() })` sets the global default. Unannotated owned helpers inherit their caller's effective numerical mode; `#[pcu(flag(non_strict))]` explicitly restores Boundary mode within that function. This inheritance concerns checking granularity; underflow flags retain their separate lexical behavior described above. Modes and underflow policies participate in prepared cache identity.

The canonical `strict_matmul` Criterion targets in both provider examples pair the actual annotated function with explicit graph execution and a native launch of the same checker. Unrestricted vendor BLAS is labeled as a different numerical contract. Historical MatMul/training timings below do not establish performance or correctness for the new strict route.

## ROCm benchmarks

### Strict F32/F64 MatMul

September 29, 2026, RX 6900 XT, Criterion release, CPU affinity 8. Shape is R×K×C.
Actual `#[pcu(flag(strict))]` source, explicit graph and native HIP use the identical ordered
checker, fresh output/status, terminal completion and timed output release. Full host includes
input upload and output readback; resident excludes payload transfers. Changing inputs and
bitwise oracles run outside timing. Each triple below is **source / graph / native**, in µs.

| Scalar | Shape | Full host µs | Resident µs |
|---|---|---:|---:|
| F32 | 4×2×4 | 100.87 / 102.24 / 99.54 | 64.53 / 64.49 / 62.03 |
| F32 | 32×32×32 | 216.89 / 229.36 / 212.85 | 101.95 / 101.89 / 99.09 |
| F32 | 64×256×64 | 457.83 / 467.10 / 469.37 | 379.33 / 376.13 / 374.65 |
| F64 | 4×2×4 | 101.35 / 102.21 / 102.32 | 64.18 / 66.08 / 67.15 |
| F64 | 32×32×32 | 209.20 / 208.67 / 203.57 | 99.97 / 102.22 / 97.33 |
| F64 | 64×256×64 | 469.18 / 476.08 / 456.36 | 367.36 / 367.00 / 363.62 |

All 48 groups passed. Twenty samples per group; process wall 329.32 seconds, initial GPU
activity 0%. These are complete API boundaries, not exclusive Rust overhead. Unrestricted
rocBLAS is a separate unchecked, preallocated-output control, not a numerically equivalent
peer. Primary graph timings predate the subsequent typed-adapter allocation reduction;
allocation changes and timing changes require separate evidence. See the example's
`strict_matmul` target for numerical preflights, paired diagnostics and allocation census.

### Checked F32/F64 ReLU

September 29, 2026, RX 6900 XT, Criterion release, CPU affinity 8. All routes execute HIP generated from the same checked PCU IR. Each iteration changes input; bitwise output and fault oracles run outside timing. Resident timing includes launch, terminal wait and status readback. Full-host timing also includes input upload and payload download.

| Scalar | Profile | Elements | Resident public / sequential / HIP µs | Full-host public / sequential / HIP µs |
|---|---|---:|---:|---:|
| F32 | Clamp recovery | 65 | 56.563 / 45.789 / 46.982 | 93.507 / 79.001 / 78.204 |
| F32 | Nominal | 65 | 54.090 / 28.998 / 30.366 | 91.257 / 58.499 / 59.155 |
| F32 | Clamp recovery | 1,048,576 | 107.035 / 92.455 / 89.238 | 618.730 / 690.594 / 611.166 |
| F32 | Nominal | 1,048,576 | 98.465 / 70.909 / 63.762 | 602.012 / 580.444 / 576.856 |
| F64 | Clamp recovery | 65 | 56.370 / 41.846 / 48.348 | 89.381 / 75.547 / 75.653 |
| F64 | Nominal | 65 | 56.419 / 30.067 / 30.501 | 88.372 / 61.149 / 56.216 |
| F64 | Clamp recovery | 1,048,576 | 109.060 / 88.669 / 98.212 | 1047.265 / 1006.208 / 1110.875 |
| F64 | Nominal | 1,048,576 | 100.917 / 66.581 / 69.440 | 1006.053 / 994.463 / 993.090 |

Public `submit` supports overlapping completions and allocates independent status storage per call: 3 Rust allocations / 120 requested bytes. The sequential owner retains private status storage: 1 / 32, matching generated HIP. Both sequential routes reset after faults and reuse the sentinel only after terminal success. Prepared invocation source calls already manage status reuse internally; consumers do not need to choose it.

These are sequential Criterion point estimates, not paired ratios or exclusive Rust overhead. Large transfer-heavy cases and recovered F64 results are noisy; below-native estimates do not establish a speedup. GPU precheck was 6%; the 361 recorded activity samples ranged from 2% to 92% during benchmark work. Total process wall time was 361.189 seconds. Updated direct/grid F32/F64 fixtures passed sequential binding rejection, consecutive successes, recovered clamp output and clean retry. Driver allocations are excluded from the Rust census.

A follow-up balanced run interleaved all six route orders, with 36 triples per type/profile/boundary. Inputs changed between triples; each peer received identical input. All 576 triples passed their output/fault oracles. At one million elements, the full-host sequential/native median per-triple ratios were 1.0013 (F32 nominal), 1.0011 (F32 recovery), 1.0010 (F64 nominal) and 1.0008 (F64 recovery). The earlier 79 µs F32 recovery gap did not persist: sequential/native medians were 517.588/517.432 µs. These diagnostics have no paired confidence intervals and do not establish exclusive CPU overhead. Process wall time was 37.033 seconds; GPU precheck was 0%.

### Explicit invocation clamp recovery

September 29, 2026, RX 6900 XT, Criterion release, CPU affinity 8. The kernel evaluates separate checked `input * 2.0 / 2.0 + 1.0` operations. Recovery cases move one signed maximum-finite input between lanes; all final output bits and recovered fault indices are checked outside timing. HIP uses the identical PCU-lowered program, so this compares orchestration rather than an independent arithmetic implementation.

| Profile | Elements | Prepared PCU µs | Direct `#[pcu]` µs | PCU-lowered HIP µs |
|---|---:|---:|---:|---:|
| Exception-free | 65 | 63.444 | 63.767 | 63.242 |
| Range recovery | 65 | 79.923 | 78.713 | 76.822 |
| Exception-free | 1,048,576 | 643.48 | 679.66 | 654.22 |
| Range recovery | 1,048,576 | 689.81 | 786.04 | 615.70 |

Full host timing includes upload, launch, terminal completion, status readback and completed output download. Every route uses one Rust allocation / 32 requested bytes, excluding driver allocations. Balanced 36-triple prepared/direct-to-HIP median ratios are 1.0119/1.0116 and 1.0111/1.0058 at 65 elements, and 1.0479/1.0077 and 0.9941/0.9814 at one million (exception-free/recovery respectively). These paired diagnostics have no confidence intervals. Large-shape estimates are noisy: the direct recovery interval is 671.26–981.10 µs; neither sequential gaps nor ratios below one establish a speedup. GPU activity was 4% before launch; total process wall time was 127.073 seconds. Recovery resets the status word on the next call; only proven prior success permits reset reuse.

### Checked F32/F64 Add

The latest inline-owned-shape pass removed the source wrapper's extra rank-one allocation. All four cases now use **5 Rust allocations / 208 requested bytes** for source, raw PCU and native. Rank 0–4 metadata stays inline; higher ranks retain owned heap storage. The matched Criterion run passed all correctness checks; CPU affinity 8, GPU precheck 8%, process wall 103.066 seconds.

| Scalar | Elements | Latest source µs | Latest raw PCU µs | Latest same-helper HIP µs |
|---|---:|---:|---:|---:|
| F32 | 65 | 84.502 | 73.492 | 57.914 |
| F32 | 1,048,576 | 320.14 | 317.48 | 303.29 |
| F64 | 65 | 77.108 | 71.585 | 61.607 |
| F64 | 1,048,576 | 494.80 | 415.11 | 326.94 |

This run was noisy: the F64 million-element source interval was 417.48–576.03 µs. Balanced 36-triple source/raw median ratios were 1.0142/1.0195 for F32 and 1.0439/1.0112 for F64 (65/1M). Allocation elimination is independently confirmed; these timings do not establish a before/after speedup. The earlier measurements below retain their original allocation profile.

September 29, 2026, RX 6900 XT, Criterion release, CPU affinity 8. F32 uses bounded integer significands and sticky rounding; F64 uses its existing bounded checker. Raw PCU and native HIP compile the **identical generated checker** for each width, so this compares orchestration rather than independent arithmetic implementation quality.

| Scalar | Elements | Source `#[pcu]` µs | Raw PCU µs | Native HIP µs |
|---|---:|---:|---:|---:|
| F32 | 65 | 74.885 | 61.529 | 52.316 |
| F32 | 1,048,576 | 292.28 | 287.03 | 279.38 |
| F64 | 65 | 68.748 | 63.799 | 54.430 |
| F64 | 1,048,576 | 352.57 | 328.31 | 311.81 |

Timed work includes fresh output/fault allocation, dispatch, terminal fault observation and release. Compilation, changing input refresh and output oracle readback are excluded. Source refresh uses cached staging kernels; raw/native refresh uses transfers. Rust census for both widths and sizes is source **6 allocations / 216 bytes**, raw/native **5 / 208 bytes**, excluding driver allocations; no reallocations occurred.

A supplemental diagnostic rotates all six route orders across 36 triples with the same current inputs:

| Scalar | Elements | Source median µs | Raw median µs | Native median µs | Median paired source/native ratio |
|---|---:|---:|---:|---:|---:|
| F32 | 65 | 65.432 | 63.352 | 54.121 | 1.2267 |
| F32 | 1,048,576 | 288.806 | 283.906 | 274.061 | 1.0644 |
| F64 | 65 | 66.951 | 64.886 | 56.222 | 1.1887 |
| F64 | 1,048,576 | 355.478 | 335.892 | 332.077 | 1.0669 |

Paired medians have no Criterion confidence interval; ratios are medians of per-triple ratios rather than ratios of medians. Preparation differs between routes, so neither table isolates pure Rust overhead or establishes arithmetic optimality. The F32 tiny-source interval was 72.627–78.579 µs; F64 one-million source was 347.01–358.94 µs. GPU activity was 6% before the run; process wall time including compilation and excluded work was 134.080 seconds. Activity sampling includes the benchmark's own work.

The earlier source allocation reduction from 9 / 592 bytes to 6 / 216 was independently confirmed. The latest inline-shape pass above removes the remaining shape allocation. Checked F32/F64 source and raw-dispatch correctness tests pass on the RX 6900 XT.

### Checked F32/F64 division

The same checked-helper comparison, using ordinary source `/`, with fresh output/fault storage and terminal fault observation. Input refresh, compilation and payload oracle readback are excluded; invalid, overflow, zero-divisor and exact-subnormal preflights pass.

| Scalar | Elements | Source `#[pcu]` µs | Raw PCU µs | Native HIP µs | Balanced paired source/native |
|---|---:|---:|---:|---:|---:|
| F32 | 65 | 65.480 | 61.905 | 52.220 | 1.2064 |
| F32 | 1,048,576 | 301.87 | 294.18 | 287.41 | 1.0561 |
| F64 | 65 | 66.814 | 61.954 | 54.001 | 1.2264 |
| F64 | 1,048,576 | 335.55 | 325.88 | 317.07 | 1.0555 |

Paired ratios come from 36 triples rotating all six route orders and have no confidence intervals. Source/raw/native Rust allocation census is 6/5/5 allocations and 216/208/208 bytes per execution, excluding driver allocations. GPU activity was 0% before launch; CPU affinity 8, process wall time 104.056 seconds. These controls measure orchestration with identical generated arithmetic, rather than independent native arithmetic optimality.

### Checked invocation conversions

Ordinary narrowing and widening casts, compared with native HIP compiling the identical checked kernel. Full host timing includes input upload, launch/completion, terminal fault readback and payload download. Changing inputs, output oracles and compilation are outside the timer.

| Direction | Elements | Prepared PCU µs | Direct `#[pcu]` µs | Native HIP µs |
|---|---:|---:|---:|---:|
| F64→F32 | 65 | 55.028 | 57.472 | 55.558 |
| F64→F32 | 1,048,576 | 740.12 | 740.17 | 721.08 |
| F32→F64 | 65 | 54.771 | 55.388 | 54.685 |
| F32→F64 | 1,048,576 | 686.37 | 685.13 | 686.10 |

Each route makes one 32-byte Rust allocation per call, excluding driver allocations. The 36-triple diagnostic rotates all six route orders. Median actual prepared/native and direct/native ratios are respectively 1.004/1.005 at narrowing 65, 0.979/0.991 at narrowing 1M, 1.008/1.004 at widening 65, and 0.996/1.013 at widening 1M. These diagnostics have no confidence intervals; differences below native do not establish a PCU speedup. GPU activity was 2% before launch; process wall time was 98.057 seconds.

### Checked invocation kernels

The typed-kernel benchmark executes `output[id] = input[id] * 2.0 + 1.0` with separate checked multiply/add operations. Native HIP compiles the identical generated kernel and uses the same retained fault-word protocol. Successful prepared calls reuse the proven sentinel; first use and fault retry reset it, and every call reads terminal fault status.

| Elements | Host prepared PCU µs | Host direct `#[pcu]` µs | Host native µs | Resident prepared PCU µs | Resident native µs |
|---:|---:|---:|---:|---:|---:|
| 65 | 53.600 | 53.493 | 53.305 | 27.738 | 27.834 |
| 1,048,576 | 522.25 | 526.64 | 529.86 | 55.005 | 55.605 |

Host timing includes input upload, checked launch/completion, status readback and payload download. Resident timing includes checked launch/completion and status handling; changing input refresh and payload readback/oracle are outside both timers. Compilation is excluded. Both host routes allocate **once / 32 Rust bytes** per call, including the direct facade; prepared resident PCU has the same census. Driver allocations are not counted.

The 36-triple balanced host diagnostic gives prepared/direct/native medians of 56.441/56.386/56.351 µs at 65 and 516.621/516.927/516.231 µs at 1M. Median actual prepared/native ratios are 1.0034 and 1.0006. The 32 alternating resident pairs give 30.880/30.405 µs and 54.701/54.716 µs, with median ratios 1.0107 and 1.0019. These diagnostics have no Criterion confidence intervals. Sequential estimates below native do not establish a PCU speedup.

CPU affinity 8, GPU activity 8% before launch, 77 activity samples median 0%/maximum 84%; total process wall **77.039 seconds** includes excluded work. All changing-input exact oracles and invalid/overflow/retry preflights passed. Earlier resident comparisons with asymmetric payload-download timing were rejected and replaced by these matched boundaries.

### Earlier profiles

Earlier accepted measurements from September 28–29, 2026, on an **AMD Radeon RX 6900 XT (`gfx1030`)**, with an **AMD Ryzen 9 5950X** on Arch Linux. Criterion release benchmarks used CPU affinity 8 with insights disabled. Inputs changed between samples, correctness was verified outside timing, and GPU activity was checked before each run.

- **PCU source** executes actual `#[pcu]` functions; **raw PCU** invokes the prepared executor directly.
- **Native** uses HIP kernels or rocBLAS with the corresponding ownership/allocation/completion boundary.
- Times are **microseconds per measured operation**, unless marked otherwise. **1M = 1,048,576 elements**.
- Pointwise/MatMul setup, compilation, input refresh and readback are untimed. Fresh-output and multi-owner profiles include the specified resource releases; single-donor profiles retain backing across calls. Training has its own whole-route boundary.
- **Paired ratios** are medians of actual matched sample ratios, not ratios calculated from independent Criterion estimates. The tables collect each profile's latest run, not one simultaneous sweep.

These floating-point results **predate the checked-fault retrofit**. Checked F32 Add/Sub/Mul now use fresh outputs and unfused execution; their historical donation/fusion figures below do not describe current defaults. Future checked-mode comparisons must use native controls with the same fault contract.

### Ownership and selected inputs

| Policy | Elements | PCU source (µs) | Raw PCU (µs) | Native (µs) | Paired source/native |
|---|---:|---:|---:|---:|---:|
| Fresh-output binary | 65 | 41.173 | 39.901 | 37.977 | 1.0638× |
| Fresh-output binary | 1M | 357.240 | 351.330 | 353.070 | 1.0154× |
| Two moved owners, donation | 65 | 37.955 | 38.096 | 35.400 | 1.0280× |
| Two moved owners, donation | 1M | 237.430 | 241.750 | 236.440 | 0.9959× |
| Three moved owners, two selected | 65 | 38.045 | — | 35.958 | 1.0813× |
| Three moved owners, two selected | 1M | 233.240 | — | 230.930 | 1.0207× |
| One moved, two borrowed | 65 | 28.354 | — | 25.357 | 1.0842× |
| One moved, two borrowed | 1M | 188.950 | — | 177.890 | 1.0391× |

All donation rows match native's **one benchmark-thread Rust allocation / 32 requested bytes**, excluding driver allocations and untimed setup. Fresh-output source/raw/native counts are **7 / 9 / 3** allocations. Allocation counts alone do not prove physical reuse; separate hardware tests verify allocation identity and alias preservation.

### Single-donor operations

A moved owner and a readonly peer permit in-place execution when graph legality, exclusivity, layout, disjointness and readiness are proven. These routes retain the donor across calls; they exclude its final release.

| Operation | PCU, 65 (µs) | Native, 65 (µs) | PCU, 1M (µs) | Native, 1M (µs) |
|---|---:|---:|---:|---:|
| Add, left donor | 15.371 | 13.694 | 26.191 | 24.758 |
| Subtract, left donor | 15.377 | 13.723 | 27.177 | 25.707 |
| Subtract, right donor | 15.895 | 13.696 | 26.711 | 25.336 |
| Multiply, left donor | 15.150 | 13.630 | 26.811 | 25.056 |
| ReLU, consuming/in-place | 14.946 | 13.792 | 23.533 | 22.737 |

### Historical fresh ReLU and MatMul

These measurements precede the checked numerical-contract retrofit. They are historical evidence,
not current checked MatMul acceptance. MatMul includes matched output allocation, completion and
release. Its additional native control uses an explicitly bound stream/rocBLAS handle prepared
outside timing; the default-stream control remains visible.

| Workload | PCU source (µs) | Raw PCU (µs) | Native (µs) | Explicit-stream native (µs) |
|---|---:|---:|---:|---:|
| ReLU, 65 | 29.943 | 29.726 | 28.197 | — |
| ReLU, 1M | 231.930 | 235.380 | 237.890 | — |
| F32 MatMul, 4×2×4 | 42.862 | 39.686 | 32.361 | 38.176 |
| F32 MatMul, 256³ | 66.931 | 67.083 | 51.167 | 65.897 |
| F64 MatMul, 4×2×4 | 32.477 | 31.710 | 30.584 | 30.263 |
| F64 MatMul, 256³ | 77.544 | 81.018 | 67.351 | 68.673 |

Sequential and paired measurements show material cadence differences, particularly for tiny MatMul. Stream configuration alone has not been established as the explanation for its gap; these totals are not a measurement of Rust-only overhead.

### Historical training

The last training baseline is from September 28; it predates the compound numerical gates.
These graph targets now reject unsupported MatMul/MSE/SGD/backward contracts at admission and
need checked implementations before rerunning. Each historical sample measures **two
forward/loss/backward/optimizer steps**. The deep MLP is **1024→2048→2048→1024, batch 256**.
These use the graph execution APIs rather than the owned-source functions above.

| Workload | PCU | Native | Paired PCU/native |
|---|---:|---:|---:|
| Linear training 4×2, output bank | 196.36 µs | 187.39 µs | 1.013× |
| Linear training 1024×64, output bank | 253.27 µs | 254.81 µs | 0.985× |
| Linear training 8192×1024, output bank | 1.4919 ms | 1.5005 ms | 1.002× |
| Deep MLP, fresh outputs | 6.5321 ms | 6.8805 ms | 1.160× |
| Deep MLP, output bank | 5.6670 ms | 6.8805 ms | 1.015× |
| Deep MLP, batched bank / queued native | 5.5108 ms | 5.3601 ms | Not measured |

## CUDA benchmarks

### Strict F32/F64 MatMul

September 29, 2026, RTX 3080 12 GiB, SM86, Criterion release. The same strict source/graph/native
comparison uses device ordinal 0 and block size 256. Times are **source / graph / native**, in µs;
the transfer, completion, fresh-output and oracle boundaries match the ROCm strict table.

| Scalar | Shape | Full host µs | Resident µs |
|---|---|---:|---:|
| F32 | 4×2×4 | 52.71 / 50.78 / 47.68 | 35.79 / 35.12 / 30.35 |
| F32 | 32×32×32 | 91.25 / 93.03 / 90.63 | 70.51 / 69.29 / 65.59 |
| F32 | 64×256×64 | 442.14 / 441.96 / 437.71 | 339.11 / 337.73 / 333.84 |
| F64 | 4×2×4 | 50.66 / 50.78 / 45.83 | 40.22 / 33.59 / 30.93 |
| F64 | 32×32×32 | 103.47 / 100.34 / 100.02 | 74.06 / 72.65 / 70.33 |
| F64 | 64×256×64 | 504.99 / 522.95 / 494.52 | 367.21 / 366.42 / 363.12 |

All 48 groups passed; 20 samples, one-second warmup and two-second measurement per group,
process wall 214.27 seconds. Large resident median paired source/native ratios were 1.0120 F32
and 1.0101 F64 over 36 triples/six route orders; paired ratios have no confidence intervals.
Tiny F64 source timing was noisy (37.10–44.27 µs). cuBLAS remains a separate unchecked,
preallocated-output control: large resident estimates were 22.26 µs F32 and 108.28 µs F64.
That comparison includes numerical/algorithm/ownership differences, not just PCU overhead.

Source and same-checker native each use 5 Rust allocations / 208 requested bytes. Primary graph
timings above used 8 / 752. The subsequent adapter reduction independently measures 6 / 304
for every profile; all 12 affected graph groups passed again, without a consistent speedup.
Instrumentation is absent from primary builds and excludes driver/device allocations.

### Checked source facade

The source-facade example compares an executable `#[pcu]` function with its prepared source entry and native CUDA compiled from the same lowered program. Separate checked multiply/add operations, device ordinal and 256-invocation block geometry match throughout.

| Elements | Full host ordinary / prepared / native | Resident ordinary / native |
|---:|---:|---:|
| 65 | 26.574 / 26.366 / 26.168 µs | 17.246 / 16.726 µs |
| 1,048,576 | 2.0639 / 2.0863 / 2.1412 ms | 38.532 / 37.822 µs |

Full-host timing includes automatic input upload, launch, terminal wait/status and output download; every measured output is verified after stopping its timer. Resident timing excludes transfers and alternates two preloaded input banks into one retained destination/status/executable for both routes. First-fault and clean retry preflights pass. The million-element host intervals overlap; sequential estimates below native do not establish a speedup. Primary process wall time was 43.521 seconds.

An independent `allocation-census` feature run measured **one Rust allocation/free, zero reallocations and 32 requested bytes per warm call** for every route at both shapes. Driver/runtime internal allocations are excluded. The caller-thread allocator probe is entirely absent from the primary build. See [the CUDA example](Examples/cuda/README.md) for exact boundaries and commands.

September 29, 2026, RTX 3080 12 GiB, SM86, Criterion release. The checked U32 Add routes compile the same generated kernel; inputs change and complete output oracles run outside timing. These retained-device measurements include launch, completion and checked-status readback, excluding upload and payload download.

| Elements | Native driver µs | Sequential PCU µs | Independent-status PCU µs | Native graph with captured reset µs |
|---:|---:|---:|---:|---:|
| 256 | 16.031 | 16.201 | 23.627 | 15.472 |
| 65,536 | 34.401 | 34.510 | 42.908 | 33.953 |

Native and sequential PCU use the same proven success-sentinel protocol; their confidence intervals overlap. Independent submission owns fresh status per completion. Graph replay resets status each replay and uses a different completion protocol, so its physical work differs. These figures do not measure the end-to-end source facade.

| Reused-allocation upload + download | Pageable synchronous | Pinned with event completion and staging reclaim |
|---|---:|---:|
| 4 KiB each direction | 10.857 µs | 15.361 µs |
| 4 MiB each direction | 1.9839 ms | 1.3021 ms |

Both transfer routes exclude allocation and include the complete roundtrip. Pinned staging adds completion/reclaim work and is beneficial here only at the larger shape. GPU activity was 0% with no foreign compute applications before execution. The full CUDA hardware suite passed 37 tests; shared ordinary source ownership/fault acceptance passed separately.

## Run the comparisons

Benchmarks are being aligned with the source contract: each workload must execute a `#[pcu]` entry, paired with any explicit graph/IR and native controls. Older explicit-only targets remain diagnostic evidence while missing frontend contracts are implemented; they do not establish source semantics or checked training parity.

From the repository root, with ROCm installed and the GPU available:

```sh
cargo bench -p fusion-pcu-example-rocm --bench owned_checked_integer
cargo bench -p fusion-pcu-example-rocm --bench owned_checked_float
cargo bench -p fusion-pcu-example-rocm --bench owned_checked_float_div
cargo bench -p fusion-pcu-example-rocm --bench checked_float_convert
cargo bench -p fusion-pcu-example-rocm --bench checked_relu
cargo bench -p fusion-pcu-example-rocm --bench strict_matmul
# Historical profiles below need checked-native control reconciliation.
cargo bench -p fusion-pcu-example-rocm --bench owned_multi_consuming
cargo bench -p fusion-pcu-example-rocm --bench owned_selected_owners
cargo bench -p fusion-pcu-example-rocm --bench owned_binary_donor
cargo bench -p fusion-pcu-example-rocm --bench owned_relu
# owned_matmul and training targets await checked compound migration.
# On a CUDA-equipped machine:
cargo bench -p fusion-pcu-example-cuda --bench source_facade
cargo bench -p fusion-pcu-example-cuda --bench strict_matmul
# Separate census validation; primary timing keeps these probes compiled out.
cargo bench -p fusion-pcu-example-cuda --features allocation-census --bench source_facade -- --test
cargo bench -p fusion-pcu-cuda --all-features --bench checked_dispatch
cargo bench -p fusion-pcu-cuda --all-features --bench pinned_transfer
```

Benchmarks live in the canonical Cargo `benches/` layout, with composition separated from supporting machinery. Further workload details are in [the ROCm example README](Examples/rocm/README.md).

Licensed under Apache-2.0 © Private Asylum LLC.
