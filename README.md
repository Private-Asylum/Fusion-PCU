# Fusion PCU - Universal Peripheral Compute Unit

Fusion PCU is a Rust framework for describing work through a backend-neutral intermediate representation and executing it on capable devices. Per-function `#[pcu]` annotations capture supported Rust expressions; ordinary borrows and moves describe resource access and ownership. PCU handles device discovery, RAM staging, resident values and prepared execution.

The project is experimental. ROCm is the current implemented GPU backend; the other GPU backends are scaffolds. The examples below use the current source checkout, whose frontend has advanced beyond the published `0.0.2` release.

## Use it today

Enable the providers you want to compile. Device selection happens at runtime; there is no implicit CPU fallback. This example requires a working ROCm installation and a compatible GPU.

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
        // Bias is staged automatically. Proven-safe backing may be reused for the result.
        let result = add_bias(result, &bias)?;

        // Explicit device -> RAM boundary, into the caller's stack array.
        result.read_into(&mut output)?;
        // Scope exit drops the device result. Prepared cache storage has its own lifetime.
    }
    println!("Result in stack RAM: {output:?}"); // [1.0, 2.0, 1.0, 5.0]
    Ok(())
}
```

These calls are synchronous: a successful result has reached terminal completion. Supported owned compositions currently include homogeneous `f32`/`f64` identity, ReLU, MatMul and Add/Sub/Mul, with bounded marked helpers and shape-aware inputs. This is a bounded frontend, rather than an arbitrary Rust compiler. See [the ROCm examples](Examples/rocm/README.md) for larger compositions and execution controls.

Checked faults are the required default contract. Uniform overflow, underflow and other arithmetic-fault enforcement is still being implemented, including for existing floating-point paths; explicit per-function/global resolution flags are planned. The example and measurements below do not establish that this retrofit is complete.

## ROCm benchmarks

Latest accepted measurements from September 28–29, 2026, on an **AMD Radeon RX 6900 XT (`gfx1030`)**, with an **AMD Ryzen 9 5950X** on Arch Linux. Criterion release benchmarks used CPU affinity 8 with insights disabled. Inputs changed between samples, correctness was verified outside timing, and GPU activity was checked before each run.

- **PCU source** executes actual `#[pcu]` functions; **raw PCU** invokes the prepared executor directly.
- **Native** uses HIP kernels or rocBLAS with the corresponding ownership/allocation/completion boundary.
- Times are **microseconds per measured operation**, unless marked otherwise. **1M = 1,048,576 elements**.
- Pointwise/MatMul setup, compilation, input refresh and readback are untimed. Fresh-output and multi-owner profiles include the specified resource releases; single-donor profiles retain backing across calls. Training has its own whole-route boundary.
- **Paired ratios** are medians of actual matched sample ratios, not ratios calculated from independent Criterion estimates. The tables collect each profile's latest run, not one simultaneous sweep.

These floating-point results **predate the checked-fault retrofit**. Future checked-mode comparisons must use native controls with the same fault contract.

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

### Fresh ReLU and MatMul

MatMul includes matched output allocation, completion and release. Its additional native control uses an explicitly bound stream/rocBLAS handle prepared outside timing; the default-stream control remains visible.

| Workload | PCU source (µs) | Raw PCU (µs) | Native (µs) | Explicit-stream native (µs) |
|---|---:|---:|---:|---:|
| ReLU, 65 | 29.943 | 29.726 | 28.197 | — |
| ReLU, 1M | 231.930 | 235.380 | 237.890 | — |
| F32 MatMul, 4×2×4 | 42.862 | 39.686 | 32.361 | 38.176 |
| F32 MatMul, 256³ | 66.931 | 67.083 | 51.167 | 65.897 |
| F64 MatMul, 4×2×4 | 32.477 | 31.710 | 30.584 | 30.263 |
| F64 MatMul, 256³ | 77.544 | 81.018 | 67.351 | 68.673 |

Sequential and paired measurements show material cadence differences, particularly for tiny MatMul. Stream configuration alone has not been established as the explanation for its gap; these totals are not a measurement of Rust-only overhead.

### Training

The last training baseline is from September 28; it was not rerun during the latest ownership work. Each sample measures **two forward/loss/backward/optimizer steps**. The deep MLP is **1024→2048→2048→1024, batch 256**. These use the graph execution APIs rather than the owned-source functions above.

| Workload | PCU | Native | Paired PCU/native |
|---|---:|---:|---:|
| Linear training 4×2, output bank | 196.36 µs | 187.39 µs | 1.013× |
| Linear training 1024×64, output bank | 253.27 µs | 254.81 µs | 0.985× |
| Linear training 8192×1024, output bank | 1.4919 ms | 1.5005 ms | 1.002× |
| Deep MLP, fresh outputs | 6.5321 ms | 6.8805 ms | 1.160× |
| Deep MLP, output bank | 5.6670 ms | 6.8805 ms | 1.015× |
| Deep MLP, batched bank / queued native | 5.5108 ms | 5.3601 ms | Not measured |

### Run the comparisons

From the repository root, with ROCm installed and the GPU available:

```sh
cargo bench -p fusion-pcu-example-rocm --bench owned_multi_consuming
cargo bench -p fusion-pcu-example-rocm --bench owned_selected_owners
cargo bench -p fusion-pcu-example-rocm --bench owned_binary_donor
cargo bench -p fusion-pcu-example-rocm --bench owned_relu
cargo bench -p fusion-pcu-example-rocm --bench owned_matmul
cargo bench -p fusion-pcu-example-rocm --bench tensor_train_step
cargo bench -p fusion-pcu-example-rocm --bench tensor_mlp_train
```

Benchmarks live in the canonical Cargo `benches/` layout, with composition separated from supporting machinery. Further workload details are in [the ROCm example README](Examples/rocm/README.md).

Licensed under Apache-2.0 © Private Asylum LLC.
