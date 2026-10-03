# CUDA backend

`fusion-pcu-cuda` implements NVIDIA execution for Fusion PCU. Ordinary source
examples belong to the facade; backend-specific tests and paired benchmarks
belong here. Backend and device choices remain runtime policy, with no implicit
CPU fallback. The development hardware is an RTX 3080.

The facade's direct checked invocation example runs from the repository root:

```sh
cargo run -p fusion-pcu --example cuda-transform \
    --features cuda --release
```

## Criterion comparisons

These canonical Cargo benchmark targets pair actual `#[pcu]` source with
explicit graph and matched native controls. They preserve their selected
numerical contract, resource lifetime and host/resident completion boundaries.
Cold preparation, allocation census and paired diagnostics are separate from
primary timing. A usable CUDA driver/compiler and compatible hardware are
required. Check GPU activity before measurements.

```sh
cargo bench -p fusion-pcu-cuda --bench strict_matmul --features tensor
cargo bench -p fusion-pcu-cuda --bench strict_sgd --features tensor
cargo bench -p fusion-pcu-cuda --bench checked_neg --features tensor
cargo bench -p fusion-pcu-cuda --bench native_matmul --features tensor
cargo bench -p fusion-pcu-cuda --bench native_mse --features tensor
cargo bench -p fusion-pcu-cuda --bench native_sgd --features tensor
```

The native compound comparisons explicitly select backend-defined arithmetic;
they do not weaken ordinary scalar checking or imply checked training support.
Checked Strict F32/F64 SGD is available through ordinary `pcu::sgd_update` under
`flag(strict)`: separate checked rate multiplication and subtraction, with the
F32 rate widened exactly for F64. Default checked Boundary SGD remains guarded.
See [strict SGD's guide](benches/strict_sgd/README.md) and its ordinary source example:

```sh
cargo run -p fusion-pcu-cuda --example strict_sgd --features tensor --release
```

See [native MatMul's guide](benches/native_matmul/README.md) for retained Lt
plans, separate classic controls and matched full-host/resident boundaries.
See [native SGD's guide](benches/native_sgd/README.md) for its rounding/FMA
profiles, changing-input oracle, full timed boundary and diagnostic modes.

The existing low-level `checked_dispatch` and `pinned_transfer` benchmarks also
remain registered. All benchmark executables remain `[[bench]]` targets with
Criterion and `harness = false`.

## Tests

Hardware tests remain explicitly ignored in ordinary host runs. Run authored
MatMul and SGD hardware tests serially on an idle CUDA machine:

```sh
cargo test -p fusion-pcu-cuda --test native_matmul_source --features tensor,allocation-census \
    -- --ignored --test-threads=1 --nocapture
cargo test -p fusion-pcu-cuda --test native_sgd_source --features tensor \
    -- --ignored --test-threads=1 --nocapture
cargo test -p fusion-pcu-cuda --test strict_sgd_source --features tensor \
    -- --ignored --test-threads=1 --nocapture
```

The `allocation-census` feature selects separate allocator diagnostics. Primary
timing builds omit those counters. The `tensor` feature remains optional and
forwards to the facade only in the development graph; normal library builds do
not acquire that development dependency.

## Development dependency and publication

The actual-source tests and PCU/native Criterion comparisons run from this
repository checkout. They use a shared path-only facade development dependency
to avoid a backend/facade publication cycle. Cargo omits that dependency and its
feature forwarding from normalized registry manifests; production dependencies
remain versioned. The standalone backend archive does not provide a
self-contained harness for these facade-dependent development targets.

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
