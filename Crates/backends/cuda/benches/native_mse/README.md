# Native CUDA F32 mean squared error benchmark

This canonical benchmark compares actual `#[pcu]` source, a frozen explicit tensor graph and a private native same-policy control. Each route explicitly permits non-strict, backend-defined compound arithmetic and unspecified reduction order. Preserve and backend-optimized precision are separate cases. These permissions do not grant checked training, strict numerical behavior or portable vendor reduction order.

Cases are dense same-shape F32 arrays of 65 and 1,048,576 elements. Each warm full host call retains two device input allocations, performs two uploads of changing inputs, allocates fresh squared-error scratch and one fresh scalar output, executes separately rounded F32 subtraction and multiplication, completes its squared kernel event/wait, executes cuBLAS SASUM and F32 reciprocal/count scaling with device synchronization and pointer-mode restoration, reads one scalar and releases scratch/output. All three bind the same configured precision environment to their own ordered stream, without numerical mode retuning. There is no numerical fault-status buffer under this explicitly permitted native contract.

An untimed actual source MatMul-to-MSE chain and matching frozen graph check intermediate owner/scheduling overlap across the MatMul event and synchronous MSE boundary against independently known scalar17. A native MSE known-projection control checks that same final scalar without claiming native MatMul chain parity.

Preflight checks an independent integer/dyadic difference-square-sum oracle, scalar shape, output ownership after input arrays and thread cache drop, changed-input retry, NaN/infinity permission and cold rejection of default/strict/PortableV1/tight-underflow options. The bounded witness makes every square and partial sum exact; only the stated F32 reciprocal scale rounds. It does not certify arbitrary cuBLAS numerical results. Oracle verification is outside per-call timing.

Build from the Fusion-PCU workspace:

```sh
cargo +stable build --locked --release -p fusion-pcu-cuda --bench native_mse --features tensor --message-format=json
cargo +stable build --locked --release -p fusion-pcu-cuda --bench native_mse --features tensor,allocation-census --message-format=json
```

Select each exact executable from its Cargo JSON. Run the uninstrumented primary with `/usr/bin/time -p <primary-executable> --bench` only after authorized device activity checks. It has 30 samples, 95% confidence intervals, 500ms warmup and 2s measurement for each of twelve groups. Capture pre/post activity and full prebuilt process wall separately from build time. Preserve current Criterion `new/` estimates/samples/outliers; do not infer performance from stale historical baselines.

Run the separate census executable independently. That build performs preflight and caller-thread Rust allocation reporting, and skips Criterion timing. Rust allocation counts are diagnostic and do not measure CUDA, driver or cuBLAS allocations. The two fresh device allocations per full call are established by the resource path, not by this Rust allocator. Allocation-census must never instrument the primary measurements.

Host compilation and strict Clippy have passed with Rust1.98.1. GPU preflight, timing and census remain subject to the authorized CUDA lab hardware handoff; this file records no GPU performance result.
