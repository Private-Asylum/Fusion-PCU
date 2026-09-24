# Fusion ROCm host smoke test

The first executable verifies the installed ROCm runtime on a real host by launching a small HIP
kernel and running a rocBLAS SGEMM. The second executes a bounded Fusion PCU Dispatch program by
lowering it to HIP source, compiling a `gfx1030` code object, launching it, and checking readback.
It also exercises the Rust ROCm backend's rocBLAS SGEMM adapter and verifies its output.
Both target the RX 6900 XT.

The PCU example enumerates visible ROCm devices at runtime. Its own example policy chooses the
device with the most physical memory, breaking ties by lowest index; set `FUSION_ROCM_DEVICE` to
an explicit device index to override that policy. PCU itself does not select or rank devices.

```sh
cargo run -p fusion-example-hosted-compute-rocm --bin fusion-rocm-smoke --release
cargo run -p fusion-example-hosted-compute-rocm --bin fusion-rocm-pcu --release
cargo run -p fusion-example-hosted-compute-rocm --bin fusion-rocm-discover --release
```

`fusion-rocm-discover` uses PCU's bounded registry facade to report backend readiness,
targets, devices, location, and coarse capabilities. It can run without a GPU and reports
`Unavailable` when the HIP runtime cannot see one. This is a discovery probe, not a kernel
execution test.

Requirements: `hipcc`, the ROCm HIP runtime and headers, rocBLAS headers/library, and accessible
`/dev/kfd` plus a render node. Set `HIPCC` to the compiler path or `FUSION_ROCM_ARCH` to change
the offload target. The execution binaries return exit code 1 on compile or runtime failure.
