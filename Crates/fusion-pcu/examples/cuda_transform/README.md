# CUDA transform

An ordinary checked `#[pcu]` function selects CUDA through the common facade and
stages caller RAM automatically.

```sh
cargo run -p fusion-pcu --example cuda-transform \
    --features cuda --release
```

Real CUDA hardware and a usable NVRTC/nvcc compiler are required. Canonical
paired source/graph/native benchmarks live in `Crates/backends/cuda/benches`;
they retain their numerical and completion contracts.
