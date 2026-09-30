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
cargo bench -p fusion-pcu-cuda --bench checked_neg --features tensor
cargo bench -p fusion-pcu-cuda --bench native_matmul --features tensor
cargo bench -p fusion-pcu-cuda --bench native_mse --features tensor
cargo bench -p fusion-pcu-cuda --bench native_sgd --features tensor
```

The native compound comparisons explicitly select backend-defined arithmetic;
they do not weaken ordinary scalar checking or imply checked training support.
See [native SGD's guide](benches/native_sgd/README.md) for its rounding/FMA
profiles, changing-input oracle, full timed boundary and diagnostic modes.

The existing low-level `checked_dispatch` and `pinned_transfer` benchmarks also
remain registered. All benchmark executables remain `[[bench]]` targets with
Criterion and `harness = false`.

## Tests

Hardware tests remain explicitly ignored in ordinary host runs. Run authored
SGD hardware tests serially on an idle CUDA machine:

```sh
cargo test -p fusion-pcu-cuda --test native_sgd_source --features tensor \
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
