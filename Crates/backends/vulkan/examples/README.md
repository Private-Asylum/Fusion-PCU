# Vulkan backend examples

`compute-copy` is the existing limited synchronous Vulkan/SPIR-V copy proof.
It creates a window, selects a Vulkan device, lowers a legacy PCU dispatch and
checks the copied values after completion. It demonstrates the backend's fixed
three-buffer prototype; it does not claim modern `#[pcu]` facade integration or
checked floating-point arithmetic support.

Run from the repository root with a Vulkan loader, compatible GPU and window
system available:

```sh
cargo run -p fusion-pcu-vulkan --example compute-copy \
    --features hosted --release
```

The `hosted` feature gates this example. The SPIR-V regression proving that
unsupported checked floating-point arithmetic is rejected belongs to the
`fusion-pcu-spirv` test suite:

```sh
cargo test -p fusion-pcu-spirv --test checked_float_rejection
```

The example and test are Cargo targets within their owning packages, with no
separate example package or change to their numerical contracts.
