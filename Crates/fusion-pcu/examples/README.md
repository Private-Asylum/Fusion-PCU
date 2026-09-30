# PCU usage examples

These Cargo examples consume the public `fusion-pcu` API through per-function
`#[pcu]` annotations. They demonstrate ordinary Rust arguments, helper
composition, resident ownership and explicit readback. Backend features make
providers available; backend and device selection remain runtime policy.
Required features keep the examples out of builds that do not enable a provider.

Run from the repository root:

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

- `scalar-composition` captures individually annotated scalar helpers inside an
  invocation kernel and publishes the result into an ordinary stack array.
- `owned-tensor-borrows` composes helpers, borrows resident values, mutates an
  exclusive resident output and reads back into stack RAM before ordinary Drop.
- `owned-tensor-composition` uses const-shaped matrices, resident weights,
  explicit strict checked MatMul and a consuming activation. Strict checking
  is selected because default-boundary checked MatMul remains unsupported.
- `cuda-transform` explicitly selects CUDA, then calls an ordinary invocation
  function with borrowed host input and output.

Native runtime/toolkit and compatible hardware are required to execute these
examples. There is no CPU fallback. Building an example does not execute it.
Scalar examples use checked arithmetic; unsupported combinations return errors.

Backend discovery, native compilation and low-level interop examples live in
their backend's `examples/` directory. Matched PCU/graph/native Criterion
comparisons live in the corresponding backend's `benches/`, with allocation
census and insights separate from primary timing. The independent Snake
application remains a submodule under the repository's `Examples/ai/`.

The facade's usage examples are included in its package. Backend tests and
comparisons that depend on the facade use shared path-only development aliases
to avoid a backend/facade publication cycle. Run those targets from the
repository checkout; their facade development dependency is intentionally
omitted from normalized registry manifests.
