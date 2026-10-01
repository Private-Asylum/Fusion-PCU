# Fusion PCU MLX

MLX is an explicitly selected delegated tensor provider for Apple silicon macOS,
alongside native objc2 Metal. Other Cargo hosts remain SDK-free and return
`UnsupportedPlatform` before foreign loading. GPU requests never run CPU tensor arithmetic.

Production calls upstream `mlx_*` C entrypoints directly through a cold typed Rust
function table. Every opaque owner, callback, symbol and foreign pointer stays under
`ffi/`. The supported family is exact mlx-c **0.7.0**, commit
`a341b4925024b88b2c593468f16e12f5e17315da`, plus the recorded minimal safety fixes,
linked against exact MLX **0.32.3**. A stable zero-argument safety ABI query rejects
unknown families before any layout-dependent foreign call. The native manifest then
checks source revision, header/runtime version, handle widths and F32 enum before
opening owners.
Unmodified installed mlx-c is rejected because its error path does not meet the
provider contract. No mlx-rs or mlx-sys dependency exists.

The full upstream C source and all 668 upstream exports are retained. Source fixes
replace centralized reporting with bounded literal-safe scoped TLS capture, contain
standard/unknown exceptions at 45 selected entrypoints, synchronize independent global
handler/data access, and make selected callback/cleanup paths nonthrowing. PCU installs
no global handler and changes no default device/stream, compiler mode/cache or allocator
limit. Cleanup captures an active scope without reentering a global handler during
teardown. The one native compatibility adaptation inserts an absent optional
`global_scale` in upstream `gather_qmm` for the 0.32.3 signature; that operation is
outside the safe admitted PCU surface.

The admitted contract remains authentic captured/selected single MatMul source,
positive dense rank-two F32 operands, Boundary granularity, explicit BackendDefined
compound arithmetic, BackendOptimized precision and Unspecified reproducibility.
Checked defaults, Strict, PortableV1, Preserve, Clamp, stronger underflow contracts,
transpose, F64, training and general graphs reject before requested tensor work.
An exported C operation or dtype enum does not establish its PCU numerical or GPU law.

Rust owns official C array/device/stream/closure/vector holders. A null empty handle
is a valid setter sentinel; successful populated holders are validated separately.
Descriptor sharing uses official `mlx_array_set`, never duplicate raw ownership.
Checked dimensions/bytes and native dtype/rank/shape/strides establish dense initialized
F32 storage. Host upload copies; no borrowed/no-copy or Metal-buffer import is offered.

Cold preparation traces upstream `mlx_compile` through a nonthrowing Rust callback
whose operation is upstream `mlx_matmul` on an explicit GPU stream. Public C has no
unmaterialized-descriptor constructor, so cold tracing uses copied zero descriptors.
A narrow extension retains the resulting validated Matmul primitive and rebinds new
inputs during warm execution. It does not mirror C++ operators. Warm replay performs
no symbol search, version discovery, trait-object dispatch, frontend reconstruction or
compiler/cache/default lookup. MLX still allocates and schedules native work.

Input clones, output, stream and loaded image survive eval + explicit-stream sync +
wait. Unknown terminal completion poisons the session and quarantines actual holders
and library. Readback proves availability/contiguity, obtains the pointer and completes
fallible cloned-holder release before copying the complete prefix; host tails and failed
outputs remain unchanged. Owners and sessions are thread-confined. Calling-thread
barriers do not contain SDK worker faults, native noexcept-destructor termination,
process failures or arbitrary driver faults.

On Apple silicon, ordinary Cargo builds prepare the pinned C and native MLX sources,
apply verified exact-source patches, build the matching runtime and Metal resource,
and keep its complete dynamic-library closure in Cargo's output. Source acquisition,
integrity checks, native compilation and runtime selection are automatic. The build
requires CMake and Xcode's compiler/Metal tools; Python and installed MLX are unnecessary.
Other targets compile without native SDKs or downloads.

Verified source archives and immutable extracted trees are reused on incremental builds;
source drift fails closed. `CARGO_NET_OFFLINE=true cargo build --offline` also prevents
native source downloads and reports a missing pin explicitly. Cargo's `--offline` flag
alone governs crate resolution; the environment setting governs this native acquisition.

```sh
cargo build -p fusion-pcu-mlx --features tensor
cargo test -p fusion-pcu-mlx --all-features
cargo clippy -p fusion-pcu-mlx --all-features --all-targets -- \
  -D warnings -W clippy::all -W clippy::pedantic -W clippy::nursery
```

After checking external GPU activity on Apple silicon:

```sh
cargo test -p fusion-pcu-mlx --all-features -- \
  --ignored --skip instrumented_ --test-threads=1
cargo run -p fusion-pcu-mlx --features tensor --example prepared_matmul
```

`MlxRuntime::load_default()` and `MlxDiscovery::discover_default()` select this matching
closure. `PCU_MLX_LIBRARY` is an optional explicit override which receives the same
source/ABI/runtime checks. Runtime loading itself performs no compilation or installation.
Cold symbol resolution and the retained library lease stay under `ffi/rust/c_api`; Rust
build helpers live under `ffi/rust/build`. The isolated native project places C ABI
headers in `ffi/native/c/include` and C++ implementation/private headers in
`ffi/native/cpp`. No source folder mixes Rust and C/C++ files. Its operator tree is
the complete prepared upstream mlx-c source, with original input files preserved.
Native generation and downloaded
source/resource payloads stay in Cargo's output rather than the packaged crate.

The default path refers to retained Cargo build output. A copied or cargo-installed
application must ship its matching runtime closure and select that deployed location.
The recorded relocated-bundle proofs qualify those exact closures separately.

The three ignored `instrumented_` tests run in separate subprocesses. Two use a
separately instrumented test image to prove unchanged host publication and actual
native-holder quarantine through controlled pre-scheduling/evaluated-data failures.
The third uses an SDK-free unknown-ABI sentinel whose manifest endpoint exits if
called, proving rejection before layout-dependent calls. Production exports no fault
hooks. These fixtures do not claim a real OOM or destructive driver-fault experiment.

Qualification uses M4/ten GPU cores, macOS26.6.2, Xcode26.6, native Rust1.94 and
local SDK-free Rust1.98.1. The automatic source-built runtime uses SDK26.5 and Metal
compiler32023.883; its deployment target is macOS26.2. Older OS/device families
remain unqualified. The actual Cargo/default-runtime example, eight GPU cases,
preparation/reuse/refusal tests, bounded failure fixtures and all 30 canonical
Criterion smoke cases passed. Source tests include actual per-function
`#[pcu]`, changing inputs, complete oracles/tails, escaped owners, source/shape/session
identity and singleton/rectangular shape borders. Discovery still has unknown physical
registry identity, capacity, isolated budgets, workspace and cost. Ordinary facade/global
opaque MLX storage integration remains shared work.

Canonical Criterion `compiled_matmul` compares actual annotated source, explicit graph,
native retained C replay, upstream frontend and public compiled wrapper with identical
physical boundaries. Cold preparation, reused inputs, full copied-host calls, balanced
order and Rust allocation census are separate. The compiled control includes real upstream
vector/closure/cache work; the frontend has no compiler-trace claim. Driver/native physical
allocations are not inferred from Rust counters. Historical timings and superseded ABI2/
ABI3 sources remain outside the live crate as revision-scoped evidence.

Opaque C layouts reduce native coupling without promising arbitrary installed versions.
The compatibility matrix records full pristine C/native pairs, observed adjacent build
failures, one unchanged fixture executable, safety status and loader/resource identities.
A same executable success fixture is weaker than full production safety/numerical acceptance.
Bundling requires the complete matching native/Metal-resource closure and relocation proof;
unified RAM does not establish arbitrary pointer wrapping, Metal interchange or isolated
capacity. Unknown future C/native releases remain unsupported until independently qualified.
