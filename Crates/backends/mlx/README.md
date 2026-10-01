# Fusion PCU MLX

MLX is an explicitly selected delegated tensor runtime for Apple silicon macOS,
alongside the native Metal provider. SDK-free Cargo builds remain supported on other
hosts; execution returns `MlxError::UnsupportedPlatform` before loading a library.
An explicit GPU request never falls back to CPU tensor arithmetic.

The bounded contract is positive dense rank-two F32 MatMul with authentic PCU graph
provenance, Boundary checking granularity, explicit BackendDefined compound arithmetic,
BackendOptimized precision and Unspecified reproducibility. Checked defaults, Strict,
PortableV1, Preserve and stronger explicit underflow policies reject before work.
Transpose, F64, training and general scalar IR are not admitted.

`MlxRuntime::load` opens a trusted private bridge, checks ABI2 and exact MLX0.32.3,
and opens an explicit Metal GPU device/stream. All foreign calls remain under `ffi/`.
The C++ bridge contains calling-thread standard and unknown exceptions using bounded
per-call error storage. It leaves mlx-c's independent error handler alone and changes
no default device, default stream or global compilation mode. Cold compilation
populates and retains its own compiler-cache entry; it does not clear other clients'
cache entries. Warm retained replay bypasses the compiler wrapper and its cache lookup.

Native Metal uses `objc2`. MLX has an independent
[official API family](https://github.com/ml-explore/mlx) and
[C binding](https://github.com/ml-explore/mlx-c). The private bridge uses pinned public
C++ array/primitive APIs to retain and reconstruct one Matmul primitive. These
backend-oriented APIs may change; an exact header/runtime pin is a current bridge
constraint. One compiled bridge does not establish compatibility with arbitrary installed
MLX versions. A bounded prototype ran the unchanged compiled annotated example after
relocating an application-local exact bridge/runtime/Metal-resource bundle twice; loader
provenance confirmed the copied native libraries. Bundling a pinned closure is a deployment
option under investigation, with packaging/coexistence gates still open.

Ordinary slices copy into initialized MLX-owned arrays. Cold preparation traces the
public compiler once and checks exact shape, dtype, input topology and explicit stream.
Warm replay binds fresh inputs, creates one logical result descriptor, evaluates,
synchronizes that stream and waits before publishing an owned terminal result. MLX owns
physical allocation, kernel selection and scheduling. Host reads resolve fallible
materialization before copying bytes and preserve caller tails. Unknown completion
poisons the session and quarantines the real arrays, compiled owner, stream and library.
No borrowed no-copy array, Metal-buffer import or duplicate physical capacity is claimed.

The pinned [scheduler](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/scheduler.cpp)
records ordinary task standard exceptions, and
[Metal synchronization](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/device.cpp)
reports command-buffer errors on the caller. Calling-thread containment does not establish
universal containment of worker exceptions, process faults or every future graph.

Build the bridge explicitly using an existing exact SDK. There is no Cargo SDK fetch,
installation or global package mutation:

```sh
cmake -S Crates/backends/mlx/ffi -B /path/to/isolated/bridge-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/mlx
cmake --build /path/to/isolated/bridge-build
cargo test -p fusion-pcu-mlx --all-features
cargo clippy -p fusion-pcu-mlx --all-features --all-targets -- \
  -D warnings -W clippy::all -W clippy::pedantic -W clippy::nursery
```

After independently checking GPU activity on Apple silicon:

```sh
PCU_MLX_BRIDGE=/path/to/isolated/bridge-build/libpcu_mlx_bridge.dylib \
  cargo test -p fusion-pcu-mlx --all-features -- --ignored --test-threads=1
PCU_MLX_BRIDGE=/path/to/isolated/bridge-build/libpcu_mlx_bridge.dylib \
  cargo run -p fusion-pcu-mlx --features tensor --example prepared_matmul
```

The accepted M4 proof uses MLX0.32.3, macOS26.6.2 and native Rust1.94; seven production
GPU tests cover changing inputs, retained/escaped owners, source capture, exact binding
and shape/session rejection, native exception/retry and transactional readback. Optional
C evaluation adds one diagnostic GPU test. The isolated official SDK wheel requires
macOS26.2; this is a validation profile, not a general deployment baseline. Hardware tests
remain ignored on SDK-free hosts and require an explicit trusted bridge plus activity check.

Discovery exposes generation-scoped delegated offers with exact numerical requirements;
physical registry identity, capacity, workspace and cost remain unknown. Prepared programs
retain source Arc/ValueIds. Typed host adapters accept exact Rust `f32`, reject other
scalars and reuse canonical upload/readback without conversion or intermediate allocation.
Capture-to-prepared execution is accepted; ordinary facade direct/global MLX source and
opaque storage integration remain separate work.

Canonical Criterion `compiled_matmul` pairs authentic `#[pcu]` capture/preparation with
explicit graph and native retained controls. `c_api_comparison` adds the private C controls.
Cold setup, changing reused inputs and copied-host/full-readback boundaries are separate,
with complete oracles, live trace counts and allocation/drop parity. Activity guards support
Apple silicon only. A separate allocation census counts Rust allocations; native/driver
allocations remain unmeasured. Historical measured sources and estimates stay in external
plans/evidence and do not certify subsequent source changes.

The exact installed mlx-c0.6.0_4 audit found exception text used as a printf format,
allocation inside catch reporting and unsynchronized shared handler/data without restore.
A Rust returning handler does not repair those foreign defects. Production remains ABI2.
The opt-in `c-api-evaluation` feature uses a safety-patched selected C subset, hidden
handler image and outer exception containment; it is diagnostic, not the installed package.
Its compiled control performs an ambient SDK cache lookup. Same-SDK paired measurements
found only small adapter-specific differences and establish no broad C++ speed advantage.
