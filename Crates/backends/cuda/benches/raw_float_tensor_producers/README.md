# Raw wide floating producer transport

This target compares genuine zero-argument `#[pcu]` Constant/Uniform producers,
explicit owned graph producers and an independent CUDA Driver upload/readback
control. It transports F128/F256 encodings as data. It performs no floating
arithmetic, conversion, normalization or operand classification.

The closed shapes are `[65]`, `[4096]` and a genuine inline-const `[2, 3]` matrix
(key extent6). Fixed independent limb vectors include exceptional encodings,
negative zero, the least encoded subnormal and high bits in every limb.
Row-major bytes, shape, untouched destination tails, short-read refusal,
escaped output mutation/consumption, immutable replay, source cache-clear,
graph plan-drop and independent native-control-drop ownership are verified.

Requested headers are Boundary/Strict, Checked/BackendDefined compound
permission, Preserve/BackendOptimized precision and all three underflow
policies:24 combinations. Two carriers and three shapes give144 profiles,
288 producer lifetime witnesses and864 source/graph/native semantic keys.
Producer nodes intentionally carry no arithmetic mode or underflow rule;
their full numerical options and actual empty selected binding list are
checked. Range remains Reject and reproducibility remains Unspecified.
Core arithmetic constructors reject these opaque float types. Portable,
compact representation and GPU zeroextent admission remain refused.

The SDK control directly loads stable Driver allocation/copy/context functions.
It does not import PCU IR, lowering, rewrites, owners or emitted code. Each
route creates a fresh device output, uploads its immutable payload, reads the
whole prefix into caller storage and releases that output. SDK payloads are
independent fixed little-endian limb vectors; graph values are decoded from
those vectors, while ordinary source payloads are declared separately.

Pageable H2D may return after staging while device DMA remains pending. D2H
completes before return. The SDK owner therefore fences the legacy default
stream on unread-output Drop; successful readback makes that extra fence
unnecessary. Copy/selection/wait failures retain uncertain device allocations
and their context/library. This follows the documented
[CUDA synchronization behavior](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html).
It makes no device-loss or implicit-driver-synchronization measurement claim.

`PCU_RAW_FLOAT_PRODUCER_SEMANTICS=1` or `--test` runs correctness without
Criterion sampling. `PCU_RAW_FLOAT_PRODUCER_CPU_REFERENCE=1` runs288 ordinary
CPU producer/source lifetime rows without constructing a CUDA control. A
separate `allocation-census` build measures864 scopes of64 immutable replay
calls (55,296 calls). These are immutable replays, not changing-input requests.
Caller-thread Rust heap, wrapped PCU API and independent Driver API counts
remain separate; CUDA/internal allocator activity is outside the Rust census.
SDK rows require zero Rust heap,64 fresh native allocations/frees,64 uploads
and downloads and256 context selections; readback rows require zero extra
stream waits. Cold unread-drop and detached-owner witnesses are outside all
warm census scopes. No timing follows from correctness or census output.

The backendless/no fallback policy remains unchanged. Rank-two authoring and
CPU transport are separate shared proofs; this target must qualify its own
provider execution before any GPU matrix claim.
