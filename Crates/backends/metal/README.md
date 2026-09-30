# Metal backend

This backend executes a bounded checked map profile on an actual macOS Metal device. Other
platforms return `Unsupported`; device work never falls back to CPU evaluation.

`MetalSession` opens a device, compiles fixed MSL programs with fast math disabled, and owns its
queue and resources. `MetalDiscovery` implements cold physical discovery and activation.
`MetalMemoryProvider` implements reusable shared storage and checked byte transfers. All foreign
and unsafe SDK calls reside in `ffi/ffi.rs`. Production Objective-C integration uses the typed
objc2 bindings; no manual ABI, selector registration, retain/release, pool or legacy control route remains.

## Typed Objective-C dependencies

The macOS-only dependencies are exactly `objc2 = 0.6.4`, `objc2-foundation = 0.3.2` and
`objc2-metal = 0.3.2`, all with default features disabled. Explicit features are:

| Package | Features | Declared MSRV | License |
| --- | --- | --- | --- |
| objc2 | std | 1.71 | MIT |
| objc2-foundation | std, NSString, NSError, NSObject, NSArray | 1.71 | MIT |
| objc2-metal | std, MTLDevice, MTLCommandQueue, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipeline, MTLLibrary, MTLBuffer, MTLResource, MTLTypes, MTLAllocation | 1.71 | Zlib OR Apache-2.0 OR MIT |

Generated Metal features also enable their declared Foundation feature dependencies, including
`NSEnumerator`. They do not prove OS/device availability: physical `registryID`, `hasUnifiedMemory`
and `maxBufferLength` selectors are checked at runtime before use. Explicit `PhantomData<Rc<()>>`
keeps sessions, buffers and pipelines confined to their creating thread despite protocol Send/Sync
annotations. `ProtocolObject<dyn MTLDevice>` represents Objective-C protocol typing, not a Rust
trait-object vtable. The historical ARM64 comparison verified thin pointers and Objective-C dispatch.

[Retained ownership](https://docs.rs/objc2/0.6.4/objc2/rc/struct.Retained.html) and
[autorelease pools](https://docs.rs/objc2/0.6.4/objc2/rc/fn.autoreleasepool.html) replace manual
ownership messaging. Every pool returns only retained native owners or owned Rust values. Generated
[Metal methods](https://docs.rs/objc2-metal/0.3.2/objc2_metal/trait.MTLDevice.html) translate compiler
and pipeline NSError results into owned Rust error text. Missing functions/queues/encoders remain
operational errors. Objective-C exception/catch-all features are disabled; unexpected Objective-C
exceptions are not translated into MetalError. Release builds retain typed ABI/method-family
ownership and explicit extent/session/quiescence checks; objc2 debug encoding assertions are not
claimed as release guards.

Terminal Completed and Error commands release their owners after the wait. Unknown status
quarantines the session and intentionally retains command, encoder, four buffer leases and
additional device/queue/pipeline leases. Typed bindings do not weaken status-before-publication or
shared-memory copy invariants. The focused native regression verifies malformed-MSL NSError and
missing-entry errors survive pool drain, then successfully recompiles and submits actual work.

## Supported execution

| Entry point | Admitted operations |
| --- | --- |
| `prepare_u32_kernel` | Neutral U32 identity or one checked Add/Sub/Mul map, direct or grid-stride indexing |
| `prepare_f32_unary_kernel` | Neutral F32 checked Neg or ReLU map, direct or grid-stride indexing, range policy Reject |
| `PcuHostKernelBackend` | These admitted profiles through generated `#[pcu]` prepared source callables |
| `PcuDeviceKernelBackend` | These profiles on typed resident allocations, direct GPU output writes |
| `PcuOwnedDispatchBackend` | These profiles with generation-bound device identity and owned shared allocation leases |
| `prepare_integer_map` | Low-level U32 identity, checked Add/Sub/Mul/Div |
| `prepare_f32_neg`, `prepare_f32_relu` | Low-level checked F32 encoding maps |

F32 operations use integer encoding operations exclusively. Neg flips the sign bit of finite
inputs, including signed zero and exact subnormals. ReLU retains finite positive inputs and maps
nonpositive inputs to positive zero. Nonfinite inputs fault. The neutral
`RejectSubnormalResult` policy faults on a subnormal result; default and gradual policies preserve
exact subnormal bits. There is no native F64 arithmetic implementation.

Admission rejects unsupported graphs, widths, access schemas, policies, broadcasts, raw ALU,
`MatMul`, and training before compilation. Discovery reports one bounded checked-map executor,
one compute context per device, and a shared memory domain with unknown capacity. Coarse type and
instruction capabilities do not admit arbitrary graphs; preparation checks the complete bounded
structure. The facade's optional `metal` feature selects these maps through ordinary `#[pcu]` source calls.
Automatic policy scores compatible compiled providers; explicit Metal/device requests never
substitute another provider or ordinal. Warm thread-local specializations retain their prepared
session and captured policies without another environment scan. Unproved PortableV1 requests
currently reject cold before discovery.

`MetalDiscovery::open_owned_device` activates a validated discovery reference into
`MetalOwnedDispatchBackend`. It implements neutral owned dispatch, memory-session binding, and
host source preparation. Prepared dispatch snapshots the schema and executable; every submission
checks the real session, allocation size, access, and type against metadata again. Shared memory
reports `Shared` while bound leases exist and `Exclusive` after the final lease is dropped.

## Completion and publication

Every execution allocates fresh output and one private status word per logical invocation, resets
the sentinel, dispatches, and waits for terminal completion. The host reads the entire status
array, validates the completion protocol, and selects the earliest invocation fault. This is host
status processing, not a GPU reduction; it requires no atomic64 support. There is one arithmetic
operation per admitted invocation, so operation fault ordering is unambiguous.

Prepared host source validates all binding references, types, access, and lengths before uploads.
It uploads current input on every call, publishes output only after successful completion and the
fault gate, and preserves the caller's output tail. Arithmetic failure leaves caller output
unchanged. Prepared owners outlive the original session handle. A nonterminal completion
quarantines the session and retains pending native resources rather than releasing uncertain
leases.

Owned dispatch reads resident input prefixes and computes into fresh scratch output and status.
After the terminal fault gate, it publishes the successful prefix to the bound output with a
checked host byte transfer. This output publication is host mediated; it does not claim a GPU
resource-copy capability. Padded inputs and output tails survive, including grid-stride maps
whose logical extent exceeds their physical invocation shape. Completion is synchronous and
already terminal; structured arithmetic faults remain observable completion outcomes, and bound
output is unchanged on those faults.

Typed resident source uses generated `*_prepare_device` callables and the same checked admission.
It performs no implicit input upload or output data transfer: the GPU writes directly to the
exclusive output allocation, and only fresh private diagnostic records are read by the host.
Arithmetic failure may modify its output prefix, as the neutral resident contract permits; tails
remain intact. Calls validate typed declared lengths against real allocation size, access, session,
and complete binding coverage before device work. Its prepared owner survives the activating
backend handle. Reusable shared providers also support the neutral typed allocation extension.

`PcuTensor::from_device_buffer` consumes explicitly initialized typed Metal storage into a static
facade owner without transfers or arithmetic. The initialization proof is Metal-specific: private
resource allocation uses Metal’s documented native zero-clearing for every byte and external resource imports reject. Generic typed
allocation does not promise initialization. Each import currently creates a distinct facade
affinity root; one imported owner composes with RAM source arguments, while separately imported
owners cannot share a source call even if physically related. Multiple owners require a future
shared import-session seam. Metal owned tensor graphs, MatMul and training remain unsupported.

Memory allocations currently require positive sizes divisible by four and alignment at most
four. They preserve session and pool affinity, observable backing ownership, and transfer access
permissions. Resource copy, import, mapping, and device-local guarantees are unsupported.
Capacity and process/system memory telemetry remain unknown. `maxBufferLength` is a native
single-buffer limit, not pool capacity.

## Validation and benchmark

Portable structural admission, extent, ownership, and binding tests run without a Metal device.
Hardware tests are explicitly ignored by default and run with `cargo test -p fusion-pcu-metal
-- --include-ignored` on macOS. Annotated source tests exercise current inputs, exact float bits,
fault priority, unchanged output on failure, retries, tails, and owner lifetime.

The `checked_neg` Criterion benchmark compares actual annotated F32 Neg prepared source, ordinary
global source, neutral graph, and the low-level checked Metal program for 257 and 65,536 elements. Each call changes inputs
and includes one fresh shared input upload, fresh output and private status allocation, sentinel
reset, terminal wait, full status transfer and host fault selection, and host output publication.
The low-level route uses the same integer encoding checker; it is not native floating ALU or a
vendor-library comparison. Preparation and compilation are outside per-call estimates.

The benchmark checks observed GPU utilization before measurements and blocks if the activity
query is unavailable or exceeds five percent. It uses 30 samples and 95% confidence intervals.
Run it with `cargo bench -p fusion-pcu-metal --bench checked_neg`. On other platforms it reports
that hardware is required and produces no CPU timings.

Native validation was performed on an Apple M4 with ten GPU cores, macOS 26.6.2, Xcode 26.6,
and the already installed Rust 1.94.0. Local strict Clippy validation uses Rust 1.98.1. This
qualifies the tested device and toolchains; it does not establish support for every Metal device
or general floating arithmetic.

### Fresh production typed-ABI acceptance run

Apple M4 (10 GPU cores), macOS26.6.2, Xcode26.6, installed native Rust1.94.0; local strict
feature gates used Rust1.98.1. All four routes use production objc2. Values are microseconds
per matched full host call, 30 samples and 95% confidence intervals.

| Elements | Route | Estimate us | 95% CI us |
| --- | --- | --- | --- |
| 257 | source | 167.406 | [166.317, 168.175] |
| 257 | global_source | 164.988 | [158.515, 168.851] |
| 257 | graph | 166.809 | [165.428, 167.641] |
| 257 | native_checker | 163.431 | [161.449, 164.679] |
| 65536 | source | 270.156 | [269.319, 270.872] |
| 65536 | global_source | 275.462 | [275.121, 275.806] |
| 65536 | graph | 266.727 | [264.112, 268.737] |
| 65536 | native_checker | 269.711 | [268.649, 270.490] |

Fresh build wall10.40s; final artifact refresh1.28s; exact prebuilt process wall12.72s
(user21.67s, system2.04s). GPU activity was0% before and after. This primary had no allocation
counter. Cold compilation is outside samples; ordinary global source includes its warm cache.
Three fresh native buffers per full call are established by the resource path, not an allocation
census. Short fixed-order sequential interactive samples and outliers limit generalization. Only
current Criterion `new/` estimates are reported; historical change lines are ignored. The native
checker uses the same checked integer-encoding MSL, not floating GPU ALU or vendor BLAS. No manual
comparison/control route remains. Exact measured dependency source and raw evidence are archived
outside the repository; the snapshot is not claimed to include later unrelated shared SGD edits.

The Shared staging path copies exact host byte extents into owned buffers without temporary
word vectors. Status bytes are filled in place before each submission and all status records
are inspected through a scoped immutable view only after terminal completion. The seven typed
submission leases use a fixed array; unknown completion still retains every owner permanently.
`upload_bytes` and `read_into_bytes` are bounded copying helpers, not borrowed mappings or
no-copy imports. Native zero initialization is documented by
[Apple’s buffer allocation contract](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28length%3Aoptions%3A%29?language=objc).

The optional `allocation-census` benchmark feature reports warm caller-thread Rust allocations
separately and skips Criterion timing. It does not count Objective-C, Metal, driver or device
allocations; the uninstrumented primary retains the same three physical Shared buffers per call.
