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

Ordinary source resident outputs track logical validity separately from initialized memory and
terminal completion. Preflight/staging failure preserves the previous value; a completed
unrecovered failure after possible output writes discards that value, so `PcuTensor` read and
source borrowing reject it. Unknown completion quarantines access. Successful or recovered
publication remains readable; retry after a fatal write uses a fresh logical owner. The host
adapter reports possible writes separately from completion uncertainty for this shared rule.

## Supported execution

| Entry point | Admitted operations |
| --- | --- |
| `prepare_integer_kernel` | Neutral I32/U32/I64/U64 identity or one checked Add/Sub/Mul map, direct or grid-stride indexing |
| `prepare_u32_kernel` | Neutral U32 identity or one checked Add/Sub/Mul map, direct or grid-stride indexing |
| `prepare_float_unary_kernel` | Neutral six-format checked Neg/ReLU, canonical direct/grid, range Reject; low formats use exact bytes and F64 uses two U32 limbs |
| `prepare_f32_unary_kernel` | Neutral F32 checked Neg or ReLU map, direct or grid-stride indexing, range policy Reject |
| `prepare_float_binary_kernel` | Neutral one-op F16/BF16/F32/F64/OFP8 checked Add/Sub/Mul/Div, direct/grid-stride, exact SSA operands, range Reject |
| `prepare_f32_binary_kernel`, `prepare_f64_binary_kernel` | Width-restricted entries into that complete binary admission |
| `prepare_f32_binary`, `prepare_f64_binary`, `prepare_low_precision_binary` | Separately proved low-level exact binary integer realizations with all three underflow policies |
| `PcuHostKernelBackend` | These admitted profiles through generated `#[pcu]` prepared source callables |
| `PcuDeviceKernelBackend` | These profiles on typed resident allocations, direct GPU output writes |
| `PcuOwnedDispatchBackend` | These profiles with generation-bound device identity and owned shared allocation leases |
| `prepare_i32_map`, `prepare_i64_map`, `prepare_u64_map` | Low-level exact-width integer identity, checked Add/Sub/Mul |
| `prepare_integer_map` | Low-level U32 identity, checked Add/Sub/Mul/Div |
| `prepare_f32_neg`, `prepare_f32_relu`, `prepare_f64_neg`, `prepare_f64_relu` | Low-level checked F32/F64 encoding maps |

I32 Add/Sub classify two's-complement unsigned wrapping encodings by sign transitions.
Multiply checks unsigned operand magnitudes against the signed result limit before multiplying.
No signed overflow, signed minimum negation or native floating arithmetic is executed. Results
below `i32::MIN` are underflow; results above `i32::MAX` are overflow. This is the neutral PCU
integer law. Ordinary and Strict maps each contain one declared arithmetic operation.

F32/F64 unary operations use integer encoding operations exclusively. Neg flips the sign bit of finite
inputs, including signed zero and exact subnormals. ReLU retains finite positive inputs and maps
nonpositive inputs to positive zero. Nonfinite inputs fault. The neutral
`RejectSubnormalResult` policy faults on a subnormal result; default and gradual policies preserve
exact subnormal bits. F64 uses two U32 limbs for these exact operations; no native double ALU is required.
Each F64 invocation gets one private diagnostic record while input/output extents retain eight
bytes per element.

F64 binary Add/Sub/Mul/Div is synthesized entirely from unsigned integer encodings.
53-bit significands plus three guard/round/sticky bits fit U64; Add/Sub extended sums are
below 2^57. Multiply splits each significand into 27 low/26 high bits, retains the exact
106-bit product in two U64 limbs, and jams only after finding its leading bit. Division
emits 56 quotient bits with remainder below 2^54, using no integer division instruction.
Guarded shifts implement nearest ties-even packing and unbounded-exponent tininess after
precision rounding (IEEE 754-2019 clauses 4.3.1 and 7.5(a)). Nonfinite input, overflow,
zero divisor and policy-rejected underflow are explicit PCU faults. No native double ALU,
contraction, FTZ/DAZ assumption, native compound permission or PortableV1 claim is used.
Default/Strict one-op source, all three underflow policies, swapped SSA and grid-stride
maps have actual M4 proof. F32 binary Add/Sub/Mul/Div separately uses U32 significands, guarded jamming, exact
32x32 products from bounded 16-bit partial products, and 27-bit long division with
remainder below 2^25. Its independent MSL realization uses neither F64 nor native F32
arithmetic. Typed F32/F64 admission and host/device ownership share a static prepared
map; each width retains its own shader, opcode and full numerical proof. Broadcast,
Clamp, MatMul and training remain unsupported.
The `checked_f32` and `checked_f64` benchmarks each have 32 genuine prepared/ordinary source, neutral graph and
native integer-checker semantic peers with copied inputs, fresh output/status, terminal
completion and explicit prefix readback. Full changing-bank core oracles run outside samples.
These are correctness smoke results; existing timing tables remain historical.

F16, BF16, OFP8 E4M3FN and E5M2 binary maps use exact U16/U8 payload encodings with
U32 synthesis. Their significands have at most 11 bits: extended Add/Sub sums are below
2^15, exact products require at most 22 bits, and long division emits at most 14 quotient
bits with remainder below 2^12. Guarded ties-even packing and destination-precision tininess
match the checked core law. E4M3FN's finite final exponent and reserved NaN encodings are
classified explicitly. No native half/float arithmetic or integer division is used.
Payload allocation/copy/readback retains exact bytes, including odd FP8 and half tails;
private diagnostics remain four bytes per logical value. Unary low-format maps, conversions,
Clamp and larger graphs remain unsupported. Normal maps reject scalar broadcast; the independently
qualified PortableV1 profile below admits it. Cold preparation checks the complete numerical header.

Actual M4 low-format qualification compares 4,924,800 complete lane result bits or fault kinds:
every FP8 encoding pair, every half/BF16 encoding paired with zero and one, and separate
edge/random half probes, for all four operations and three underflow policies. Genuine
prepared/ordinary source, resident and mixed owner tests cover exact-byte extents, transactional
host failure, logical discarded owners, retries and distinct-session rejection. The
`checked_low_precision` benchmark has 128 source/graph/native semantic peers at 257 and 65,537
values, with changing exact core oracles and matched copied-input, fresh output/status,
completion and prefix-readback boundaries. Its opt-in `allocation-census` feature reports
caller-thread Rust heap calls separately from timing and excludes native/driver/device heaps.
These normal-profile results retain their original frozen source identity. The separate Portable
qualification adds requested-header proof and fixes advanced owned dispatch publication to copy exact
payload bytes, including odd FP8 extents, rather than requiring complete U32 words.

PortableV1 is admitted only for checked F16, BF16, E4M3FN and E5M2 Add/Sub/Mul/Div with Reject
range policy, matching header/node underflow, and nonzero canonical 1D direct/grid maps. Read-only
scalar broadcast, swapped operands and repeated SSA operands are included. Boundary/Strict and
compound/precision permissions retain the stronger scalar contract. F32/F64, integers, unary maps,
Clamp, larger graphs and MLX have no Portable admission from this proof. The implementation uses
defined integer operations rather than a device-family allowlist; native qualification records the
tested M4 and toolchain, without claiming whole-model or random-number reproducibility.

The independent requested-header native oracle covers 20,714,880 complete lane bits or fault kinds.
Ordinary/prepared source, device, resident, mixed and advanced owned paths additionally check
broadcast roles, direct/grid SSA mapping, exact tails, completion and failed-owner publication.
The `checked_portable` benchmark has 128 matched source/graph/native semantic peers; its separate
caller-thread Rust census reports zero allocation/reallocation/free calls in prepared, graph and
native peers, and two allocations/two frees (112 allocated bytes) per ordinary call. Native,
driver and device heaps are excluded; all peers retain fresh native input/output/status buffers.

Metal dispatch uses baseline uniform threadgroups with ceil group count and a bounds guard on
every entry before accessing payload or diagnostics. Padded lanes perform no access. The maximum
power-of-two threadgroup width is retained cold within the prepared pipeline's limit, and logical
counts must fit U32. Smaller counts use one exact group; larger padded grids never exceed 2^32
threads, preventing uint index wrap even near U32::MAX and irregular pipeline limits. This
scheduling change has fresh correctness/smoke proof; earlier statistical measurements remain
historical. The shader relies on exact U8/U16/U32 layouts, modular unsigned arithmetic, defined
unsigned shifts and nonzero `clz`; every shift count is guarded below 32, signed exponent arithmetic
stays bounded, and no integer division/modulo or native floating ALU is emitted. These contracts
are specified in Apple's [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
sections 2.1, 2.2, 3.1 and 6.3. Activation requires the queried device selectors and successful
library/pipeline compilation; no nonuniform-threadgroup feature or native low-float support is required.
The eight-byte configuration is copied by [setBytes](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:)),
and publication follows [waitUntilCompleted](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted()).

Admission rejects unsupported graphs, widths, access schemas, policies, broadcast profiles, raw ALU,
`MatMul`, and training before compilation. Discovery reports one bounded checked-map executor,
one compute context per device, and a shared memory domain with unknown capacity. Coarse type and
instruction capabilities do not admit arbitrary graphs; preparation checks the complete bounded
structure. The facade's optional `metal` feature selects these maps through ordinary `#[pcu]` source calls.
Automatic policy scores compatible compiled providers; explicit Metal/device requests never
substitute another provider or ordinal. Warm thread-local specializations retain their prepared
session and captured policies without another environment scan. Unsupported PortableV1 profiles
reject cold; neutral structural eligibility alone does not qualify another provider.

`MetalDiscovery::open_owned_device` activates a validated discovery reference into
`MetalOwnedDispatchBackend`. It implements neutral owned dispatch, memory-session binding, and
host source preparation. Prepared dispatch snapshots the schema and executable; every submission
checks the real session, allocation size, access, and type against metadata again. Shared memory
reports `Shared` while bound leases exist and `Exclusive` after the final lease is dropped.

## Bounded low-format unary maps

F16, BF16, OFP8 E4M3FN and E5M2 Neg/ReLU use a separately qualified U32 encoding
shader with U16/U8 payloads. Nonfinite operands fail before selection; Neg flips only
the sign bit, and ReLU maps nonpositive values (including -0) to +0. Exact subnormal
outputs obey the selected underflow policy without an invented inexact-tiny exception.
Canonical direct/grid maps and readonly scalar broadcasts require matching low-format
header/node policies. Reject and observable Clamp are admitted: a completed recovered
underflow publishes the exact payload before returning its recovered fault. Every fatal
lane outranks recoverable lanes and blocks host publication; fatal resident output is
Discarded. Portable unary remains rejected. F32/F64 Clamp/broadcast unary are separate gaps.
The generic direct F32/F64 unary endpoint now also rejects unsupported Portable headers
before structural admission/compilation; the older specialized F32 guard remains.

The independent M4 proof exhausts789,504 input encoding/op/underflow combinations
against both the core and a separate bit classifier. Native35/source37 cases cover
ordinary/prepared/device calls, odd extents, zero/NaN/subnormal faults, tails, retry,
resident/mixed owners, preflight preserve and fatal discard/read/borrow rejection.
`checked_low_unary` has genuine annotated source, ordinary, graph and native integer
checker peers at257/65,537 elements, all four formats/two ops/three policies:192
semantic peers pass. All retain copied current inputs, fresh output/status, required
completion and terminal prefix read/drop. No statistical timing/cost claim follows.
The192 warm caller-thread Rust census records report prepared/graph/native zero
allocation/reallocation/free; ordinary unary has one allocation+one free/56 bytes.
Native/driver/device heaps are excluded. Three fresh Metal buffer backings plus command
buffer/encoder acquisition remain in the matched host boundary; cold compilation is excluded.

This source cut uses a fresh independent Cargo target. An initial shared-cache run emitted
an older generic proc-macro diagnostic despite matching current source bytes; that failure
is preserved. A fresh build succeeds with current scalar `pcu::relu` lowering. Future
independent native cuts must avoid assuming a shared target proves artifact identity.

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
allocation does not promise initialization. Exact typed imports admit I32/U32/I64/U64,
F32/F64 and the four low floating encodings independently from arithmetic admission.
`MetalOwnedDispatchBackend::clone` retains the same native session and device identity without
opening or rebinding a device. Separately imported owners compose only when their retained
Metal sessions and identities match; independently opened sessions reject even on one physical
GPU. Thread confinement and quiescence remain required. Metal owned tensor graphs, MatMul and
training remain unsupported.

Memory allocations require positive exact byte sizes and alignment at most four. Payload bytes
are not rounded to word extents; status storage independently uses complete U32 words. They preserve session and pool affinity, observable backing ownership, and transfer access
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

The `checked_i32` canonical Criterion target pairs actual annotated Add/Sub/Mul with matching
explicit graph and native checker controls at 257 and 65,536 elements. Every route stages both
current inputs and owns fresh output/status, waits, checks all lanes and publishes a host prefix.
`-- --test` is correctness smoke rather than timing evidence. The activity guard rejects a busy
GPU before measured work. The owning `checked_i32` example demonstrates retained preparation,
ordinary RAM staging, signed limits, transactional host faults, retries and tail preservation.

The `checked_neg_f64` Criterion target pairs actual F64 Neg source, graph and native checker at
the same two extents. Encoding borders, complete bits, fault order, transactional host output,
retry and tails are checked outside samples. Every F64 value occupies eight payload bytes and
one four-byte status record. All F64 finite encodings, including exact subnormals, retain their
magnitude bits; RejectSubnormalResult remains explicit. See the checked source/device test for
resident eight-byte extents and tails. These are encoding unary claims, not double-ALU claims.

I64/U64 identity/Add/Sub/Mul use paired U32 limbs. Add/Sub retain carries/borrows;
multiplication derives exact 128-bit products from bounded sixteen-bit partial products and
checks the complete upper result before publication. Signed magnitudes and range limits use
unsigned limbs, including `i64::MIN`. This requires no native signed overflow or ulong division.
One diagnostic record corresponds to each logical eight-byte scalar. The `checked_i64` and
`checked_u64` Criterion targets each provide 18 matched source/graph/native checker peers.
Integer benchmarks share one generic cold graph/staging/oracle support module while each
workload retains its actual annotated concrete source. Earlier F32 timing tables remain tied
to their historical executable; this expanded shader is a new source identity.

The independent Clamp/broadcast cut is archived at
`metal-low-unary-clamp-20261001/README.md`: 1,579,008 actual GPU encoding/status cases,
77 native/source tests, the shared six-format binary gate and 384 matched semantic/census
peers passed on M4. Prepared/graph/native have zero caller Rust allocations; ordinary
has one allocation/free of56 bytes. Native heaps are excluded; these are semantic smokes,
not statistical timings or a full provider parity claim.


### Fourteen-width quotient/remainder qualification

The certified eight-width source/control/device/owned/ordinary joint quotient/remainder path remains distinct from new wide qualification. Private six-wide c884 sampled49,152 paired raw GPU quotient/remainder/status cases; a coverage audit found signedMIN/-1 was not explicitly paired. Test-only overlay56ba independently adds a full5x5 raw edge cross and exact signed-overflow lane0 versus later dividezero rollback for both host outputs. Its49,152 native pairs, prior8 source/mixed/device/owned/ordinary regressions and native strict lint pass without shader/runtime changes. These pair banks are sampled, not exhaustive full-width domains.

The public fourteen-width cut is independently closed on M4:1952-row SHA4c46e6578e233179a0a86deb25cb2ae74e9aee1c31bfe285cc9b79ef8eac35de, exactly10 Metal overlays on56ba with core/macros/facade unchanged. It lifts I/U128,256,512 into the existing retained packed U32 synthesis, using genuine generic source specialization. Five direct/Strict/grid, device/owned/mixed and ordinary native tests pass, alongside336 matched four-route semantic peers and336 separate ONE-call caller allocation/scorer scopes. Every scope reports zero Rust allocation/reallocation/free/requested bytes and zero warm scoring. The required root metal_fourteen_width_div_rem_contract subsequently passes on the same frozen source and existing binary. First/final GPU activity was1%/43% during correctness-only qualification; no idle or timing claim. Repeated/scalar-broadcast generic roles, Clamp recovery and Portable division remain outside this cut; native/device allocation remains unknown.


## Active requested-Portable unary candidate

`MetalPortableUnaryPlan::assess` now cold-assesses the neutral exact six-format
Neg/ReLU profile and retains the complete original numerical header. The separate
preparation branch preserves actual/permuted read/store roles and declared-unused
readonly metadata without staged resources or owner-affinity dependencies. Local
all-target strict lint and genuine source metadata pass; the requested-header
host/device/mixed/ordinary native fixture, common profile, example and four matched
`portable_unary` controls await their own M4 frozen proof. This candidate does not
retroactively widen earlier normal certificates or advertise a neutral discovery
offer. See `admission/unary/PORTABLE-CONTRACT.md` for integer-language, physical and
fault/publication obligations. Native/device allocation and latency are unknown.

## Active ordinary unary role normalization

The normal `Unspecified` unary cold path now consumes the neutral
`describe_checked_float_unary_map` matcher and preserves the complete original
header. Actual load/store roles survive declaration permutation; unread declarations
retain schema metadata without becoming staged resources or owner affinity.
The separately requested Portable path retains its request gate. Detached six-format
normal-role tests and strict lint pass locally; the new ordinary source, native
publication, example and `normal_unary_roles` four-route changing-bank census await
a separate immutable M4 qualification. Existing Portable and capacity certificates
are unchanged. Caller Rust counters do not establish SDK heap, JIT or device costs.
