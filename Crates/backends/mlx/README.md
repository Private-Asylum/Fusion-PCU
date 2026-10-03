# Fusion PCU MLX

MLX is an explicitly selected provider for Apple silicon macOS, with separate MLX-owned checked dispatch and delegated native tensor profiles,
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

The delegated native tensor contract is authentic captured/selected single MatMul source,
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

The safe prepared-source seam accepts ordinary sealed F32 slices and immutable opaque arrays
through `MlxProgramInput`, deriving matrix shapes from the frozen selected program. Complete
binding identities, scalar, extents, exact session affinity and quarantine checks precede host
staging. `MlxPreparedProgram::execute_mixed` retains terminal opaque output;
`execute_host_into` explicitly materializes its complete prefix transactionally. Two-element
binding/staging arrays add no Vec bookkeeping allocation. The facade's optional `mlx` feature
uses this seam for ordinary `#[pcu(flag(native_compound), flag(backend_precision))]` MatMul calls.
`PcuBackendChoice::Mlx` selects it explicitly. Automatic admits the exact bounded F32 MatMul
offer when MLX is the sole compiled GPU provider or only native Metal is also compiled; this is
a scoped Apple provider set, not general multi-provider ranking. An opaque retained input also
supplies its mandatory exact MLX session. Cold selection queries the authentic operation and
source numerical envelope before activation, retains the admitted implementation and discovery
root, and freezes host/resident input roles. Warm calls perform no discovery or migration. Host inputs stage
automatically, matching opaque inputs remain borrowed, and explicit `read_into` materializes
RAM. Static backing owns MLX arrays without forging generic device buffers or migrating
between providers. Consumed inputs remain alive through terminal work and outputs are fresh.
Generic invocation borrowing of these native F32 matrix owners rejects with a structured error; they are not encoded low-format dispatch owners. The owning example and facade
`mlx-matmul` example exercise these separate captured/ordinary routes.

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
registry identity, capacity, isolated budgets, workspace and cost. The bounded ordinary facade
route has its own source/affinity/consumption/escape tests and `mlx_source` Criterion peers;
it does not broaden this numerical/operator envelope.

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

The facade benchmark can be correctness-smoked with
`cargo bench -p fusion-pcu --features mlx-benchmark-control --bench mlx_source -- --test`.
It compares the actual ordinary call, explicit captured source, neutral graph and retained
native control at 16x16 and 128x128. Every peer copies current host inputs, terminally executes,
materializes the complete result and drops fresh backing. All banks are initialized before
registration so filtered Criterion runs do not depend on another route. Full value/tail oracles
run outside samples and AGX activity is checked. This is a separate boundary from the canonical
backend-local reused-input controls; no historical timing is assigned to the new facade route.

Ordinary offer costs distinguish all-host input staging with resident output, mixed inputs with
resident output, and entirely resident execution. Required completion and escaped owner
publication belong to each operation boundary; later explicit readback/drop belongs only to the
matched benchmark wall. Every cost and workspace estimate remains unknown (`None`). Retaining
the MLX source root preserves exact session affinity after cache eviction. Core memory headroom
and isolated-budget enforcement remain unplumbed for opaque MLX allocation.


## MLX-owned checked dispatch

A separate fixed C extension retains a real `mlx::core::Primitive` created by the pinned `fast::metal_kernel` API. MLX allocates its exact physical UInt8/UInt16/UInt32 input and payload arrays, UInt32 sibling status array, compiles its integer-only kernel and schedules it on the retained explicit MLX GPU stream. This path does not activate a PCU MetalSession or use CPU arithmetic. The logical PCU dtype is independently frozen as F16/BF16/E4M3FN/E5M2/F32/F64; F64 uses two physical UInt32 limbs. Its UInt carrier is never advertised as the logical type.

`MlxCheckedUnaryPlan::assess` is detached cold admission for precisely one checked Neg/ReLU operation, Reject/Clamp, all three underflow policies, Boundary/Strict, and direct/canonical grid/scalar-readonly broadcast. Actual SSA, binding roles, extents and the complete numerical header are verified before runtime preparation. A later active candidate admits the neutral Portable unary descriptor with the original complete header and separate implementation identities; requested-header native qualification is pending. Conversions, integer arithmetic and general dispatch graphs remain outside this unary assessor. The six-format unary source admission is closed in immutable1743 (SHA1189a76c…), with576 genuine four-route semantic peers and576 zero caller Rust allocation census rows; native/device allocations remain unknown. The independently emitted F32/F64 bit/status proof is separately closed in1740. Binary arithmetic has its separate exact assessor described below. Exact offers cover Host only, with unknown cost/workspace and unknown physical registry/capacity facts.

`PcuHostKernelBackend` preparation on an explicit `MlxSession` retains the admitted native primitive and warms its pipeline with zero-valued operands. Warm calls stage plain RAM into owned arrays; unaligned UInt16 bytes first receive an aligned C++ temporary before MLX's typed copy. Caller storage never escapes. Fresh sibling payload/status owners, their real input, stream and image survive both evals, explicit-stream synchronization and both waits. Complete status scanning chooses the lowest fatal lane before any recovered lane. Fatal/schema/native errors preserve host output; successful or fully recovered calls publish the complete logical prefix and preserve tails. Recovered Clamp returns its structured fault after publication. The completed payload stays retained until its checked release before the next call or prepared-owner drop. Unknown completion quarantines actual owners and poisons the session.

The facade's ordinary host route selects this provider explicitly with `PcuBackendChoice::Mlx`; its frozen preparation/cache carries the same real MLX session and implementation. Encoded resident dispatch owners and mixed host/resident calls retain the same actual MLX session and immutable private publication law. Cross-provider imports remain unsupported. Native matrix owners remain the separately admitted tensor representation. Neither this extension nor the matrix runtime plumbs the core's isolated memory-headroom policy.

The independent full encoding oracle and the ordinary source/publication fixture are native hardware gates. `checked_unary` supplies genuine source/graph/native Criterion peers and a separate Rust allocation census; the numerical and ownership envelope is independent of the historical MatMul benchmarks. Exact native source identities and deployment evidence are recorded in the external MLX plan, with missing external injection fixtures stated separately.


### Six-format checked binary admission

The bounded binary assessor now accepts F16/BF16/E4M3FN/E5M2/F32/F64 Add/Sub/Mul/Div, exact instruction/header range and underflow agreement, Boundary/Strict and independent scalar permission tuples. Direct/grid/broadcast, one/two actually loaded unique bindings, repeated and reversed SSA operands are frozen cold. F32/F64 binary receives a distinct implementation revision. Integer arithmetic, conversion and Portable remain rejected. F64 payloads are physical UInt32 limb pairs with logical byte/count checks; no MLX native double dtype or floating ALU is implied. Independently emitted F32/F64 payload/status proof is closed in immutable1722. Six-format prepared/source/discovery and common binary/operand gates are closed in immutable1739 (SHA4bbab3da…), with864 matched three-route host-publication semantic peers and864 separate zero caller-Rust-allocation census rows. SDK/native/device allocations remain unknown. The six-format resident/mixed/prefix publication fixture is separately closed in1743, including local policy, recovered Clamp, fatal rollback and retained sibling/drop/cache lifetimes; earlier resident fixtures qualified low4 only. No whole-provider parity claim follows.

### Bounded owned ReLU graphs

`MlxCheckedTensorPlan::assess_program` admits all six checked float formats for an authentic Input plus at most one checked ReLU effect. Immutable1760 SHA2a2b883f closes actual graph and ordinary source ownership/policy/lifetime fixtures,144 matched captured-source/graph/native/ordinary semantic peers and144 rows of64 changing caller calls. All four routes measure2 caller Rust allocations/frees256B per call,0 reallocations and0 warm scorer callbacks; native/device/driver allocations remain unknown. The original example independently expected -0 incorrectly after ReLU; its preserved failure was repaired only to the canonical +0 expected by the scalar contract, with explicit example-only source SHA39ad0d61 and native lint/execution. No arithmetic or resource repair is hidden in that delta. Graph Clamp, Portable, ReLUBackward and training remain unadmitted.

### Bounded owned binary graphs

`MlxCheckedTensorBinaryPlan::assess_program` and `MlxSession::prepare_tensor_binary_program` now freeze an authentic selected six-float graph containing one/two actually selected Input leaves and one checked Add/Sub/Mul/Div. An unused checked effect remains observable before returning a selected Input. Input metadata does not invent arithmetic options; the actual effect's options/underflow must match the frozen request. Range Reject and Unspecified reproducibility only; Clamp graphs and Portable are rejected. Extra effects, unproved layouts/dtypes and malformed selected closures reject before native work.

The retained `MlxPreparedTensorBinaryProgram::execute_mixed` preflights every exact ValueId/type/span/actual session before any host upload, stages each actual unique input once, and completes the same independently proved MLX checker into a fresh immutable encoded owner. Old owners remain readable after fatal errors. Unknown work poisons the actual retained session/pending resources. Selected Input after checked work shares the immutable input only after the checked effect completes; there is no fabricated extra CarrierCopy.

`MlxTensorBinaryRequest` has a separate implementation family and unknown cost/workspace. Offers cover HostInputsResidentOutput and Resident, with MixedInputsResidentOutput only for two actual inputs. Host readback costs are not inferred from this escaped-owner API. Backend-only1751 (SHAfd079a3f…) is independently closed. Ordinary facade integration1753 (SHA897f3d25…) closes authentic six-format owned roles,144 dtype-policy cohorts, foreign actual session rejection, checked-unused faults and escaped-owner lifetime. Separate1755 (SHA5a95688e…) supplies576 four-route semantic peers and576 rows of64 changing caller calls, including genuine ordinary `#[pcu]`, with3 allocations/3 frees384B per call,0 reallocations and0 warm scorer callbacks. Native allocations and latency remain unknown. Existing native F32 MatMul and encoded Identity/ReLU routes remain separately admitted.

### Current contract gaps

These are distinct provider contracts. A transported encoding or upstream MLX dtype is not arithmetic admission. Native/device allocation counts and memory-headroom enforcement remain unqualified.

| Contract | Qualified scope | Remaining work |
| --- | --- | --- |
| Scalar carrier maps | All22 byte-aligned types; direct/grid/scalar broadcast, exact bits | Longer resident readonly-input prefixes |
| Checked scalar binary | All6 checked float formats, Add/Sub/Mul/Div; Boundary/Strict,3 underflow policies, Reject/observableClamp, actual unique/repeated/ordered operands | Portable profile; conversions |
| Checked scalar unary | All6 checked float formats, Neg/ReLU; same numerical tuple and publication laws, immutable1743 | Portable unary candidate awaits independent requested-header native/source/census proof |
| Integer scalar arithmetic | All14 Add/Sub/Mul, Reject/observable Clamp; ordinary scalar, offers, actual roles and672 matched four-route peers qualified in1876 | Portable integer profile; general multi-effect owned graphs |
| Integer DivRem | Fourteen-width distinct-input source/control/ordinary joint publication, exact offers/session aggregate and required shared14 gate closed in1956/ff3a | Repeated/scalar-broadcast generic role adapter locally strict-clean; independent native proof pending |
| Owned carrier graphs | All22 exact Input leaves, immutable escaped owners | General selected graph transport/topology |
| Owned checked unary graphs | All6 checked formats, used/unused ReLU effects; Reject only;1760 source and separately declared example-only repair | Central graph Clamp representation; general unary graph closure |
| Owned checked binary graphs | Six float formats and14 integer widths; used/unused checked effects, Reject; separate ordinary/four-route certificates1755 and1919 | Central graph Clamp representation; general multi-effect graphs |
| Native MatMul | Dense rank-two F32, no transpose; Boundary with explicit native-compound/backend-precision permissions | Checked/Strict integer-synthesized MatMul, F64, transpose, general graph execution |
| Native/encoded F32 composition | Cached official bit views on the same actual session, qualified chains and descriptor/API census | No arbitrary dtype import or universal physical zero-copy claim |
| Training/autodiff | No provider-qualified MSE/ReLUBackward/SGD spine | Independently checked implementations and actual source/ownership/common gates |

The integer primitive has a distinct C handle and exception boundary, retains an actual MLX GPU primitive, and uses UInt8/UInt16/UInt32 physical carriers with up to16 exact limbs. It does not overload floating format IDs, invoke MetalSession or substitute CPU arithmetic. Immutable1757 SHA400ac68f independently closes the raw primitive after one constant-address-space broadcast repair; the original failed shader cut is retained. Immutable1838 SHA bb926b51 closes public direct-control and separately prepared typed source qualification: actual14-width host/resident/mixed tests,504 matched source/independent-IR/native semantic peers and504 separate64-changing-call caller rows all pass. All32,256 measured calls report zero caller Rust allocations/reallocations/frees/bytes; native/device allocations are unknown. The source adapter uses `assess_checked_integer_binary_operands` to retain one/two actually read bindings, actual SSA operand/index roles and byte extents cold; each unique host input is uploaded once, including repeated operands. Checked outputs remain distinct from preexisting immutable owners. Reject and Clamp statuses are sibling outputs from the same actual retained primitive; unknown terminal work retains and quarantines the real resources/session.

The exact scalar capability/offer and static session Integer lift is now independently qualified by immutable1876: original SHA2b75690e… and one test-fixture-only repair SHA9615846f…. Actual ordinary fourteen-width Reject/Clamp/operand-role gates pass, together with672 matched four-route semantic peers and43,008 changing caller calls reporting zero Rust allocation and zero warm invocation-score callbacks. The original empty-owner fixture failure and subsequent test-lock poisoning are preserved; the repair creates a lawful undersized foreign owner without changing production behavior.

The selected owned-integer graph executor independently admits one checked Add/Sub/Mul effect over one/two actual Input leaves, with optional selected Input only after the unused checked effect completes. Actual14-width owner/lifetime tests and252 three-route semantic peers pass in1876;16,128 separate changing caller calls each report3 Rust allocations384B at the escaping-owner/read/drop boundary. Native/device allocations remain unknown. Owned graph Clamp and Portable remain rejected. Exact `MlxTensorIntegerRequest` offers and ordinary facade owner routing are now qualified in immutable1919 SHA1545c4b4. Actual fourteen-width ordinary roles, both mixed directions, consumed/borrowed owners, selected checked-unused faults and true foreign-session negatives pass; unused EMPTY host declarations also succeed through the ordinary route. The required fourteen-width, twenty-two-carrier and four-low-format owner gates pass in the same cut. Its336 four-route semantic peers and21,504 changing caller census calls include genuine ordinary `#[pcu]`; all routes retain3 Rust allocations384B/3 frees per escaped-owner/read/drop call,0 realloc and0 warm scorer callbacks. Native allocations and latency remain unknown. A private eight-width DivRem prototype remains distinct from the public candidate. No conversion, wrapping or checked native-library integer promise follows from these bounded implementations.


### Public quotient/remainder and separately pending ordinary integration

`MlxCheckedDivRemPlan` is a detached exact assessor for all14 signed/unsigned widths8/16/32/64/128/256/512, four homogeneous bindings, two distinct loaded operands and two independent stores, direct/canonical grid, Reject and Unspecified reproducibility. It verifies the shared integer DivRem schema before activation. Repeated self-division and scalar-broadcast dispatch schemas remain outside this initial public profile; direct controls retain explicit native broadcast roles independently. This describes the certified eight-width baseline. The separately captured fourteen-width public candidate below is not covered by that certificate.

`MlxSession::checked_div_rem_backend()` prepares a separate `MlxPreparedDivRemHostKernel`. Its metadata names both input and output bindings, byte lengths and actual arity; there is no single-output getter. Host calls preflight all four arguments, read both completed private results into retained cold scratch, and complete fallible temporary cleanup before writing either caller prefix. Mixed/resident execution returns two completed fresh immutable owners from the same retained MLX primitive. Existing owners are unchanged on fatal work; unknown completion retains actual pending resources and poisons the real session. `MlxEncodedArray::release` consumes a completed handle: unique uncached holders receive checked official release, while shared Rust handles leave the retained native holder live. Unique cached native F32 views reject this bounded cleanup seam. Public DivRem staging performs checked temporary cleanup before returning its completed sibling pair; unknown sessions keep their real quarantined resources.

Immutable1928 SHA db3bfae1, an explicit22-file MLX-only overlay on1919, closes actual M4 public source/graph/control and joint-owner qualification without repair. Both normal/census all-target strict gates, actual public source fixture, prior private joint regression,48 three-route semantic peers and48 separate64-changing-call caller rows pass. All3,072 measured calls report zero Rust allocation/reallocation/free/bytes; native/device allocations remain unknown. Earlier1928 SHA41fc6a18 was unexecuted and superseded before any native proof. No ordinary fourth route or warm scorer census belongs to db3.

The later exact eight-width coarse caps/Host offers and static Session aggregate retain explicit `MlxDispatchOutputLayout::Single|DivRem` and `MlxDispatchCompletion::Single|DivRem`; no quotient-only getter hides a second output. Immutable1952 SHA84375edd is closed:61 actual native facade library tests, required ordinary common8 gate, backend source/offer2, prior owner/invocation routes and both examples pass. Its64 four-route semantic peers and4,096 separate changing caller calls report0 caller Rust allocation/reallocation/free and0 warm scorer callbacks. Native/device heap and latency are unknown. The independent private six-wide prototype1955/aff70d24 is also closed:73,728 paired native quotient/remainder/status comparisons with full signedMIN/-1 edges and scalar broadcast roles, plus prior8 regressions. Neither proof certifies public wide preparation by itself.

The public fourteen-width cut is independently closed on M4:1956-row SHAff3a6ebf40486a1bd96a3c51821921bfffaa516f6e61d34246800989e6137657, exactly9 backend overlays on aff70, excluding later shared role and identity-policy changes. It admits the same bounded distinct-indexed-input profile for all14 types and appends six exact implementation IDs while preserving old eight IDs/revision. Three native source/owner/offer/aggregate/ordinary tests,112 matched four-route semantic peers and112 separate64-call Rust/scorer census scopes pass. The7,168 changing calls each verify outputs/tails outside capture and report zero Rust allocation/reallocation/free/requested bytes and zero warm scoring. The required root mlx_fourteen_width_div_rem_contract subsequently passes on the unchanged source and existing binary. Observed first/final GPU activity was0%; no timing or native/device allocation claim. Clamp, Portable and broader generic source roles remain outside that certificate. The additive MlxCheckedDivRemRolePlan/Backend/Prepared API currently passes local type/strict/pure role checks, but awaits its own real M4 source/ownership proof and exact-offer opt-in.


## Active Portable unary candidate (not a completed native certificate)

`host_kernel/unary/PORTABLE-CONTRACT.md` audits the six floating formats' exact
unsigned encoding implementation. The cold neutral unary descriptor and original
requirements select distinct IDs `0x1700..0x172f`, revision
`0x0003_0020_0003_1400`; old normal IDs and scope are unchanged. Declared-unused
readonly bindings retain metadata and stage no input. Requested Portable native
fixtures cover full input capacity/lifetime, reordered/unused host declarations
and full low-format encoding banks. The `portable_unary` example and Criterion
control carry actual deterministic `#[pcu]` source, with source/IR/native/ordinary
matched retained-input to fresh-private-output/read/drop boundaries. All-target
local strict lint and three pure tests pass; actual M4 Portable conformance remains
pending. Earlier normal-policy proofs are not retroactively relabeled.

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
