# Fusion PCU Vulkan

Vulkan execution is explicit and default-off. `hosted` enables discovery and synchronous scalar invocation; `tensor` also enables the bounded retained tensor plans. No CPU fallback executes a Vulkan workload. Software Vulkan devices are excluded from native qualification.

The qualified profiles are recorded below. A scalar carrier or coarse device capability does not admit every operation on that type.

| Boundary | Qualified concrete profile |
| --- | --- |
| Scalar transport | All 22 sealed carriers, dense identity and scalar broadcast, arbitrary raw bits |
| Checked integer map | All 14 signed/unsigned widths 8–512, Add/Sub/Mul, Reject or observable Clamp; actual repeated/unused/reversed read roles |
| Checked joint division | All 14 integer widths, canonical and actual three/four-declaration read roles, direct/grid and readonly scalar operands, Reject; zero and signed MIN/-1 faults preserve both outputs |
| Checked float map | F16/BF16/E4M3FN/E5M2/F32/F64 Add/Sub/Mul/Div and Neg/ReLU, all three underflow policies, Reject or observable Clamp |
| Checked conversion | F32/F64 widening/narrowing, exact finite/rounding/underflow law, Reject or observable Clamp |
| Portable scalar map | Four low float formats: one Reject binary operation; all 14 integers: one Add/Sub/Mul operation with Reject or Clamp and the exact operand descriptor |
| Owned leaf | All 22 carriers: selected Input/Constant/Uniform transport; genuine ordinary source qualification covers Input identity |
| Owned pointwise | All 14 integers and six checked floats: bounded checked Binary; six-format ReLU and ReLUBackward; Reject/Unspecified |
| Owned compound | Ordered Strict F32/F64 MatMul/MSE/SGD with four compound/precision permission tuples and all three underflow policies; Reject/Unspecified |
| Requested gradient | Bounded Strict F32/F64 target-selective MSE training, with checked forward effects and complete source/graph/native controls |
| Borrowed resident invocation | Same logical-device owners and host/native mixtures with exact prefix publication, retained roots, tails and preflight/fatal rollback |

Joint division operand roles have a separate native certificate; their mixed/resident routes remain unqualified. Portable joint division has descriptor-only cold opt-in and is awaiting its independent native qualification. Graph Clamp, general Portable tensors/compounds, asynchronous execution, cross-thread ownership transfer, external memory and graphics/ray execution are not implied by these profiles.

All current checked scalar arithmetic and transport modules use U32 storage/bit synthesis. They do not depend on native Int64/Float64 arithmetic or infer IEEE preservation from Vulkan float controls. The separate legacy public F64 bit-map API keeps its own `shaderFloat64` feature gate. Integer precision/compound permissions and scalar Boundary/Strict requests retain the stronger exact checked implementation. Strict compounds perform the specified scalar round/check sequence and preserve node, output-lane and reduction/step fault provenance; a finite final result cannot erase an intermediate checked fault.

Cold preparation validates the numerical header, typed SSA, actual operand roles, original declarations, logical extents and native device limits. It freezes independent indexed/element-zero operands and their actual spans. Schema-proved unread declarations retain type/access checks but require zero elements; repeated reads share one physical input. Warm invocation performs no discovery, ranking, shader compilation or graph evaluation. Numerical requirements remain part of the exact request and cache key.

Host calls stage fresh inputs into retained private storage, submit retained commands and wait for terminal completion. Fatal status is scanned before either public result prefix is written. Reject faults and argument failures preserve the complete caller outputs. A completed scalar Clamp map with recovered range faults publishes all useful values and returns the earliest logical recovered fault; any fatal lane takes precedence and prevents publication. Tails remain untouched.

`PcuVulkanOwnedBuffer<T>` retains its originating logical-device root and real native allocation. Facade `PcuTensor` owners retain that backend and cached shape across borrowing, cache clear and sibling drops. Borrowed invocation uses a dedicated native byte argument, never a forged host pointer or a recreated session. Private output computation precedes public copies; Vulkan byte copies use the exact positive logical prefix, including odd U8/FP8/F16 lengths. Unknown completion quarantines the full allocation/code/command/queue/device roots. A poisoned shared device makes borrowed owners unusable even if uncertainty arose before public writes. Current facade owners are thread-local and carry no cross-thread transfer guarantee.

The runtime negotiates the loader's supported core API up to Vulkan 1.4 and bounds device queries by the physical device version. Core 1.3 synchronization2 is queried and explicitly enabled before using `vkQueueSubmit2`; unsupported sessions use legacy submission with the same fence and quarantine contract. The stable ash 0.38 bindings describe Vulkan 1.3.281; negotiation of 1.4 does not advertise unimplemented 1.4 commands or features. Descriptor indexing and buffer-device-address capability queries recognize their 1.2 promotion, and optional extension structures are queried only when supported.

Current bounded compute plans retain fixed storage-buffer descriptor layouts, one retained set/pool and stack-written updates after prior completion. Descriptor indexing, update-after-bind, descriptor buffers and graphics layouts need explicit future feature and lifetime contracts. Dynamic rendering will matter when graphics/shader execution is added; compute parity does not require a render pass. Current per-buffer allocation respects queried native requirements and declares required/preferred dedicated allocations on core 1.1+ devices. Compatible host-visible coherent memory remains required; noncoherent memory needs coordinated flush/invalidate boundaries for uploads and full status scans before it can be admitted safely.

`memory_realizations()` reports actual buffer allocation bytes, memory types and property flags. Compatible host-visible coherent/cached memory is preferred; this does not imply zero-copy caller RAM or external interoperability. Repeated joint reads use four physical buffers (input, quotient, remainder, status) behind five descriptors; distinct reads use five. Cold and escaping-owner allocations are separate from warm caller allocation counts.

Hardware fixtures stay ignored on machines without the required physical device and tools. Genuine `#[pcu]` prepared/ordinary routes are paired with explicit graph diagnostics and separate GLSL/compiler/ash ownership controls. Controls share audited shader arithmetic where stated; independent full-bit integer oracles validate values and status. Constant/Uniform backend diagnostics do not establish ordinary Uniform/Constant source support. Matched ownership/publication boundaries and allocation/API census are recorded per workload; no universal zero-allocation escaping-owner or SDK-allocation claim follows from a scalar caller census.

Examples of bounded qualification targets:

```sh
cargo test -p fusion-pcu-vulkan --features tensor --test wide_div_rem -- --ignored --test-threads=1
cargo test -p fusion-pcu-vulkan --features tensor --test div_rem_roles -- --ignored --test-threads=1
cargo bench -p fusion-pcu-vulkan --features tensor --bench div_rem_roles -- --test
cargo run -p fusion-pcu-vulkan --features tensor --example div-rem-roles
```

Criterion `--test` provides semantic evidence without latency estimates. Statistical timing requires the unchanged GPU activity guard and exclusive task ownership; correctness under recorded user activity is not an idle performance result. Modern Khronos synchronization validation, exact frozen source/binary hashes, native device/driver information, independent oracles and measured source/graph/native census are retained in the assigned SPIR-V/Vulkan plan and text-only evidence archives. Earlier certificates remain tied to their original source cuts.

Offline shader maintenance scripts live beside each SPIR-V profile. Crate execution consumes checked-in words and needs no shader compiler. Facade-dependent development targets use repository-only path dependencies; a standalone published backend archive is not claimed to include that complete harness.


Runtime shader artifacts are configured on `PcuVulkanBackend` with
`PcuVulkanShaderCachePolicy::{Disabled, MemoryOnly, Disk}`. Disk configuration chooses a directory,
RAM retention and explicit invalid-module rebuilding. `beside_executable()` selects `pcu-shaders`
next to the executable. Typed host preparation persists a standard little-endian `shader.spv`
and separate complete request metadata by publishing a temporary directory atomically. Disk IO
and identity failures remain explicit. Direct leaf and tensor preparation currently refuse disk
mode because they do not carry the complete typed request context.

This first artifact format is producer and host specific. Its revision-bound typed Hash recording
is not a portable canonical IR serialization. Every hit still lowers the request and verifies
its exact expected words, metadata and bounded SPIR-V structure; it proves validated artifact
reuse, not a compiler-work bypass or arbitrary shader-import semantics. Filename digests are
index hints, and exact words also validate RAM hits. The cache retains at most 256 entries and
32 MiB aggregate key/module bytes. The fallible request-key API refuses nested/cyclic regions,
more than 4,096 total visited operations, oversized declaration tables or complete metadata
above 1 MiB. Disk preparation reports these failures explicitly; MemoryOnly may skip artifact
retention when only the metadata budget is exceeded. Disabled caching skips key recording.
Prepared calls perform no artifact IO, hashing or lookups.
Changing policy or deleting persisted artifacts does not retire existing native pipeline owners.

The `shader-artifacts` example uses genuine `#[pcu]` source, writes/reloads its module and executes
both retained pipelines after deleting the artifact. Persistence alone does not reduce embedded emitter/template binary size. The separately
configured external composed source below excludes the qualified family's embedded word arrays.
Other shader families retain their current source until separately qualified. Vulkan driver-specific
`VkPipelineCache` persistence is also separate from portable SPIR-V words and is not enabled here.


External composed templates are selected at runtime with
`PcuVulkanShaderSource::ExternalComposed { directory, retain_in_memory }` independently of artifact
retention. The `beside_executable()` source helper selects `pcu-shader-packages`. Source packages
contain standard `template.spv` plus `rewrite.bin` with the exact function IDs, call offsets,
argument IDs and specialization offsets. The no_std compiler validates a known SHA256 over a
specified little-endian, length-framed codec and producer/family revision before rewriting.
Mutated words, metadata or another shader family refuse; structural validity alone never admits
an arbitrary imported shader. Package IO and validation stay in cold preparation. Loaded package
RAM can be cleared without retiring existing native pipelines.

The default `embedded-composed` feature retains the built-in GLSL-derived composed assets.
`--no-default-features --features hosted` excludes those composed WORDS arrays and requires a
runtime external source for composition. Other template families remain embedded. The facade's
weak default `embedded-vulkan-composed` feature forwards embedding without enabling a backend;
facade no-default builds can explicitly add it when desired.

Export a deployment package with `cargo run -p fusion-pcu-vulkan --features hosted
--example export-shader-packages -- /path/to/pcu-shader-packages`. Consume it with
`cargo run -p fusion-pcu-vulkan --no-default-features --features hosted
--example shader-packages -- /path/to/pcu-shader-packages`. The latter uses genuine annotated
source and explicitly configured prepared Vulkan execution. Cache compiler-work bypass and
hardware-specific VkPipelineCache persistence remain separate future gates.


Ordinary facade `#[pcu]` calls configure these same runtime settings through
`global::vulkan::configure(PcuVulkanShaderOptions { source, cache })`. The facade's
`vulkan-shader-cache` example exercises ordinary disk persistence; new sessions use the updated
settings while existing resident owners retain their originating configuration. This global
integration has its separately owned source/native proof; it does not widen disk admission to
unqualified direct leaf or tensor factories.

Bounded integer composition now admits all fourteen homogeneous signed/unsigned integer
widths 8–512 for checked Add/Sub/Mul, with four declarations/resources and at most 64 static
steps, direct or canonical grid. Fixed UInt32 arithmetic uses four packed lanes for 8-bit,
two for 16-bit, and limbs for wider types. Each packed word has a single writer and each
logical lane retains its own status and ordered private stores/reloads. Header and operation
Reject/Clamp must match; publication remains transactional. Narrow byte addressing is bounded
to u32. Cross-index read/write, wrapping, division, mixed types, integer literal producers and
arbitrary control remain outside this composed profile. Primitive integer maps retain their
existing executor. Discarded one-effect checked Add is separately admitted for widths up to
128 bits, preserving its observable fault.

Normal ten-width source qualification closes the remaining 8–128-bit widths in embedded and
external-only builds, with 410 independent fixed goldens, 480 numerical/range cohorts and
19,680 staged plus 19,680 discarded-effect requests per build. The original four-wide normal
and requested Portable certificates remain separate. The remaining ten widths also pass their own explicitly requested Portable native matrix
in embedded and external-only builds, through the same neutral descriptor. Strict and determinism remain independent and shader arithmetic does not change
with the requested reproducibility header.

External composition has five certified compiler-built package families: three float families,
`composed-integer-wide` and `composed-integer-narrow`. Their fixed rewrite metadata and known
SHA256 identities certify those exact compiler assets, rather than arbitrary imported shaders.
Previous family identities remain unchanged.

`wide_composed`, `narrow_composed` and the corresponding `portable_*` source fixtures exercise
staged Add→store/reload→Mul→Sub with faults, rollback, Clamp notices, complete prefix bits,
tails and retry. `composed_integer_maps` compares genuine two-effect source, generated prepared
IR, independent hand-authored IR and direct native GLSL compilation. Its original four-wide
native control and census certificate is closed. The all-fourteen-width two-effect and
ten-width discarded-effect extension also passes 9,216 route comparisons, 216 fault edges,
and its separate zero-allocation/rescoring warm census. All fourteen widths pass direct/grid
WriteOnly stage checks; narrow external-package owner lifetime is qualified separately. `PCU_COMPOSED_CENSUS=1` selects a separate 64-call
changing-payload Rust heap/rescoring census, which does not claim driver-internal allocations.
Correctness captures on a busy device do not establish latency or performance parity.

The later independent arithmetic cut replaces the integer SDK control's shared shader
helpers with handwritten unsigned 32-bit limb arithmetic and signed bounds. It passes
10,752 route keys and 252 edge witnesses for all fourteen widths, N65, two-effect
`(input + input) * input` and discarded Add, normal/requested Portable headers,
Reject/Clamp and all inert underflow flags. Its separate 64-changing-call census has
zero Rust heap activity or cold rescoring. Official shader validation covers 336
variants; mechanical host adaptation covers 486 staged goldens. Staged/scalar-seed/Sub
and N4096 independent native controls remain separate gates. Older helper-sharing
certificates keep their original provenance; production arithmetic is unchanged.

The optional `insights` feature exposes `PcuVulkanApiInsights`, a caller-owned fixed
count-only ledger. `PcuVulkanBackend::with_api_insights` attaches before loader/device
construction; `with_api_insights_scope` attaches only to newly constructed owners on
that thread. Scope nesting and unwinding restore the previous attachment. Existing
owners keep their originating ledger, including owners retained separately from the
facade invocation cache. Clear that cache before attempting to attach new ordinary
calls; clearing it does not replace separately retained device owners.

Counts represent attempted provider-visible Vulkan SDK calls, including distinct
buffer/memory creation, map/unmap, module/pipeline/descriptor/command construction,
submission and wait boundaries. They do not describe driver-internal allocations.
No clock is sampled implicitly. Warm probes borrow the retained owner ledger, and
compilation without `insights` removes the fields, TLS attachment and probe sites.
Native API census qualification is separate from the existing arithmetic and Rust
heap certificates; the `api_insights` test checks source and independently authored
IR, alternate attachment isolation, stable warm creation/mapping counts and terminal
owner balance.

The separate native API ledger certificate passes 128 changing source/hand-authored-IR
calls: exactly 128 submissions, waits and fence resets, no warm object creation or
mapping, alternate attachment isolation, balanced terminal SDK owners and no retained
ledger leak. Its counter-disabled optimized LLVM contains no ledger/TLS/probe symbols.
This qualifies the explicit F32 prepared source/IR owner slice; ordinary global and
tensor/mixed ledger lifecycle remain separate gates.

The later integer benchmark SDK cut passes 10,752 separate 64-changing-call
source/prepared/hand-IR/handwritten-native keys: only fence reset, submission and wait
occur warm, with zero Rust heap. All 168 handwritten native owners and 504
buffer/allocation/map lifetimes have balanced cold/terminal API counterparts. This
includes ordinary source warm calls, but provider thread-cache destruction remains
outside the complete terminal proof. `PCU_COMPOSED_API_CENSUS=1` explicitly attaches
the two ledgers in an `insights` build. Counter-disabled optimized benchmark LLVM
contains no census/TLS/probe symbols. No driver-internal heap or latency claim follows.

The separate four-wide discarded-Add certificate closes U/I256/512 one-effect
composition with normal and actually requested Portable headers: embedded/external
scalar-seed direct/grid source goldens,1,536 paired route keys,36 native edges and
1,536 zero-heap/rescoring warm census keys. Together with the ten narrower widths,
all fourteen carriers have qualified discarded checked Add within those envelopes.
Wide one-effect Sub/Mul remain refused by the composed provider; existing primitive
and multi-effect profiles retain their separate contracts. The static wide template
and external package identity remain unchanged.
