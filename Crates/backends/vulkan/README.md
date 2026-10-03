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
