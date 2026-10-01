# Fusion PCU Vulkan

Enable `hosted` for explicit Vulkan discovery and synchronous execution. Default
features load no Vulkan runtime. Per-function source uses
`negate_prepare(&backend)` through the neutral prepared host interface.

The admitted source slice is exact F32/F64 transport and checked F32/F64 sign-bit
Neg. F64 is admitted only on a device reporting `shaderFloat64`; activation enables
that feature. F64 loads/stores use Float64 with two uint32 bitcast words, without
Int64 capability or floating arithmetic. Cold discovery and exact offers carry
this device-dependent type support.
Neg rejects nonfinite operands, preserves signed zero and subnormal bits, and
supports the three explicit underflow policies. Clamp and portable numerical
profiles are not admitted. Boundary and Strict have the same scalar contract.
The implementation uses integer bit operations and per-element diagnostic status.
No shader floating-point arithmetic or host CPU execution of the workload occurs.

Preparation validates the exact IR schema and device limits and retains pipeline,
three owned coherent mappings, command buffer and fence. Every call stages fresh
host input, waits for terminal completion, checks the lowest logical fault, and
publishes only the initialized output prefix on success. A checked fault leaves
caller output unchanged. Unknown completion quarantines all native owners and
the retained session; a poisoned session rejects further calls. Borrowed caller
RAM is never an asynchronous device pointer.

The host path prefers a reported `HOST_CACHED | HOST_COHERENT | HOST_VISIBLE`
memory type when compatible; it falls back to coherent host-visible memory.
`memory_realizations()` reports the actual selected native memory type, property
flags and allocation bytes for input, output and status. Cached host storage
does not imply shared caller allocations, unified RAM or external interop.

`PcuVulkanDiscovery` implements bounded generation-scoped discovery. Activation
validates references and rechecks the physical Vulkan device UUID and native
vendor/device IDs. CPU software Vulkan devices are excluded. Stable device
identity does not establish CUDA/HIP/Vulkan import compatibility. Discovery
reports optional physical facts and query-only capabilities; descriptor indexing
and device-address bits are not enabled execution offers.

Cold `PcuImplementationOffers` describes exact Copy/Neg host requests for a
discovered backend. Offers carry the discovered device/executor identity, exact
numerical envelope and unknown native workspace/performance cost. Logical status
uses four bytes per element; prepared allocation facts report actual padding. Resident,
native library, PortableV1, graphics/ray and external memory execution remain
unsupported. The warm call contains no discovery, compilation, ranking or
allocation of pipeline/storage/submission objects.

Repository targets:

```sh
cargo run -p fusion-pcu-vulkan --features hosted --example checked-neg
cargo test -p fusion-pcu-vulkan --features hosted --test prepared_bit_map -- --ignored --test-threads=1
cargo bench -p fusion-pcu-vulkan --features hosted --bench checked_neg
cargo test -p fusion-pcu-spirv --test bit_map -- --ignored
```

Hardware tests are ignored by default. Criterion has actual per-function source,
explicit graph, ordinary global calls and a matched independent native ash/GLSL
control at both scalar widths. The control owns a separate logical device and
builds its shader with `glslangValidator`, then performs actual Vulkan calls; it
does not call PCU lowering or execution. UUID matching ensures the same physical
device. All peers use retained coherent/cached buffers, identical bit checking,
dense status, terminal fence wait and transactional host publication. Every
invocation changes its input and checks all output and padded tails outside timing.
`PCU_VULKAN_CENSUS=1` reports warm Rust heap allocation counts, full call wall
time and independently sampled submission/completion/upload/status-publication
stages. `call_profiled` provides optional graph stage measurements; ordinary
calls do not read a measurement clock. Driver-internal C allocation counts are
not measured. Each benchmark has a GPU idle guard. It measures the full synchronous host boundary, excluding cold
preparation. Guard configuration uses `PCU_VULKAN_GPU_BUSY_PATH`; the default is
the local RX 6900 XT counter. It requires three consecutive <=5% readings and
permits a bounded ten-second cooldown after a previous sample.

Facade-dependent development targets run from this repository checkout. Their
path-only development dependencies are omitted from normalized published
manifests. A standalone backend archive is not claimed to contain the complete
development harness.
