# CUDA source facade example

The executable selects CUDA through `fusion_pcu::global::configure` and calls an ordinary
`#[pcu]` function. All resource management and dispatch policy remain inside the facade:

```sh
cargo run -p fusion-pcu-example-cuda --release
cargo bench -p fusion-pcu-example-cuda --bench source_facade
```

The Criterion target uses the same checked F32 source specialization,
`output[id] = input[id] * 2.0 + 1.0`, at 65 and 1,048,576 invocations. Multiplication and
addition remain separate checked operations; the reference checks exact output bits.
The native comparison launches the CUDA executable prepared from `transform_ir`, with
its exact buffer/status ABI and launch geometry. This preserves the generated arithmetic
semantics and partial final block guard, including the 65-invocation case.

| Group | Routes | Included in each measurement |
| --- | --- | --- |
| Host | Ordinary automatically staged RAM call, explicitly prepared typed source call, native same lowering | Input upload, launch, terminal completion wait, checked status readback, output download |
| Resident | Ordinary `&PcuTensor` / `&mut PcuTensor` call, native retained buffers | Launch, terminal completion wait, checked status readback |

Compilation, device selection, initial device/host buffer allocation, changing input preparation and exact
output oracles run outside the primary timings. Host inputs change at element zero for every iteration;
resident iterations alternate between two preloaded input banks. All routes retain a
terminally observed successful status sentinel and reset after an arithmetic fault.
First-fault, recovery and two distinct successful input preflights establish that contract
before measurement. Host measurements use `iter_custom`: each full call is timed, then every
output element is checked outside the accumulated duration, including native output already
downloaded by that call. One second warmup, two second measurement and twenty samples apply.
The resident native route shares one output, status, executable and stream across both input
banks, matching the ordinary resident route. Device ordinal and block size 256 are pinned
identically; selected device facts and exact native geometry are logged. The native launch follows the existing backend benchmark's documented
unsafe launch pattern; allocation, copies and terminal event waits use public CUDA APIs.

Per-call event/owner allocation remains inside each call; no claim of zero Rust heap or
CUDA driver allocations is made. Initial buffer allocation wall time is reported separately.
An opt-in `allocation-census` feature replaces the Rust global allocator with a forwarding
counter and reports allocation attempts, reallocation attempts, frees and requested bytes for
one warm caller-thread call per route. Requested bytes sum allocation layouts and reallocation
destination sizes; they are not live or peak heap usage. Run it with
`cargo bench -p fusion-pcu-example-cuda --bench source_facade
--features allocation-census -- --test`. This counts the caller thread's Rust global allocator
activity only, excluding other threads and device/driver/runtime internal allocations.
Existing inputs, buffers and output oracles remain outside capture; reporting starts after
capture has been disabled. Primary timings use the default feature set, where
the counter and its allocator are completely absent.

A fatal borrowed mutable resident fault invalidates its destination; its preflight rebuilds
that destination before verifying recovery. The native preflight retains its destination.
These differing fault publication contracts are excluded from nominal success timings.
Host complete-writer proof elides the incoming output upload for all three host routes.
Resident output download and verification occur outside measurement for both routes.

Separate `diagnostic/.../process_wall` lines report cold preparation, allocation, first call,
terminal resident download and whole-process wall time. They are observations outside the
Criterion samples, not a GPU-only or warm-facade speedup claim. Raw kernel-only timings and
CUDA graph replay are different boundaries and are deliberately not mixed into these groups.

Benchmark setup, native execution and reference machinery are in `benches/support/`;
`benches/source_facade.rs` composes and executes the measurements. CUDA hardware and a usable
CUDA compiler are required to execute either target; compilation alone checks no GPU result.

### Strict MatMul benchmark

`cargo +stable bench -p fusion-pcu-example-cuda --bench strict_matmul` runs executed
`#[pcu(flag(strict))]` source against an explicit prepared graph and a native launch of the
same public generated ordered checker. F32/F64 profiles are 4×2×4, 32×32×32 and 64×256×64.
The selected provider and device ordinal 0 are explicit, with block size 256 on every route.
Run only after the lab GPU ownership gate and backend hardware conformance tests pass.
The benchmark also checks utilization before setup and each profile. Execute the binary on
the CUDA hardware host with `nvidia-smi` and NVRTC available; a build on another host is only
a compile check, and its standalone benchmark binary must be copied to the hardware host.

Every strict route uses increasing K, separate destination-precision multiply/add checks,
fresh output, fresh private fault status, terminal completion/status and timed output release.
Full-host timing includes input upload and completed readback; resident timing alternates two
preloaded input banks and excludes readback. Explicit graph and native inputs reuse allocated
storage. The source route owns its ordinary staging/cache behavior. These compare complete API
boundaries and do not isolate scheduler cost. All output bits are checked outside the summed
elapsed time. Thirty-six resident interleaved triples repeat all six strict route orders equally;
printed ratios are medians of paired ratios, with no confidence interval.

Boundary mode is asserted Unsupported, without treating rejection latency as compute throughput.
Vendor BLAS is a separate unchecked, different-semantics throughput control with a preallocated
output. Its timing does not establish checked numerical equivalence. Nominal dyadic inputs permit
bitwise comparison without granting the vendor route a fault contract.

Untimed F32/F64 preflights compare source, explicit graph and native checks against the core
checked reference for intermediate/final underflow, multiply/add overflow, later cancellation,
exact subnormal and zero controls, gradual/reject-subnormal policies, no-FMA rounding and invalid
operands. A successful retry follows every case through the same prepared graph/native status.
No failed output is read. `--features allocation-census` selects a separate resident Rust allocation
census run and skips Criterion timings; device and driver internal allocations are excluded.

