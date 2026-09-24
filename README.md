# fusion-pcu

`fusion-pcu` is a `no_std` instruction-decomposition and execution-contract library. It defines
program vocabulary, capability and discovery descriptors, bounded runtime registration,
activation law, and explicit memory admission types. A backend may compile, synthesize,
interpret, or delegate work. The core neither selects a backend nor owns a physical allocator.

## Contract status

The public API is **experimental**. The core checks on Rust 1.93.1 with default features
disabled, but no MSRV or publication guarantee is established yet.
Its current five built-in families are Dispatch, Stream, Command, Transaction, and Signal; they
are vocabulary, not a promise that every backend executes every operation. Unknown binary plugin
loading and a stable executable ABI are outside the current registry. The registry selects among
linked providers, and a consumer owns heterogeneous session storage or an enum when needed.

The currently tested executable profiles are deliberately small:

| Adapter | Executable evidence | Important limit |
| --- | --- | --- |
| Hosted CPU Stream | Shared U32 reference vectors pass | The common PIO profile permits one pattern and no parameters or bindings; broader CPU Stream behavior is separate. |
| RP2350 PIO Stream | Adapter compiles for `thumbv8m.main-none-eabihf` | Hardware parity on a board is not yet demonstrated. |
| ROCm Dispatch | RX 6900 XT runs a 65-value f32 map and Rust rocBLAS SGEMM | Only a bounded f32 indexed-map subset lowers to HIP; architecture is supplied by the caller. |
| SPIR-V/Vulkan Dispatch | Vulkan example runs a 256-value map on RX 6900 XT | SPIR-V 1.0–1.3 only; current emitter uses LocalSize `[1, 1, 1]` and the runner is a synchronous, fixed three-buffer, one-dimensional adapter. |
| AML Command/Signal | Firmware VM fixtures execute; target declarations exist | No AML method is lowered and executed through PCU yet. |

`PcuRuntimeDiscovery` exposes backend, target, device, context, and memory-domain facts with
generation-scoped references. `PcuDeviceActivation` opens an owned session from an explicitly
selected reference. ROCm additionally checks live PCI bus ID against its discovery snapshot.
The memory ledger models ratio admission and caller-serialized reservations; it does **not**
guarantee capacity against unrelated processes. `PcuMemoryProvider` separately describes
allocation, import, mapping, and transfer. The ROCm adapter implements allocation and checked
transfers, but import and mapping are unsupported. It reports device-wide used memory via HIP,
leaves process usage and physical device-local placement unknown, and rejects strict device-local
requests until placement can be affirmed. `allocate_with_policy` sequences a fresh snapshot,
atomic system/process ratio reservation, provider allocation, and rollback on failure. The ROCm
host example uses a 95/100 system-used policy; accounting remains the caller's responsibility
until all aliases and in-flight uses have ended.

The experimental owned Dispatch contract admits only tightly packed scalar buffers with one
element per logical invocation; it rejects other layouts and ports. Its completion law retains
resources through uncertain waits and releases them only after quiescence. ROCm now implements
that contract for the bounded f32 map subset. Cloned ROCm device buffers share an access gate,
so safe copies and rocBLAS reject overlapping use while a launched kernel owns an allocation.
The host example proves asynchronous submission, pre-wait access rejection, completion, and
readback on the RX 6900 XT. The original synchronous helper remains available. Command has a
typed read-result verifier, but there is no AML-to-PCU executor yet.

`F32MapBuilder` constructs the current scalar f32 indexed-map subset without heap allocation and
with automatic nonzero value IDs. Both ROCm and SPIR-V lowerers accept its output in tests. It is
not a Rust-to-PCU compiler, and value handles are scoped by caller-assigned kernel ID rather than
by a generative Rust lifetime. The optional dispatch macro has compile-fail tests for unsupported
statements and expressions and a renamed-crate compile-pass test.

The additive `dialect` module describes namespaced, versioned external operations with typed
operands/results and declared effects. A consumer supplies the authoritative operation signatures;
the core validates local value flow and can compose fragments with value-ID remapping. VM-style
and Stream-style test consumers exercise the protocol, but it does not execute external opcodes or
define a binary plugin ABI. The optional `fusion-pcu-tensor` crate is a separate f32 CPU reference
graph with shape checks and reverse-mode gradients. It does not yet lower through PCU or run on a
device backend.

For the precise first Stream semantics and test vectors, see
[`STREAM-U32-PIO-PROFILE.md`](STREAM-U32-PIO-PROFILE.md). For the remaining architecture work,
see `/volumes/projects/rust/fusion-pcu-plan.md` in this workspace.
