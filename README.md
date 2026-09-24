# fusion-pcu

`fusion-pcu` is a `no_std` instruction-decomposition and execution-contract library. It defines
program vocabulary, capability and discovery descriptors, bounded runtime registration,
activation law, and explicit memory admission types. A backend may compile, synthesize,
interpret, or delegate work. The core neither selects a backend nor owns a physical allocator.

## Contract status

The public API is **experimental**. The core checks and passes strict Clippy on Rust 1.93.1 with
default features disabled, but 1.93.1 is only the oldest installed stable toolchain tested here;
no MSRV or publication guarantee is established yet. Inert Fusion target-feature aliases were
removed from the core manifest; the retained `hosted` feature forwards to `std`.
Its current five built-in families are Dispatch, Stream, Command, Transaction, and Signal; they
are vocabulary, not a promise that every backend executes every operation. Unknown binary plugin
loading and a stable executable ABI are outside the current registry. The registry selects among
linked providers, and a consumer owns heterogeneous session storage or an enum when needed.

The currently tested executable profiles are deliberately small:

| Adapter | Executable evidence | Important limit |
| --- | --- | --- |
| Hosted CPU Stream | Shared U32 reference vectors pass | The common PIO profile permits one pattern and no parameters or bindings; broader CPU Stream behavior is separate. |
| Fusion RP2350 PIO Stream integration | Adapter compiles for `thumbv8m.main-none-eabihf` in the Fusion repository | Hardware parity on a board is not yet demonstrated; the Cortex-M adapter stays in Fusion. |
| ROCm Dispatch | RX 6900 XT runs a 65-value f32 map and Rust rocBLAS SGEMM | Only a bounded f32 indexed-map subset lowers to HIP; architecture is supplied by the caller. |
| SPIR-V/Vulkan Dispatch | Vulkan example runs a 256-value map on RX 6900 XT | SPIR-V 1.0–1.3 only; current emitter uses LocalSize `[1, 1, 1]` and the runner is a synchronous, fixed three-buffer, one-dimensional adapter. |
| Fusion AML Command/Signal integration | Firmware VM fixtures execute in the Fusion repository | No AML method is lowered and executed through PCU yet; the ACPI integration stays in Fusion. |

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
readback on the RX 6900 XT. ROCm now also exposes a reusable prepared executable for this f32
profile: lowering, HIP compilation, module/function resolution, and stream creation happen at
prepare time, while each launch owns its bindings and completion. The hosted proof submits it
repeatedly and reports cold preparation separately from warm bind/submit/wait samples. Per-launch
binding checks, argument allocation, access gates, and events remain; no zero-overhead claim is
made. The original synchronous helper remains available. Command has a
typed read-result verifier, but there is no AML-to-PCU executor yet.

`F32MapBuilder` constructs the current scalar f32 indexed-map subset without heap allocation and
with automatic nonzero value IDs. Both ROCm and SPIR-V lowerers accept its output in tests. It is
not a Rust-to-PCU compiler, and value handles are scoped by caller-assigned kernel ID rather than
by a generative Rust lifetime. The optional dispatch macro has compile-fail tests for unsupported
statements and expressions and a renamed-crate compile-pass test. The macro accepts only
`invocations = N`; the old logical-thread spelling is rejected. The core shape and context use
invocation terminology directly. The macro still accepts only a literal count and its narrow f32
assignment body; generic `R * C` expressions and grid-stride loops remain planned frontend work.

The additive `dialect` module describes namespaced, versioned external operations with typed
operands/results and declared effects. A consumer supplies the authoritative operation signatures;
the core validates local value flow and can compose fragments with value-ID remapping. An external
Stream integration test executes two composed fragments, and the separate `no_std`
`fusion-pcu-vm-reference` crate executes a bounded input/add/emit program using caller-owned
storage. Their operation meanings remain consumer-owned; the core does not execute those opcodes.
`PcuDialectProgram` adds exact named, typed input/output ports and per-operation typed scalar
immediates around a fragment. The consumer defines the port and immediate schema; validation
checks those contracts and SSA flow before execution. Port-bearing programs can form a
straight-line pipeline by explicitly binding every input of the second program to an output of
the first. Composition checks both source programs, the projected contract, caller storage, and
ID remapping before writing output buffers. The public boundary is the first program's inputs
and the second program's outputs. General graph wiring and region/control flow remain open.
Neither reference consumer establishes a general VM ABI or binary plugin interface.
`PcuDialectBuilder` is a bounded, caller-storage-backed Rust surface for fragment operations.
Its typed `bool`, `u32`, and `f32` SSA handles support fan-out; callers give separate builders
distinct scope IDs. It validates each append against the selected consumer support table, while
program ports and immediates are supplied through the program wrapper. The optional
`fusion-pcu-tensor` crate is a separate f32 CPU
reference graph with shape checks and reverse-mode gradients. It does not yet lower through PCU or
run on a device backend.

The Cortex-M, PIO hardware adapter, AML, HAL/PAL, and driver integrations remain in Fusion.
The PIO-named U32 Stream profile here is a shared IR conformance subset used by that consumer;
it contains no board driver. For its precise semantics and test vectors, see
[`STREAM-U32-PIO-PROFILE.md`](STREAM-U32-PIO-PROFILE.md).
