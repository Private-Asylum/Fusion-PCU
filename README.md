# fusion-pcu

`fusion-pcu` is a `no_std` instruction-decomposition and execution-contract library. It defines
program vocabulary, capability and discovery descriptors, bounded runtime registration,
activation law, and explicit memory admission types. A backend may compile, synthesize,
interpret, or delegate work. The core neither selects a backend nor owns a physical allocator.

The repository root is a virtual Cargo workspace. The public library package lives in
`Crates/fusion-pcu`; its Rust modules sit beside that package's manifest without a `src` directory.
The macros, runner, backends, and dialects are separate packages under `Crates/`; hosted demos
live under `Examples/`.

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
| Hosted CPU Stream | Stream transforms pass local tests | The hosted CPU adapter and its broader Stream behavior live in Fusion. |
| Fusion RP2350 PIO Stream integration | Adapter compiles for `thumbv8m.main-none-eabihf` in the Fusion repository | Hardware parity on a board is not yet demonstrated; the Cortex-M adapter stays in Fusion. |
| ROCm Dispatch | RX 6900 XT runs a 250-invocation/2048-value grid-stride f32 map; separate HIP and rocBLAS smoke checks pass | Only a bounded f32 indexed-map subset and its structured grid-stride loop lower to HIP; device architecture discovery and runtime compilation have limited hardware coverage. |
| SPIR-V/Vulkan Dispatch | Vulkan example runs a 250-invocation/2048-value grid-stride map on RX 6900 XT | SPIR-V 1.0–1.3 only; current emitter uses LocalSize `[1, 1, 1]` and the runner is a synchronous, fixed three-buffer, one-dimensional adapter. |
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

The experimental owned Dispatch contract admits only tightly packed scalar buffers, sized for
the full indexed extent of a supported grid-stride loop; it rejects other layouts and ports. Its
completion law retains resources through uncertain waits and releases them only after quiescence. ROCm now implements
that contract for the bounded f32 map subset. Cloned ROCm device buffers share an access gate,
so safe copies and rocBLAS reject overlapping use while a launched kernel owns an allocation.
The host example prepares once, submits twice through the PCU owned-dispatch contract, and proves
completion and readback on the RX 6900 XT.
ROCm also exposes a reusable prepared executable for this f32
profile: lowering, HIP compilation, module/function resolution, and stream creation happen at
prepare time, while each launch owns its bindings and completion. The separate benchmark submits it
repeatedly and reports cold preparation separately from warm bind/submit/wait samples. Per-launch
binding checks, argument allocation, access gates, and events remain; no zero-overhead claim is
made. The original synchronous helper remains available. Command has a
typed read-result verifier, but there is no AML-to-PCU executor yet.

`PcuScalar` is sealed to `f32` and `u32` for now, with explicit host layout and lossless
little-endian encoding; it does not grant a backend a zero-copy device ABI or arithmetic support.
`F32MapBuilder` constructs the current scalar f32 indexed-map subset without heap allocation and
with automatic nonzero value IDs. Both ROCm and SPIR-V lowerers accept its output in tests. It is
not a Rust-to-PCU compiler, and value handles are scoped by caller-assigned kernel ID rather than
by a generative Rust lifetime. The optional dispatch macro has compile-fail tests for unsupported
statements and expressions and a renamed-crate compile-pass test. The macro is also exported as
`#[pcu]` for the guiding-star spelling. It accepts literal counts
and checked const-generic `usize` expressions such as `invocations = R * C`; specialization rejects
zero, overflow, and counts beyond `u32`. The old logical-thread spelling is rejected. The core
shape and context use invocation terminology directly. Macro resources use `&[f32]` for read-only
access and `&mut [f32]` for read/write access. The macro still accepts only its narrow f32
assignment body and one canonical grid-stride loop; generic element types and general control flow
remain planned frontend work. Within that subset, source may use
`pcu::context::global_invocation_id()` and `pcu::context::invocation_count()`; other function calls
are rejected. The loop carries a semantic extent distinct from the launch width,
so a smaller dispatch can cover a larger buffer with repeated per-lane iterations.
These references are parsed into IR access descriptors; the generated builder does not yet
borrow host buffers or enforce their lifetime and exclusivity through Rust's borrow checker.
The shared f32 profile admits add, subtract, multiply, and divide, but full IEEE edge behavior is
not yet normalized across CPU, HIP, and SPIR-V (in particular NaNs, signed zero, subnormals,
overflow/underflow, and contraction/reassociation). Current cross-backend conformance vectors
therefore use finite normal operands with exact results: `1.25 + 2.5 = 3.75`,
`7.5 - 2.25 = 5.25`, `1.5 * -2 = -3`, and `7.5 / 2.5 = 3`. These should be compared by
f32 bit pattern; they define the tested baseline, not a promise about excluded edge cases. The CPU
reference also accepts Min and Max as a CPU-only extension; the shared GPU profile rejects them.
For synchronous host execution, the bounded `PcuHostScalarBinding` path holds real `&[T]` and
`&mut [T]` references through admission and execution. Its backend trait is unsafe to implement:
device access must end even on an error before the call returns. Asynchronous execution continues
to use the owned-resource completion contract.
`fusion-pcu-cpu` is a separate opt-in backend crate. Its current allocation-free reference
interpreter covers the bounded f32 indexed-map and U32 Stream transform profiles; it does not
silently replace a selected device when that device rejects a kernel. Broader built-in dialect
coverage remains planned.

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
`fusion-pcu-tensor` crate defines an f32 graph with shape checks, explicit operation assessment,
execution planning, and CPU reference reverse-mode gradients. ROCm executes bounded MatMul through
rocBLAS and same-shape Add, Sub, Mul, and ReLU through PCU Dispatch on a selected device, without
implicit CPU fallback. It also supports transpose-aware MatMul, `ReLU` backward, scalar MSE, and reusable device
inputs. The ROCm training example builds a linear-regression gradient with `backward_mse`, then
composes an SGD update graph. It feeds each GPU-produced weight tensor directly into the next
step and checks both against CPU reverse-mode gradients. The example reads each output back for
verification; reusable intermediate storage, optimizer state, and a fully device-resident training
loop remain open. Synthesized elementwise operations cache a bounded
set of session-bound prepared Dispatch executables by operation and flattened element count,
avoiding recompilation on repeated execution while retaining per-call binding and device checks.

The Cortex-M, PIO hardware adapter, AML, HAL/PAL, and driver integrations remain in Fusion.

## License

Copyright 2026 Private Asylum LLC. Licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) and [NOTICE](NOTICE).
