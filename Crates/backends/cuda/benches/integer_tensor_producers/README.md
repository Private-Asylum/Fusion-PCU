# Immutable integer source producers

Genuine `#[pcu]` source and explicit graph both compute `(input + constant) * uniform`
using typed immutable Constant and Uniform nodes. Fourteen integer carriers (signed
and unsigned 8/16/32/64/128/256/512 bits), extents65 and4096, Boundary/Strict,
Checked/BackendDefined compounds and Preserve/BackendOptimized precision are covered.
Range remains Reject, reproducibility Unspecified, and float underflow IEEE.
Graph PortableV1 and Clamp compounds are not admitted by this target.

Payloads use `pcu::constant(const { array })` and
`pcu::uniform_like(input, const { scalar })`. Mandatory Rust inline const prevents
runtime payload capture outside preparation cache identity. Uniform uses only the
anchor shape. Cold controls assert the exact high bits, options and selected input;
literal-only and uniform-only companions are genuinely zero-argument functions with
zero declared and selected bindings. Uniform uses an immutable Constant only as its
logical shape witness; no device input resource is fabricated. The arithmetic
pipeline still requires its real input, and its paired route timing/census remains
distinct from the224 cold zero-argument ownership/replay witnesses.

Every profile checks literal output escape/mutation/consumption, replay of original
bits, validity after cache clear, and explicit CUDA selection with an unused CPU
resident anchor. A used CPU resident still fails affinity preflight. Automatic
routing still examines original residents before capture and can choose opaque
execution for an unused foreign owner; this target does not claim that case.

The handwritten SDK control uses the retained independent checked-limb arithmetic
and separate ordered Add/Mul kernels/status words. It receives three dense banks;
source and graph keep immutable producers in cold storage. Values and phase fault
order match, but SDK transfer and ownership topology differs. Controls require full
high-bit output, untouched tails, Add versus Mul mixed-stage fault ordering, Reject
rollback, short-span refusal and successful retry before performance measurement.

The separate allocation-census build records64 changing calls per route, Rust heap
activity and PCU CUDA API deltas. Independent SDK heap is required to be zero and its
own explicit uploads/downloads/reset/launch/event vector is checked. SDK allocations,
module lifetime and SDK-internal allocations are outside those raw control counters.
Compilation and literal escape probes are cold; correctness or census elapsed time
is never latency evidence. No timing is inferred from previous input-bank fixtures.

CPU empty literal transport preserves its defined zero-extent constructor behavior
and does not write destination tails. Selected actual empty data inputs remain
refused. GPU zero-extent capability is not broadened or claimed by this target;
its native source controls remain extents65 and4096. The completed declared-anchor
predecessor certificate remains immutable and cannot be relabeled as zero-argument
source execution.
