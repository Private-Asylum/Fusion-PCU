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
literal-only and uniform-only source have zero selected bindings, while retaining a
declared type/shape anchor. Actual zero-argument macro signatures remain unsupported.

Every profile checks literal output escape/mutation/consumption, replay of original
bits, validity after cache clear, and explicit ROCm selection with an unused CPU
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
activity and PCU ROCm API deltas. Independent SDK heap is required to be zero and its
own explicit uploads/downloads/reset/launch/event vector is checked. SDK allocations,
module lifetime and SDK-internal allocations are outside those raw control counters.
Compilation and literal escape probes are cold; correctness or census elapsed time
is never latency evidence. No timing is inferred from previous input-bank fixtures.

Qualified successor source `fa8f844d...` passes all 672 native source/graph/SDK semantic keys and a separate 672-key 64-changing-call census, with strict/pure and224 frozen CPU references. Source retains3 Rust allocations/frees percall; graph6; handwritten SDK0. Real producers reduce source H2D3→1 versus the earlier input-bank fixture. No latency claim. Failed predecessor `bd74706d...` remains preserved (literal-only zero-input shared executor refusal before semantic markers; census never executed). The repaired empty-binding executor uses this distinct successor proof. Actual zero-argument source signatures remain separate from the qualified zero-selected-binding functions.
