# Active candidate: exact PortableV1 floating unary

This candidate is checked Neg/ReLU on F16, BF16, E4M3FN, E5M2, F32 and F64.
It has passed local cold source/typing and strict lint; actual requested-header
Metal qualification is pending. It does not cover integer unary, tensor graphs,
composed maps, conversions, binary operations or whole-model reproducibility.
Earlier normal-policy certificates retain their original headers and limits.

`MetalPortableUnaryPlan::assess` uses the neutral bounded unary descriptor,
checks byte/status extents and retains the complete original request. Preparation
retains that request and actual read/store identities in `MetalPreparedFloatKernel`.
Cold host validation orders the actual output role independently of caller
parameter order. Declared-unread readonly roles have zero physical read span;
owned dispatch excludes them from resource requirements, and mixed validation
checks metadata without importing their resource/session. Legacy normal profiles
remain separate. No neutral discovery offer, measured cost or generic Portable
support is inferred from a coarse backend capability.

The existing encoding shaders use unsigned integer classification/sign selection:
finite Neg flips the sign bit; ReLU selects positive bits or canonical +0.
Nonfinite input is fatal, selected exact subnormal results remain legal under
IEEE-after-rounding and gradual underflow, and the explicit tightened policy
reports them. Observable Clamp retains the exact selected bits with a recovered
notice. BF16 and OFP8 retain their specified encodings and are not called IEEE
basic formats. F64 uses two U32 limbs, with no native double arithmetic.

[Apple's Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
revision 2026-06-04 sections2.1/3.1, defines the unsigned carriers, modular unsigned
operations, bitwise operations and logical right shifts. Emitted shifts are
bounded below32, narrowing casts fit their destination, and no signed overflow,
native floating computation, contraction, division or arrival-order reduction
selects results. Baseline guarded threadgroups avoid nonuniform-dispatch reliance.
Existing actual platform/device/compiler guards remain in force; no invented
M4-only numerical admission gate is added.

Each nonzero decoded status is checked against the frozen operation/policy fault
law and logical visited extent before fatal/recovered arbitration. Completion,
readback and fallible work precede host publication. In-place resident output can
be discarded after a possible-writing fatal call; recovered completion remains
readable and uncertain completion quarantines actual pending resources/session.
Unsupported profiles, foreign sessions, short extents and wrong access fail cold
or preflight. Private owners and existing mutable owners retain distinct laws.

The new required actual source fixture covers host/device/mixed/ordinary and
reordered-unused metadata, while the common all-six Portable unary gate covers
complete numerical tuples, direct/grid/broadcast and fault/tail/retry semantics.
A dedicated deterministic source benchmark pairs captured-source, hand-built IR,
direct native and ordinary source at the same current-host staging, fresh
output/status, terminal host-prefix read/drop boundary. Its separate64-changing
caller census records allocations and warm scorer callbacks; native/device heap
and runtime internal work are not inferred. Statistical timing requires recorded
idle activity and is separate from correctness and census qualification.
