# Literal-fed floating operators

Authoring preflight currently covers six carriers F32/F64/F16/BF16/E4M3FN/E5M2, actual2×3 matrices and flat65, with genuine Constant-fed Sub/ReLU/ReLUBackward and UniformLike-fed Div. ReLU has a genuine zero-argument signature. Requested Boundary/Strict × two compound permissions × two precision permissions × three UF policies gives1,152 source cohorts. Scalar Sub/Div/ReLU modeNone is intentional; the derivative retains its requested compound mode. Producer mode/UFNone, exact math UF/options, selected input count and actual shape are asserted.

`PCU_LITERAL_OPERATOR_CPU_REFERENCE=1 cargo bench -p fusion-pcu-rocm --features tensor --bench literal_operators --offline` is explicitly a CPU source/metadata preflight. It checks healthy full bits and output tails. This authoring target is not yet a native backend qualification or a latency benchmark. Native standalone arithmetic controls, explicit typed IR, fault/phase/ownership cases and separate allocation/API census are required successor gates. Existing PCU-lowered HIP derivative controls are not independent references.

Normal Unspecified/Reject only; no Portable, Clamp, compact or compound-operation widening. Root owns production Constant upload changes; this target owns source/driver/control authoring only.
