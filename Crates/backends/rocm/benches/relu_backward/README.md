# ReLU derivative source/graph/native comparison

Canonical Criterion target `relu_backward` runs actual per-function `#[pcu]`
Strict checked F32/F64 selection beside an explicitly frozen graph and the same
validated source compiled/launched through a direct native control. Both inputs
change between two physical banks. Every invocation validates complete output
against independent signed integer input construction outside timing.

Shapes 65, 65,536, and 1,048,576 compare full-host (upload, terminal completion/status,
readback, output release) and resident (terminal completion/status/release,
payload readback outside timing) boundaries. Cold setup/compilation is reported
separately. An opt-in allocation census is a separate run; it includes untimed
output validation calls and does not establish isolated device throughput.

20 samples, 500 ms warmup, 2 s requested measurement, Criterion 95% intervals.
Activity guards reject foreign compute processes and busy GPUs. Compilation or
registration alone is not current hardware/performance acceptance.
