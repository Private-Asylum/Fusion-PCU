# Requested Portable unary maps

The measured workload is genuine `#[pcu(flag(deterministic))]` Neg/ReLU for F16, BF16, E4M3FN, E5M2, F32 and F64. Exact requested Boundary/Strict, all three underflow policies and Reject/observable Clamp are retained in source, prepared IR and native peers. The native control submits the same admitted kernel and retained status buffer, with matching full-host upload, completion, status read and logical-prefix readback. All three routes preserve two trailing host elements; native device-tail reads are cold correctness work only.

The canonical sweep has 864 source/prepared/native groups (288 workloads, two extents65/4096). Under `allocation-census`, each scope runs 64 changing-input calls and reports every API counter. Numerical native qualification and statistical timing are separate evidence; this document makes no latency claim.

Private fixtures compare 1,677,312 encoding-only unary results/statuses per provider: every F16/BF16/FP8 encoding and deterministic full-width F32/F64 samples across both operations and six underflow/range combinations. They also exercise fault-to-reset-to-success and subsequent sentinel reuse. Source fixtures exercise host rollback, earliest fatal/recovered precedence, inactive ReLU, broadcast/grid, resident discard and recovered publication. The common facade Portable unary gate covers all eight compound/precision/mode permission tuples independently.

Admission is the neutral single-op unary descriptor. Composed Portable maps, tensors and unrelated operations remain outside this offer. Identity used to form resident storage makes no tensor-arithmetic determinism claim.
