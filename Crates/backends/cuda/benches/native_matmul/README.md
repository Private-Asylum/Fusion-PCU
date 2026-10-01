# Retained native MatMul comparison

```sh
cargo +stable test -p fusion-pcu-cuda --features tensor,allocation-census \
  --test native_matmul_source -- --ignored --test-threads=1
cargo +stable bench -p fusion-pcu-cuda --features tensor --bench native_matmul
cargo +stable bench -p fusion-pcu-cuda --features tensor,allocation-census \
  --bench native_matmul -- --test
```

Run device tests serially and use an idle authorized CUDA device before benchmarking.
The canonical executable also checks utilization and foreign compute PIDs before
setup and each profile. `--test` on the primary executable verifies every group
without estimating throughput. The census feature selects a separate diagnostic
executable and excludes Criterion measurements.

Actual `#[pcu]` source and the explicit frozen graph execute the per-node cuBLASLt
plan selected during cold preparation. The independent Lt control owns its own
prepared descriptors, algorithm, selected workspace, runtime and stream. Classic
cuBLAS remains a separately identified reference. All routes explicitly request
Boundary, BackendDefined compound arithmetic and Unspecified reproducibility.
F32 Preserve uses 32F; BackendOptimized permits 32F_FAST_TF32. F64 uses 64F under
both permissions. This permits vendor exceptional results and does not certify
checked constituent operations or portable bits. Explicit gradual-underflow and
reject-subnormal requirements reject cold, as does a present experimental
`CUBLAS_BATCH_INVARIANCE_FLAGS` override. There is no CPU fallback.

The 48 full-host Criterion groups cover four routes, two widths, two precision
permissions and 4×2×4, 128×256×128, 1024×2048×1024 products. Each call includes two
uploads into retained inputs, one fresh output allocation, a submitted MatMul,
one terminal stream event/wait, full readback and output release. Two changing
input fixtures alternate across measured calls. Every output is compared with
an independent complete exact-domain oracle outside the accumulated duration.
Large fixtures use a proved split-depth integer domain; they do not establish
all vendor numerical behavior.

The 24 resident groups cover source, graph and independent Lt control at 4×2×4
and 64×96×32, both widths and permissions. Two retained resident input banks
alternate. A fresh output, submission, final event/wait and output release are
timed. Complete readback and oracle validation are outside the accumulated
resident duration and included in process wall. Host staging and identity
executables used to create source input owners are cleared before measurement.
The retained source/graph wrapper owners remain distinct from raw control owners.

The chosen workspace remains allocated outside warm measurements for every Lt
route; its exact size and algorithm/compute/library/runtime identity are printed
at cold preparation. CUDA workspace alignment is checked at 256 bytes. Operand
heuristics use the proved scalar alignment, without assuming imported operands
have allocation-base alignment. Lt's estimated candidate order does not rank
other libraries or prove fastest cross-library execution.

Each Criterion group uses 20 samples, one second warmup, two seconds requested
measurement and 95% intervals. Cold graph/control preparation is separate; source
cold first-call wall includes preparation and its first completed operation.
Independent Lt control submission and completion host phase means are diagnostic
observations across its executed calls, with no separate confidence interval.
They do not isolate GPU throughput or source scheduler overhead.

The separate Rust census counts only the caller thread's allocator attempts,
frees and requested bytes. It excludes CUDA/driver allocations and retains the
same validation callback; resident census also includes untimed readback/oracle
callback work. Lt API census separately counts API-table resolutions, heuristic
requests and MatMul calls. Warm source, graph and Lt control must show zero table
resolutions, zero heuristics and one MatMul. The table counter is not a raw
library-loader or driver-call count. Timing builds contain neither census.
