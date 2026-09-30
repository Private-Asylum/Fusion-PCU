This target executes annotated `pcu::matmul` with `native_compound`, a prepared
graph with the same numerical options, and the safe typed rocBLAS SGEMM/DGEMM
API, including its validation and ownership work. This direct control is not a
raw SDK baseline. All routes use Boundary granularity, BackendDefined compound
arithmetic, Preserve precision, and Unspecified reproducibility. They allocate a
fresh output per call, without a numerical fault status buffer.

The original small/medium profiles cover F32/F64 at 4×4×4, 32×32×32, and
64×256×64. Positive integer operands produce products ≤35 and partial sums
≤8960; every sum is exactly representable in F32/F64 regardless of grouping.
This fixture permits exact parity checks; it makes no general BLAS determinism
claim. An independent integer oracle checks changing inputs on every iteration.

`full_host` includes two uploads, complete output readback and output
destruction. Inputs reuse storage after warmup. `resident_fresh_output`
alternates two preloaded input banks; readback and oracle checks occur outside
timing, while fresh output allocation and destruction remain included. Current
matched evidence requires the eligible owned-native batch path for source/graph
and `HipCompletionBatch` for the safe typed control. Earlier small-profile
results used device-wide waits in source/graph versus an event in the control,
so those results are differing-completion-boundary observations and will be
superseded. Ratios compare API boundaries and cannot isolate scheduler overhead.

Build: `cargo +stable bench -p fusion-pcu-rocm --features tensor --bench
native_matmul --no-run`. Correctness admission: run the executable with `--test`
only when GPU ownership is available. The ordinary Criterion primary requires
`<executable> --bench`. `--features allocation-census` builds a separate
resident Rust allocator census; driver/device allocations are outside that
census. New workload-heavy and final-event timing evidence requires hardware
acceptance after rebuilding the eligible owned-native batch path. GPU execution
remains coordinated with the parent and gated by fresh activity checks.

`PCU_NATIVE_PAIRED=1 <executable> --test` runs 36 interleaved triples per
profile, cycling all six orders equally. It reports observed median
source/native and graph/native ratios without a confidence interval, separately
from Criterion runs.

Both resident banks are restored after host measurements and checked through
every route before resident timing. The first partial primary run exposed this
setup requirement and is discarded; accepted results require the corrected
complete run.

Workload-heavy profiles add 1024×2048×1024 F32/F64. They construct fixed
matrices by repeating a single row into a heap Vec and converting its boxed
slice to a boxed array; there is no whole-matrix stack temporary. Only three
changing host phases are retained.

For those profiles, split K at floor(K/3). The first depth region uses
L[row,k]=row+1+phase and R[k,col]=1+(k+phase) mod3; the remainder uses
L[row,k]=1+(k+phase) mod2 and R[k,col]=col+1+phase. Two independent integer
weight sums determine every output in O(K+R*C). Different row/column weights
detect swapped axes. For phase0..2 every term/partial sum is a positive integer
bounded by4,902,228<2^24, making all grouping results exactly representable in
F32/F64. This controlled numerical domain does not imply universal backend
reduction parity. A small nonsquare large-K direct-dot test audits the
closed-form oracle, and a million-cell test checks both widths and heap
construction.
