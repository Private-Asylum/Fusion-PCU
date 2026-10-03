# Checked Strict CUDA SGD

The canonical Criterion target executes an actual `#[pcu(flag(strict))]` function,
a frozen explicit graph and an independent native launch of the exact admitted
CUDA checker. Both F32 and F64 use a finite frozen F32 rate, widened exactly for
F64: first round/check `rate * gradient`, then round/check `weights - product`.
Unspecified reproducibility and Strict checking are required. Native compound
and optimized precision are permissions served by the same stronger ordered
checker; the full requested tuple remains in cold cache identity. No contraction
or FMA is used. Default checked Boundary and PortableV1 remain unsupported.

IEEE underflow means tiny **after rounding and inexact**. Exact subnormals remain
legal under IEEE; `reject_subnormal_result` rejects them and
`allow_gradual_underflow` permits inexact tiny arithmetic. Nonfinite operands and
arithmetic overflow return faults. Every lane checks Multiply before Subtract;
the first element/step fault wins independently of scheduling. The structured
fault's reduction index is zero. A failed call publishes no useful output, and a
checked producer completes before dependent graph nodes execute.

The 24 groups cover 65 and 1,048,576 lanes, both widths, full-host and resident
boundaries and all three routes. Each call creates a fresh output and private
fault word, resets status, launches once, waits for its terminal event, reads
status and releases the output. Full-host timing includes two changing uploads
and complete output readback. Resident timing alternates two retained input banks
and excludes complete output readback/oracle time. Output release is timed in
both boundaries. Source, graph and native controls retain their own input owners
and streams; no cross-provider import is inferred.

Two integer/dyadic input banks change both inputs on each call. The independent
complete-output oracle uses `(2 * weight_units - gradient_units) / 16`, avoiding
circularly re-running the generated arithmetic. Every lane is verified after
every call, outside the accumulated measurement duration. Fourteen exceptional
width/policy profiles separately compare source/graph/native fault coordinates
and outputs with the core checked reference, then retry all routes. Source
fixtures also verify retained borrowed owners, escaped output after cache/owner
drop, dependent optimizer chains, signed rate-zero identity and no-FMA witnesses.
This bounded domain does not certify arbitrary training graphs or portable bits.

```sh
cargo +stable build --locked --release -p fusion-pcu-cuda --bench strict_sgd --features tensor --message-format=json
cargo +stable build --locked --release -p fusion-pcu-cuda --bench strict_sgd --features tensor,allocation-census --message-format=json
cargo +stable test --locked -p fusion-pcu-cuda --test strict_sgd_source --features tensor -- --ignored --test-threads=1 --nocapture
cargo +stable run --locked -p fusion-pcu-cuda --example strict_sgd --features tensor --release
```

Use the exact Cargo JSON executable on authorized idle CUDA hardware. The
activity guard rejects foreign compute processes and waits for utilization to
settle to at most 10 percent before each case. Run `--test` correctness first;
then run the separate census executable and the uninstrumented primary with
`/usr/bin/time -p <executable> --bench`. Primary uses 20 samples, 500ms warmup,
2s requested measurement and Criterion's 95% intervals. Record full process wall,
source/artifact hashes, compiler, driver, toolkit/runtime and the raw Criterion
`new/` tree. No historical change estimate establishes current overhead.

Cold graph prewarming and native compilation are logged separately. The source
first-call wall includes its first completed operation. On the accepted lab run,
source/graph preparation selected NVCC (`sm_86`), while the independent native
control used NVRTC (`compute_86`). Both compiled the same integer checker with
`--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`; identical PTX or
machine code is not asserted. Cold means fresh PCU preparation, without a claim
that compiler or driver caches were empty. Compiler differences and separate
owners prevent interpreting the timings as isolated wrapper cost.

One untimed independent
native phase witness reports submission and completion/status/release durations;
it has no confidence interval and is not an isolated GPU throughput estimate.
The separate allocation-census binary skips primary timing. It reports warm
caller-thread Rust allocation attempts, frees, reallocations and requested bytes,
plus the backend's instrumented Runtime/Driver invoke-boundary counts and symbol
resolution attempts. These counters are absent from primary builds. API counts
cover the named CUDA invoke wrappers, excluding internal SDK work and separately
called cuBLAS/NVRTC/initialization routines; they count attempted calls rather
than successes. Resident census also includes the untimed full readback/oracle.
A warm call must launch once and load no module. Diagnostic counts do not imply
zero device/driver allocation or a general fastest-provider claim.

The current permission extension adds all four independent compound/precision
combinations and all three underflow policies at the small boundary. Original
larger default-profile cases remain. Benchmark group and census labels retain
the requested tuple. New hardware acceptance is recorded separately in the
backend plan; these executable definitions alone do not establish device proof.
Warm census runs 64 changing-input calls for each source/graph/native route,
including matched completion, explicit output readback, oracle and release.
