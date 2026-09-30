This target executes actual `#[pcu(flag(non_strict), flag(native_compound))]`
source, a prepared owned graph with BackendDefined compound arithmetic, and a
private HIP squared-difference launch followed by the safe typed rocBLAS
`sasum_scaled` API. All routes use F32, Preserve precision, and Unspecified
reproducibility. The private kernel preserves a rounded F32 difference with
`volatile` before squaring. It waits for its terminal launch event; ASUM and
scalar scaling use the existing synchronous reduction boundary. The control
includes typed validation and ownership work and is not a raw SDK baseline.

Profiles cover 65 and 1,048,576 elements. Each call allocates fresh squared
scratch and a fresh rank-zero output, with no numerical fault status. Inputs
reuse storage. `full_host` includes two uploads, complete four-byte scalar
readback, and output destruction. `resident_fresh_output` alternates two
preloaded banks and excludes readback/oracle work from timing; output
destruction remains included. Both banks are restored after the host profile.
Every host phase and both banks pass every route before measurement.

Prediction is `(index mod5)-2+phase`, target is `(index mod3)-1`, for phase0..2.
An independent integer oracle sums squared differences, proves the sum is
below2^24 at both extents, then multiplies by the specified rounded F32
reciprocal count. All positive integer partial sums are exactly representable,
so arbitrary grouping of this controlled reduction cannot change the sum. This
does not imply arbitrary-input vendor determinism or portable
subnormal/special-value behavior. The standalone pure oracle test compares the
integer oracle with constructed F32 inputs using exact F64 arithmetic; run
`rustc +stable --edition 2024 --test
Crates/backends/rocm/benches/native_mse/oracle_tests.rs -D warnings -o
/tmp/pcu-native-mse-oracle-test` and then that executable.

Build with `cargo +stable bench -p fusion-pcu-rocm --features tensor --bench
native_mse --no-run`. Only run the executable with `--test` after GPU ownership
and activity clearance. Criterion timing requires `--bench`. The optional
`allocation-census` feature builds a separate resident warm caller-thread Rust
census, excluding SDK/device allocations. `PCU_MSE_PAIRED=1 <executable> --test`
runs 36 triples in six balanced orders per profile, with observed medians and no
confidence interval. Primary, paired and census are separate runs. No hardware
acceptance or timing is claimed by authoring this target.
