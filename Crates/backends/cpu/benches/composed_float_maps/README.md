# Checked composed CPU maps

This target executes four actual caller-owned routes: genuine ordinary `#[pcu]`,
genuine prepared `#[pcu]`, independently written Dispatch IR, and independent native
arithmetic. Six formats, Reject/Clamp, all three underflow policies and extents
1/65 produce 288 route cases. Every case changes inputs for 64 separately counted
warm calls and checks exact outputs, tails, Rust heap activity and rescoring.

The native measured inputs are explicitly bounded to signed 0.5, 1 and 2.
Their `(x+x)*x` intermediates are exact normal values. F32/F64 native controls use
host arithmetic; the four low formats use a separately implemented binary64
dyadic unpack/midpoint pack model. These controls do not establish general native
F32/F64 floating-environment conformance or exceptional-path performance.
The source fixture separately compares all 789,504 low encoding/policy cases
through both genuine prepared source and independent IR against that model.

The new executable is scalar, Unspecified reproducibility, and cold-bounded to
four declarations, 64 steps and SSA identifiers below 256. Private output bytes
are retained cold. Fatal errors publish nothing; completed recovered Clamp calls
publish the entire useful output and return an error. Existing single-operation
and SIMD implementations remain separate. No Portable composed-map admission,
tensor graph Clamp, automatic CPU fallback or SIMD realization is claimed.

Use `cargo bench -p fusion-pcu-cpu --features source-composed --bench
composed_float_maps -- --test` for untimed semantic/census execution.
Native Linux/M4 frozen qualification and any statistical estimates are separate
evidence; a successful local build or census is not a timing result.
