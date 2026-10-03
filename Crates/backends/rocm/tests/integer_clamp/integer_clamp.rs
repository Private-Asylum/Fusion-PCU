//! All14widths saturated bytes and observable recovered publication through genuine source.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/integer_clamp/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/integer_clamp/source/source.rs"]
#[allow(dead_code)] // Canonical benchmark and source proof share all authored entry points.
mod source;
#[rustfmt::skip]
use fusion_pcu::{global,PcuExecutionError,PcuExecutionFaultKind,PcuNumericalMode,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuTensor};
use oracle::Format;
fn reference<T: Format>() {
    for op in 0..3 {
        for (a, b, want, code) in oracle::rows::<T>(op) {
            let result = match op {
                0 => a.pcu_clamped_add(b),
                1 => a.pcu_clamped_sub(b),
                2 => a.pcu_clamped_mul(b),
                _ => unreachable!(),
            };
            let actual = if code == 0 {
                result.unwrap()
            } else {
                let fault = result.unwrap_err();
                assert_eq!(
                    fault.kind(),
                    if code == 3 {
                        PcuExecutionFaultKind::ArithmeticOverflow
                    } else {
                        PcuExecutionFaultKind::ArithmeticUnderflow
                    }
                );
                fault.clamped_value()
            };
            assert_eq!(actual.encode_le().as_ref(), want.encode_le().as_ref());
        }
    }
}
fn host<T: Format, const N: usize>(
    a: &[T],
    b: &[T],
    out: &mut [T],
    op: u32,
    grid: bool,
) -> Result<(), PcuExecutionError> {
    match (op, grid) {
        (0, false) => source::add::direct::<T, N>(a, b, out),
        (0, true) => source::add::grid::<T, N>(a, b, out),
        (1, false) => source::sub::direct::<T, N>(a, b, out),
        (1, true) => source::sub::grid::<T, N>(a, b, out),
        (2, false) => source::mul::direct::<T, N>(a, b, out),
        (2, true) => source::mul::grid::<T, N>(a, b, out),
        _ => unreachable!(),
    }
}
fn resident<T: Format, const N: usize>(
    a: &PcuTensor<T>,
    b: &PcuTensor<T>,
    out: &mut PcuTensor<T>,
    op: u32,
) -> Result<(), PcuExecutionError> {
    match op {
        0 => source::add::grid::<T, N>(a, b, out),
        1 => source::sub::grid::<T, N>(a, b, out),
        2 => source::mul::grid::<T, N>(a, b, out),
        _ => unreachable!(),
    }
}
fn bits<T: Format>(actual: &[T], wanted: &[T]) {
    for (a, b) in actual.iter().zip(wanted) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn format<T: Format>() {
    const N: usize = 129;
    for op in 0..3 {
        for phase in [1, 17, 51] {
            let (a, b, want, status) = oracle::inputs::<T>(N, phase, op);
            let mut out = vec![T::sentinel(); N + 2];
            for grid in [false, true] {
                oracle::result(&host::<T, N>(&a, &b, &mut out, op, grid), status);
                oracle::verify(&want, &out);
            }
            let ra = source::identity(a.as_slice()).unwrap();
            let rb = source::identity(b.as_slice()).unwrap();
            let mut ro = source::identity(out.as_slice()).unwrap();
            for _ in 0..2 {
                oracle::result(&resident::<T, N>(&ra, &rb, &mut ro, op), status);
                ro.read_into(&mut out).unwrap();
                oracle::verify(&want, &out);
            }
            let escaped = source::identity(&ro).unwrap();
            let zero = vec![T::small(0); N];
            let rz = source::identity(zero.as_slice()).unwrap();
            resident::<T, N>(&rz, &rz, &mut ro, op).unwrap();
            escaped.read_into(&mut out).unwrap();
            oracle::verify(&want, &out);
            oracle::result(&resident::<T, N>(&ra, &rb, &mut ro, op), status);
            // Recovered publication is Ready; mixed input preflight failure preserves its exact bytes.
            assert!(
                match op {
                    0 => source::add::grid::<T, N>(&a[..N - 1], &rb, &mut ro),
                    1 => source::sub::grid::<T, N>(&a[..N - 1], &rb, &mut ro),
                    2 => source::mul::grid::<T, N>(&a[..N - 1], &rb, &mut ro),
                    _ => unreachable!(),
                }
                .is_err()
            );
            ro.read_into(&mut out).unwrap();
            oracle::verify(&want, &out);
            let mut a_back = vec![T::sentinel(); N];
            ra.read_into(&mut a_back).unwrap();
            bits(&a_back, &a);
        }
        // Every independent range boundary has deterministic earliest logical attribution;
        // readonly scalar broadcast uses its original bytes, never a host arithmetic shortcut.
        for (a, b, want, code) in oracle::rows::<T>(op).into_iter().filter(|row| row.3 != 0) {
            let left = vec![a; N];
            let mut out = vec![T::sentinel(); N + 2];
            let result = match op {
                0 => source::add::broadcast::<T, N>(&left, &b, &mut out),
                1 => source::sub::broadcast::<T, N>(&left, &b, &mut out),
                2 => source::mul::broadcast::<T, N>(&left, &b, &mut out),
                _ => unreachable!(),
            };
            oracle::result(&result, 0x8000_0000_0000_0000 | u64::from(code));
            oracle::verify(&[want; N], &out);
        }
    }
}
fn configure(mode: PcuNumericalMode, underflow: PcuFloatUnderflowPolicy) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        block_size: 256,
        numerical_mode: mode,
        float_underflow: underflow,
        range_policy: PcuRangePolicy::Reject,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
macro_rules! all {
    ($f:ident) => {
        $f::<i8>();
        $f::<u8>();
        $f::<i16>();
        $f::<u16>();
        $f::<i32>();
        $f::<u32>();
        $f::<i64>();
        $f::<u64>();
        $f::<i128>();
        $f::<u128>();
        $f::<fusion_pcu::PcuI256>();
        $f::<fusion_pcu::PcuU256>();
        $f::<fusion_pcu::PcuI512>();
        $f::<fusion_pcu::PcuU512>();
    };
}
#[test]
fn independent_bigint_saturation_goldens_agree_with_core() {
    all!(reference);
}
#[test]
#[ignore = "authorized actual GPU, correctness only, run serially"]
fn fourteen_width_clamp_publication_and_source_contract() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for underflow in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            configure(mode, underflow);
            all!(format);
        }
    }
}

#[test]
#[ignore = "authorized actual GPU, exact permission axes and negative admission"]
fn stronger_scalar_clamp_contract_preserves_independent_permissions_and_portable_rejection() {
    use fusion_pcu::{
        PcuCompoundArithmeticPolicy, PcuPrecisionPolicy, PcuReproducibility, PcuNumericalOptions,
    };
    for compound_arithmetic in [
        PcuCompoundArithmeticPolicy::Checked,
        PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Rocm,
                device: Some(0),
                numerical_mode: PcuNumericalMode::Strict,
                numerical_options: PcuNumericalOptions {
                    compound_arithmetic,
                    precision,
                    reproducibility: PcuReproducibility::Unspecified,
                },
                ..Default::default()
            })
            .unwrap();
            global::clear_thread_cache().unwrap();
            format::<i64>();
            format::<fusion_pcu::PcuI512>();
        }
    }
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        numerical_options: PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let mut out = [17i64; 3];
    let before = out;
    #[cfg(feature = "allocation-census")]
    fusion_pcu_rocm::reset_rocm_api_census();
    assert!(source::add::direct::<i64, 1>(&[i64::MAX], &[1], &mut out).is_err());
    assert_eq!(out, before);
    #[cfg(feature = "allocation-census")]
    assert_eq!(fusion_pcu_rocm::rocm_api_census().kernel_launches, 0);
}

fn cold<T: Format>() {
    macro_rules! profile {
        ($op:ident,$route:ident,$bindings:ident,$ir:ident) => {{
            let bindings = source::$op::$bindings::<T>();
            let builder = source::$op::$ir::<T, 129>(&bindings).unwrap();
            builder.with_ir(|view| {
                let mut ir = *view;
                assert_eq!(
                    ir.numerical_requirements.range_policy,
                    PcuRangePolicy::Clamp
                );
                let emitted = fusion_pcu_rocm::lower_dispatch_to_hip_source(&ir).unwrap();
                assert!(emitted.contains("0x8000000000000000ull"));
                assert!(emitted.contains("fusion_range_fault_recorded"));
                ir.numerical_requirements.range_policy = PcuRangePolicy::Reject;
                assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&ir).is_err());
            });
        }};
    }
    // Explicit route names keep generated helper paths reviewable and stable.
    profile!(add, direct, direct_bindings, direct_ir);
    profile!(add, grid, grid_bindings, grid_ir);
    profile!(sub, direct, direct_bindings, direct_ir);
    profile!(sub, grid, grid_bindings, grid_ir);
    profile!(mul, direct, direct_bindings, direct_ir);
    profile!(mul, grid, grid_bindings, grid_ir);
}
#[test]
fn clamp_requested_headers_lower_and_mismatches_reject_all14() {
    all!(cold);
}
