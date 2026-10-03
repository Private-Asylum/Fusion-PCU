//! Native full-width joint division qualification; independent frozen goldens.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/wide_div_rem/native/native.rs"]
#[allow(dead_code)] // Retained host/resident control helpers share the same native fixture.
mod native;
#[path = "../../benches/wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)] // Benchmark input/verification helpers share the independent oracle.
mod oracle;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/wide_div_rem/source/source.rs"]
#[allow(dead_code)] // Explicit direct/grid controls are qualified alongside ordinary source.
mod source;
use oracle::Integer;

#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuBindingRef,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuNumericalMode,
};
fn configure(mode: PcuNumericalMode) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(0),
        numerical_mode: mode,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
fn fault(error: &PcuExecutionError, lane: u64, kind: PcuExecutionFaultKind) {
    let actual = error.arithmetic_fault().unwrap();
    assert_eq!(
        (actual.kind, actual.invocation_id, actual.recovered),
        (kind, lane, false)
    );
}
fn goldens<T: Integer>(mode: PcuNumericalMode) {
    configure(mode);
    let (_, backend, _) = selection::selected_device();
    let bindings = source::direct_bindings::<T>();
    let requirements = fusion_pcu::PcuImplementationRequirements {
        numerical_mode: mode,
        ..fusion_pcu::PcuImplementationRequirements::DEFAULT
    };
    let builder = source::__direct_ir_with_float_underflow_policy::<T, 1>(
        &bindings,
        requirements.float_underflow,
        requirements.range_policy,
        requirements,
    )
    .unwrap();
    builder.with_ir(|ir| {
        let mut prepared = backend.prepare_host_kernel(ir).unwrap();
        let mut native = native::Native::new::<T, 1>(&backend, ir, 1);
        let rows = oracle::goldens::<T>();
        assert!(rows.len() >= 90);
        for (left, right, expected) in rows {
            assert_eq!(
                oracle::evaluate(left, right),
                expected,
                "independent base256 vs frozen BigInt"
            );
            assert_eq!(oracle::domain(left, right), expected.map(|_| ()));
            let mut q = [T::SENTINEL; 3];
            let mut r = q;
            for route in 0..2 {
                let result = if route == 0 {
                    source::direct::<T, 1>(&[left], &[right], &mut q, &mut r)
                } else {
                    prepared
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[left]),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), &[right]),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut q),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut r),
                        ])
                        .map_err(|error| match error {
                            fusion_pcu_cuda::CudaHostKernelError::CheckedExecutionFault(actual) => {
                                PcuExecutionError::ArithmeticFault(actual)
                            }
                            other => PcuExecutionError::BackendFailure(other.to_string()),
                        })
                };
                match expected {
                    Ok(pair) => {
                        result.unwrap();
                        assert_eq!((q[0], r[0]), pair);
                    }
                    Err(kind) => {
                        fault(&result.unwrap_err(), 0, kind);
                        assert_eq!((q, r), ([T::SENTINEL; 3], [T::SENTINEL; 3]));
                    }
                }
                assert_eq!(
                    (&q[1..], &r[1..]),
                    (&[T::SENTINEL; 2][..], &[T::SENTINEL; 2][..])
                );
            }
            native.upload(0, &[left], &[right]);
            let word = native.submit(0);
            match expected {
                Ok(pair) => {
                    assert_eq!(word, u64::MAX);
                    native.read_resident(&mut q, &mut r);
                    assert_eq!((q[0], r[0]), pair);
                }
                Err(kind) => assert_eq!(
                    word,
                    match kind {
                        PcuExecutionFaultKind::DivideByZero => 1,
                        PcuExecutionFaultKind::SignedDivisionOverflow => 2,
                        _ => unreachable!(),
                    }
                ),
            }
        }
        // A terminal fault must reset before successful quotient/remainder reuse.
        native.upload(0, &[T::ONE], &[T::ZERO]);
        assert_eq!(native.submit(0), 1);
        native.upload(0, &[T::ONE], &[T::ONE]);
        assert_eq!(native.submit(0), u64::MAX);
        native.read_resident(&mut [T::SENTINEL; 3], &mut [T::SENTINEL; 3]);
    });
}
#[test]
#[ignore = "requires the authorized native GPU; serial correctness only"]
fn six_wide_joint_div_rem_matches_573_independent_goldens() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        goldens::<i128>(mode);
        goldens::<u128>(mode);
        goldens::<fusion_pcu::PcuI256>(mode);
        goldens::<fusion_pcu::PcuU256>(mode);
        goldens::<fusion_pcu::PcuI512>(mode);
        goldens::<fusion_pcu::PcuU512>(mode);
    }
    global::clear_thread_cache().unwrap();
}

use fusion_pcu::PcuTensor;
#[allow(clippy::too_many_lines)] // A single concrete profile keeps publication/fault/retry boundaries together.
fn profile<T: Integer>(
    mut direct: impl FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
    mut grid: impl FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
    mut strict: impl FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
    mut resident: impl FnMut(
        &PcuTensor<T>,
        &PcuTensor<T>,
        &mut PcuTensor<T>,
        &mut PcuTensor<T>,
    ) -> Result<(), PcuExecutionError>,
) {
    const N: usize = 65;
    let mut q = vec![T::SENTINEL; N + 2];
    let mut r = q.clone();
    for phase in 0..4 {
        let (lhs, rhs) = oracle::inputs::<T>(N, phase);
        direct(&lhs, &rhs, &mut q, &mut r).unwrap();
        oracle::verify(&lhs, &rhs, &q, &r);
        grid(&lhs, &rhs, &mut q, &mut r).unwrap();
        oracle::verify(&lhs, &rhs, &q, &r);
        strict(&lhs, &rhs, &mut q, &mut r).unwrap();
        oracle::verify(&lhs, &rhs, &q, &r);
    }
    let (mut lhs, mut rhs) = oracle::inputs::<T>(N, 97);
    rhs[5] = T::ZERO;
    rhs[41] = T::ZERO;
    let before = (q.clone(), r.clone());
    for run in [
        &mut direct
            as &mut dyn FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
        &mut grid,
        &mut strict,
    ] {
        fault(
            &run(&lhs, &rhs, &mut q, &mut r).unwrap_err(),
            5,
            PcuExecutionFaultKind::DivideByZero,
        );
        assert_eq!((&q, &r), (&before.0, &before.1));
    }
    if let Some(negative_one) = T::NEGATIVE_ONE {
        lhs[2] = T::MIN;
        rhs[2] = negative_one;
        fault(
            &direct(&lhs, &rhs, &mut q, &mut r).unwrap_err(),
            2,
            PcuExecutionFaultKind::SignedDivisionOverflow,
        );
        fault(
            &grid(&lhs, &rhs, &mut q, &mut r).unwrap_err(),
            2,
            PcuExecutionFaultKind::SignedDivisionOverflow,
        );
        assert_eq!((&q, &r), (&before.0, &before.1));
        assert_eq!(
            lhs[2].pcu_checked_div(rhs[2]),
            Err(PcuExecutionFaultKind::SignedDivisionOverflow)
        );
        assert_eq!(
            lhs[2].pcu_checked_rem(rhs[2]),
            Err(PcuExecutionFaultKind::SignedDivisionOverflow)
        );
    }
    rhs.fill(T::ONE);
    assert!(direct(&lhs, &rhs, &mut q, &mut r[..N - 1]).is_err());
    assert_eq!((&q, &r), (&before.0, &before.1));
    direct(&lhs, &rhs, &mut q, &mut r).unwrap();
    oracle::verify(&lhs, &rhs, &q, &r);
    let input = source::identity(lhs.as_slice()).unwrap();
    let good_divisor = source::identity(rhs.as_slice()).unwrap();
    rhs[5] = T::ZERO;
    let bad_divisor = source::identity(rhs.as_slice()).unwrap();
    let mut rq = source::identity(before.0.as_slice()).unwrap();
    let mut rr = source::identity(before.1.as_slice()).unwrap();
    let mut short_rr = source::identity(&vec![T::SENTINEL; N - 1]).unwrap();
    assert!(resident(&input, &good_divisor, &mut rq, &mut short_rr).is_err());
    rq.read_into(&mut q).unwrap();
    assert_eq!(q, before.0);
    let mut short_result = vec![T::ZERO; N - 1];
    short_rr.read_into(&mut short_result).unwrap();
    assert_eq!(short_result, vec![T::SENTINEL; N - 1]);

    fault(
        &resident(&input, &bad_divisor, &mut rq, &mut rr).unwrap_err(),
        5,
        PcuExecutionFaultKind::DivideByZero,
    );
    // Both destinations may contain device writes after a fatal result; neither is publishable.
    for result in [rq.read_into(&mut q), rr.read_into(&mut r)] {
        assert!(matches!(
            result,
            Err(PcuExecutionError::Argument(
                global::PcuArgumentError::ResidentValueDiscarded
            ))
        ));
    }
    assert!(source::identity::<T>(&rq).is_err());
    assert!(resident(&input, &good_divisor, &mut rq, &mut rr).is_err());
    rq = source::identity(before.0.as_slice()).unwrap();
    rr = source::identity(before.1.as_slice()).unwrap();
    resident(&input, &good_divisor, &mut rq, &mut rr).unwrap();
    drop(input);
    drop(good_divisor);
    drop(bad_divisor);
    global::clear_thread_cache().unwrap();
    rq.read_into(&mut q).unwrap();
    rr.read_into(&mut r).unwrap();
    rhs.fill(T::ONE);
    oracle::verify(&lhs, &rhs, &q, &r);
}

fn transaction<T: Integer>(mode: PcuNumericalMode) {
    configure(mode);
    profile::<T>(
        |lhs, rhs, q, r| source::direct::<T, 65>(lhs, rhs, q, r),
        |lhs, rhs, q, r| source::grid::<T, 65>(lhs, rhs, q, r),
        |lhs, rhs, q, r| source::direct::<T, 65>(lhs, rhs, q, r),
        |lhs, rhs, q, r| source::direct::<T, 65>(lhs, rhs, q, r),
    );
}
#[test]
#[ignore = "requires the authorized native GPU; dual output rollback/discard/retry"]
fn six_wide_source_direct_grid_and_resident_transactions() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        transaction::<i128>(mode);
        transaction::<u128>(mode);
        transaction::<fusion_pcu::PcuI256>(mode);
        transaction::<fusion_pcu::PcuU256>(mode);
        transaction::<fusion_pcu::PcuI512>(mode);
        transaction::<fusion_pcu::PcuU512>(mode);
    }
}

fn independent<T: Integer>() -> usize {
    let rows = oracle::goldens::<T>();
    for &(left, right, expected) in &rows {
        assert_eq!(oracle::evaluate(left, right), expected);
        assert_eq!(oracle::domain(left, right), expected.map(|_| ()));
    }
    rows.len()
}
#[test]
fn independent_573_full_width_goldens_match_base256_oracle() {
    let total = independent::<i128>()
        + independent::<u128>()
        + independent::<fusion_pcu::PcuI256>()
        + independent::<fusion_pcu::PcuU256>()
        + independent::<fusion_pcu::PcuI512>()
        + independent::<fusion_pcu::PcuU512>();
    assert_eq!(total, 573);
}
