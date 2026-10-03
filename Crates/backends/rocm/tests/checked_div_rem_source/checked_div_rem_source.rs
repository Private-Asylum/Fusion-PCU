//! Actual ordinary eight-width integer source proof, whole-host rollback and resident discard.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/checked_div_rem/oracle/oracle.rs"]
mod oracle;
#[path = "../../benches/checked_div_rem/source/source.rs"]
#[allow(dead_code)]
// Generated explicit preparation helpers are covered by the canonical benchmark.
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuTensor,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
use oracle::Integer;
fn selected() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        block_size: 256,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
fn fault(error: &PcuExecutionError, index: u64, kind: PcuExecutionFaultKind) {
    assert!(
        matches!(error, PcuExecutionError::ArithmeticFault(actual)
        if actual.invocation_id == index && actual.kind == kind && !actual.recovered),
        "{error:?}"
    );
}
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
#[test]
#[ignore = "requires an authorized GPU; correctness only, run serially"]
fn all_eight_widths_authored_direct_grid_strict_and_resident() {
    selected();
    macro_rules! width {
        ($module:ident, $ty:ty) => {
            profile::<$ty>(
                |lhs, rhs, q, r| source::$module::direct::<65>(lhs, rhs, q, r),
                |lhs, rhs, q, r| source::$module::grid::<65>(lhs, rhs, q, r),
                |lhs, rhs, q, r| source::$module::strict::<65>(lhs, rhs, q, r),
                |lhs, rhs, q, r| source::$module::direct::<65>(lhs, rhs, q, r),
            );
        };
    }
    width!(signed8, i8);
    width!(unsigned8, u8);
    width!(signed16, i16);
    width!(unsigned16, u16);
    width!(signed32, i32);
    width!(unsigned32, u32);
    width!(signed64, i64);
    width!(unsigned64, u64);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        numerical_options: fusion_pcu::PcuNumericalOptions {
            reproducibility: fusion_pcu::PcuReproducibility::PortableV1,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let mut q = [99_i64; 2];
    let mut r = q;
    assert!(source::signed64::direct::<1>(&[7], &[3], &mut q, &mut r).is_err());
    assert_eq!((q, r), ([99; 2], [99; 2]));
    global::clear_thread_cache().unwrap();
}
#[test]
#[ignore = "requires an authorized GPU; batches all eight-bit pairs, run serially"]
fn exhaustive_eight_bit_pairs_and_separate_terminal_faults() {
    selected();
    macro_rules! width { ($module:ident, $ty:ty) => {{
        let mut lhs = Vec::with_capacity(65536);
        let mut rhs = Vec::with_capacity(65536);
        for a in 0..=u8::MAX {
            for b in 0..=u8::MAX {
                let left = <$ty>::from_le_bytes([a]);
                let candidate = <$ty>::from_le_bytes([b]);
                lhs.push(left);
                rhs.push(if left.checked(candidate).is_ok() { candidate } else { 1 });
            }
        }
        let mut q = vec![99 as $ty; 65538]; let mut r = q.clone();
        source::$module::direct::<65536>(&lhs, &rhs, &mut q, &mut r).unwrap();
        oracle::verify(&lhs, &rhs, &q, &r);
        // Invalid pair semantics are covered separately, avoiding a fatal batch concealing all results.
        for left in [<$ty>::MIN, 0, <$ty>::MAX] {
            let mut q = [99 as $ty; 2]; let mut r = q;
            fault(&source::$module::direct::<1>(&[left], &[0], &mut q, &mut r).unwrap_err(), 0, PcuExecutionFaultKind::DivideByZero);
            assert_eq!((q, r), ([99; 2], [99; 2]));
        }
    }}; }
    width!(signed8, i8);
    width!(unsigned8, u8);
    let mut q = [99_i8; 2];
    let mut r = q;
    fault(
        &source::signed8::direct::<1>(&[i8::MIN], &[-1], &mut q, &mut r).unwrap_err(),
        0,
        PcuExecutionFaultKind::SignedDivisionOverflow,
    );
    assert_eq!((q, r), ([99; 2], [99; 2]));
    global::clear_thread_cache().unwrap();
}

#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[test]
#[ignore = "requires an authorized GPU; retained write/completion metadata, run serially"]
fn prepared_metadata_distinguishes_prelaunch_fault_and_success() {
    let (_, backend, _) = selection::selected_device();
    let bindings = source::signed64::direct_bindings();
    let builder = source::signed64::direct_ir::<3>(&bindings).unwrap();
    let mut prepared = backend.prepare_host_kernel(&builder.ir()).unwrap();
    assert!(!prepared.last_call_may_have_written());
    assert!(!prepared.last_call_completion_uncertain());
    let mut q = [99_i64; 4];
    let mut r = q;
    for (rhs, short) in [
        ([3_i64, 2, 2], true),
        ([3, 0, 2], false),
        ([3, 2, 2], false),
        ([3, 2, 2], true),
    ] {
        let result = prepared.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[7_i64, 8, 9]),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &rhs),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut q),
            PcuHostArgument::read_write(
                PcuBindingRef::new(0, 3),
                if short { &mut r[..2] } else { &mut r },
            ),
        ]);
        assert_eq!(prepared.last_call_may_have_written(), !short);
        assert!(!prepared.last_call_completion_uncertain());
        assert_eq!(result.is_ok(), !short && rhs[1] != 0);
    }
    assert_eq!((q, r), ([2, 4, 4, 99], [1, 0, 1, 99]));
    global::clear_thread_cache().unwrap();
}
#[test]
#[ignore = "requires an authorized GPU; mixed host/resident fatal publication, run serially"]
fn mixed_outputs_keep_host_rollback_and_discard_resident_result() {
    selected();
    let divisor = source::identity(&[3_i64, 0, 2]).unwrap();
    let mut r = source::identity(&[99_i64; 4]).unwrap();
    let mut q = [99_i64; 4];
    fault(
        &source::signed64::direct::<3>(&[7_i64, 8, 9], &divisor, &mut q, &mut r).unwrap_err(),
        1,
        PcuExecutionFaultKind::DivideByZero,
    );
    assert_eq!(q, [99; 4]);
    let mut observed = [0; 4];
    assert!(r.read_into(&mut observed).is_err());
    let mut rhs = [0; 3];
    divisor.read_into(&mut rhs).unwrap();
    assert_eq!(rhs, [3, 0, 2]);
    r = source::identity(&[99_i64; 4]).unwrap();
    source::signed64::direct::<3>(&[7_i64, 8, 9], &[3_i64, 2, 2], &mut q, &mut r).unwrap();
    r.read_into(&mut observed).unwrap();
    assert_eq!((q, observed), ([2, 4, 4, 99], [1, 0, 1, 99]));
    global::clear_thread_cache().unwrap();
}

#[fusion_pcu::pcu(invocations = 3, flag(clamp_range))]
fn clamp_double(input: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * 2.0;
}
#[test]
#[ignore = "requires an authorized GPU; recovered clamp stays publishable, run serially"]
fn recovered_clamp_stays_readable_and_fatal_clamp_is_discarded() {
    selected();
    let input = source::identity(&[f64::MAX, 1.0, -f64::MAX]).unwrap();
    let mut output = source::identity(&[99.0_f64; 5]).unwrap();
    for _ in 0..2 {
        let error = clamp_double(&input, &mut output).unwrap_err();
        let recovered = error.recovered_range_fault().unwrap();
        assert_eq!(recovered.kind, PcuExecutionFaultKind::ArithmeticOverflow);
        assert_eq!(recovered.invocation_id, 0);
        assert!(recovered.recovered);
        let mut actual = [0.0_f64; 5];
        output.read_into(&mut actual).unwrap();
        assert_eq!(
            actual.map(f64::to_bits),
            [f64::MAX, 2.0, -f64::MAX, 99.0, 99.0].map(f64::to_bits)
        );
        source::identity::<f64>(&output)
            .unwrap()
            .read_into(&mut actual)
            .unwrap();
    }
    let fatal = source::identity(&[f64::INFINITY, 1.0, 2.0]).unwrap();
    fault(
        &clamp_double(&fatal, &mut output).unwrap_err(),
        0,
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    assert!(matches!(
        output.read_into(&mut [0.0; 5]),
        Err(PcuExecutionError::Argument(
            global::PcuArgumentError::ResidentValueDiscarded
        ))
    ));
    assert!(source::identity::<f64>(&output).is_err());
    global::clear_thread_cache().unwrap();
}
