//! Actual fourteen-width ordinary borrowed owners and joint immutable replacement.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuNumericalMode,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalOptions,
};
#[rustfmt::skip]
use super::{
    Sample,
    same,
    source,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
};
#[pcu(crate_path=::pcu_facade)]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
fn read<T: Sample>(owner: &PcuTensor<T>, expected: &[T]) {
    let sentinel = T::raw(17);
    let mut actual = vec![sentinel; expected.len() + 2];
    owner.read_into(&mut actual).unwrap();
    same(&actual[..expected.len()], expected);
    same(&actual[expected.len()..], &[sentinel; 2]);
}
fn publish<T: Sample>(policy: global::PcuExecutionPolicy) {
    global::configure(policy).unwrap();
    let left = [T::minimum(), T::raw(23), T::raw(17), T::raw(11)];
    let right = [T::raw(1), T::raw(5), T::raw(3), T::raw(2)];
    let a = retain(&left).unwrap();
    let b = retain(&right).unwrap();
    let sentinel = T::raw(19);
    let mut quotient = retain(&[sentinel; 6]).unwrap();
    let mut remainder = retain(&[sentinel; 9]).unwrap();
    let old_q = retain(&quotient).unwrap();
    let old_r = retain(&remainder).unwrap();
    let expected_q: [T; 6] = std::array::from_fn(|i| {
        if i < 4 {
            left[i].pcu_checked_div(right[i]).unwrap()
        } else {
            sentinel
        }
    });
    let expected_r: [T; 9] = std::array::from_fn(|i| {
        if i < 4 {
            left[i].pcu_checked_rem(right[i]).unwrap()
        } else {
            sentinel
        }
    });
    source::direct::<T, 4>(&a, &b, &mut quotient, &mut remainder).unwrap();
    read(&quotient, &expected_q);
    read(&remainder, &expected_r);
    let mut host = [sentinel; 7];
    source::grid::<T, 4>(&left, &b, &mut quotient, &mut host).unwrap();
    same(&host[..4], &expected_r[..4]);
    same(&host[4..], &[sentinel; 3]);
    source::direct::<T, 4>(&a, &right, &mut host, &mut remainder).unwrap();
    same(&host[..4], &expected_q[..4]);
    same(&host[4..], &[sentinel; 3]);
    let bad = [T::raw(1), T::raw(0), T::raw(3), T::raw(0)];
    assert!(
        matches!(source::direct::<T, 4>(&a, &bad, &mut quotient, &mut remainder), Err(PcuExecutionError::ArithmeticFault(fault)) if fault.kind==PcuExecutionFaultKind::DivideByZero && fault.invocation_id==1 && !fault.recovered)
    );
    read(&quotient, &expected_q);
    read(&remainder, &expected_r);
    if matches!(
        left[0].pcu_checked_div(T::negative_one()),
        Err(PcuExecutionFaultKind::SignedDivisionOverflow)
    ) {
        let bad = [T::negative_one(), T::raw(5), T::raw(3), T::raw(0)];
        assert!(
            matches!(source::direct::<T, 4>(&a, &bad, &mut quotient, &mut remainder), Err(PcuExecutionError::ArithmeticFault(fault)) if fault.kind==PcuExecutionFaultKind::SignedDivisionOverflow && fault.invocation_id==0 && !fault.recovered)
        );
        read(&quotient, &expected_q);
        read(&remainder, &expected_r);
    }
    let mut short = [sentinel; 3];
    assert!(source::direct::<T, 4>(&a, &right, &mut quotient, &mut short).is_err());
    read(&quotient, &expected_q);
    same(&short, &[sentinel; 3]);
    source::direct::<T, 4>(&a, &b, &mut quotient, &mut remainder).unwrap();
    read(&old_q, &[sentinel; 6]);
    read(&old_r, &[sentinel; 9]);
    global::clear_thread_cache().unwrap();
    let fresh = retain(&left).unwrap();
    source::direct::<T, 4>(&fresh, &b, &mut quotient, &mut remainder).unwrap();
    drop(a);
    drop(b);
    drop(fresh);
    drop(old_q);
    drop(old_r);
    read(&quotient, &expected_q);
    read(&remainder, &expected_r);
}
fn qualify<T: Sample>() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    publish::<T>(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Mlx,
                        numerical_mode: mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            ..Default::default()
                        },
                        float_underflow: underflow,
                        ..Default::default()
                    });
                }
            }
        }
    }
}
#[test]
#[ignore = "Requires actual MLX fourteen-width ordinary joint host/resident/mixed fault and tail publication."]
fn fourteen_width_ordinary_joint_publication_and_escaped_siblings() {
    qualify::<u8>();
    qualify::<i8>();
    qualify::<u16>();
    qualify::<i16>();
    qualify::<u32>();
    qualify::<i32>();
    qualify::<u64>();
    qualify::<i64>();
    qualify::<u128>();
    qualify::<i128>();
    qualify::<PcuU256>();
    qualify::<PcuI256>();
    qualify::<PcuU512>();
    qualify::<PcuI512>();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
