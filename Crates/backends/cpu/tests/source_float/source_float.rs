//! Ordinary generic checked-float source reuses the exact concrete CPU prepared profiles.
use pcu_facade::pcu;
#[rustfmt::skip]
use pcu_facade::{
    global,
    global::PcuBackendChoice,
    global::PcuExecutionPolicy,
    PcuCheckedFloat,
    PcuExecutionFaultKind,
};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn divide<T: PcuCheckedFloat, const N: usize>(lhs: &[T], rhs: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = lhs[id] / rhs[id];
}
#[test]
fn ordinary_generic_f32_f64_keep_fault_rollback_and_tails() {
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Cpu,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    macro_rules! width {
        ($ty:ty) => {{
            let mut output = [99.0 as $ty; 4];
            divide::<$ty, 3>(&[6.0, -7.0, 20.0], &[2.0, 2.0, 4.0], &mut output).unwrap();
            assert_eq!(output.map(<$ty>::to_bits), [3.0 as $ty, -3.5, 5.0, 99.0].map(<$ty>::to_bits));
            let before = output;
            assert!(matches!(divide::<$ty, 3>(&[6.0, -7.0, 20.0], &[2.0, 0.0, 4.0], &mut output),
                Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id == 1 && fault.kind == PcuExecutionFaultKind::DivideByZero));
            assert_eq!(output.map(<$ty>::to_bits), before.map(<$ty>::to_bits));
            divide::<$ty, 3>(&[6.0, -7.0, 20.0], &[2.0, 2.0, 4.0], &mut output).unwrap();
        }};
    }
    width!(f32);
    width!(f64);
}
