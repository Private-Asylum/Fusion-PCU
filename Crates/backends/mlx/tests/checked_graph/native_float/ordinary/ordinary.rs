//! Ordinary owned `ReLU` uses the same frozen selected effects and authentic session.
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuCheckedFloat,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuImplementationRequirements,
    PcuFloatUnderflowPolicy,
    PcuTensor,
};
#[rustfmt::skip]
use super::{
    Encoding,
    same,
};
#[pcu(crate_path=::pcu_facade,flag(allow_gradual_underflow))]
fn local_gradual<T: PcuCheckedFloat>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}
#[pcu(crate_path=::pcu_facade)]
fn consumed<T: PcuCheckedFloat>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}
fn read<T: Encoding>(owner: &PcuTensor<T>, expected: &[T; 5], sentinel: T) {
    let mut actual = [sentinel; 7];
    owner.read_into(&mut actual).unwrap();
    assert_eq!(owner.shape(), &[5]);
    same(&actual[..5], expected);
    same(&actual[5..], &[sentinel; 2]);
    let mut short = [sentinel; 4];
    assert!(owner.read_into(&mut short).is_err());
    same(&short, &[sentinel; 4]);
}
pub fn verify<T: Encoding>(request: PcuImplementationRequirements, input: [T; 5], sentinel: T) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        numerical_mode: request.numerical_mode,
        numerical_options: request.numerical_options,
        float_underflow: request.float_underflow,
        ..Default::default()
    })
    .unwrap();
    let original = super::super::annotated::identity(&input).unwrap();
    let sibling = super::super::annotated::identity(&original).unwrap();
    for unused in [false, true] {
        let result = if unused {
            super::super::annotated::effect_then_identity(&original)
        } else {
            super::super::annotated::activate(&original)
        };
        if request.float_underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
            assert!(
                matches!(result, Err(PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id==0 && fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow)
            );
        } else {
            let output = result.unwrap();
            let expected = if unused {
                input
            } else {
                input.map(|value| {
                    value
                        .pcu_checked_relu_with_policy(request.float_underflow)
                        .unwrap()
                })
            };
            read(&output, &expected, sentinel);
            let changed = [T::raw(T::NORMAL); 5];
            let next = super::super::annotated::activate(&changed).unwrap();
            drop(next);
            read(&output, &expected, sentinel);
        }
    }
    // Local gradual policy overrides global Tight, preserving every unrelated permission.
    let gradual = local_gradual(&original).unwrap();
    let expected = input.map(|value| {
        value
            .pcu_checked_relu_with_policy(PcuFloatUnderflowPolicy::AllowGradualUnderflow)
            .unwrap()
    });
    read(&gradual, &expected, sentinel);
    let bad = [
        T::raw(T::NORMAL),
        T::raw(T::NAN),
        T::raw(T::NORMAL),
        T::raw(T::NORMAL),
        T::raw(T::NORMAL),
    ];
    let bad_owner = super::super::annotated::identity(&bad).unwrap();
    for result in [
        super::super::annotated::activate(&bad_owner),
        super::super::annotated::effect_then_identity(&bad_owner),
    ] {
        assert!(
            matches!(result, Err(PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered && fault.invocation_id==1 && fault.kind==PcuExecutionFaultKind::InvalidFloatingOperand)
        );
    }
    read(&bad_owner, &bad, sentinel);
    let next =
        consumed(super::super::annotated::identity(&[T::raw(T::NORMAL); 5]).unwrap()).unwrap();
    drop(original);
    global::clear_thread_cache().unwrap();
    read(&next, &[T::raw(T::NORMAL); 5], sentinel);
    read(&sibling, &input, sentinel);
    // Escaped authentic roots compose even after cached code drops.
    let again = local_gradual(&sibling).unwrap();
    drop(sibling);
    read(&again, &expected, sentinel);
}
