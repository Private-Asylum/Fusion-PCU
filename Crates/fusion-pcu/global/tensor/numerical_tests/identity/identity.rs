//! Numerical scopes describe captured arithmetic; identity does not rewrite its producer.

#[rustfmt::skip]
use super::super::{
    PcuExecutionError,
    PcuSourceShape,
    PcuTensorGraphCapture,
};
#[rustfmt::skip]
use crate::{
    PcuCheckedInteger,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuReproducibility,
};

#[crate::pcu(crate_path = crate, flag(deterministic), flag(strict))]
fn portable_add<T: PcuCheckedInteger>(
    input: &[T],
) -> Result<crate::PcuTensor<T>, PcuExecutionError> {
    pcu::add(input, input)
}

#[crate::pcu(crate_path = crate)]
fn inherited_add<T: PcuCheckedInteger>(
    input: &[T],
) -> Result<crate::PcuTensor<T>, PcuExecutionError> {
    pcu::add(input, input)
}

#[crate::pcu(crate_path = crate, flag(deterministic), flag(strict))]
fn portable_identity<T: crate::PcuScalar>(
    input: &[T],
) -> Result<crate::PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

fn check<T: PcuCheckedInteger>() {
    let (mut capture, inputs) =
        PcuTensorGraphCapture::new::<T, 1>([PcuSourceShape::Slice { length: 5 }]).unwrap();
    let defaults = PcuNumericalOptions::default();
    let output = portable_add::__pcu_capture_entry::<T>(&mut capture, inputs).unwrap();
    let node = capture.graph.node(output.value.erase()).unwrap();
    assert_eq!(node.scalar_type, T::TYPE);
    assert_eq!(
        node.numerical_options.reproducibility,
        PcuReproducibility::PortableV1
    );
    // One integer operation already checks its entire domain; compound mode is inapplicable.
    assert_eq!(node.numerical_mode, None);
    assert_eq!(capture.numerical_options.get(), defaults);
    assert_eq!(capture.numerical_mode.get(), PcuNumericalMode::Boundary);

    let ordinary = inherited_add::__pcu_capture_entry::<T>(&mut capture, inputs).unwrap();
    let original = capture.graph.node(ordinary.value.erase()).unwrap();
    assert_eq!(original.numerical_options, defaults);
    assert_eq!(original.numerical_mode, None);
    let identity = portable_identity::__pcu_capture_entry::<T>(&mut capture, [ordinary]).unwrap();
    assert_eq!(identity.value.erase(), ordinary.value.erase());
    let retained = capture.graph.node(identity.value.erase()).unwrap();
    assert_eq!(retained.numerical_options, defaults);
    assert_eq!(retained.numerical_mode, None);
    assert_eq!(capture.numerical_options.get(), defaults);
    assert_eq!(capture.numerical_mode.get(), PcuNumericalMode::Boundary);

    let requested = PcuNumericalOptions {
        reproducibility: PcuReproducibility::PortableV1,
        ..defaults
    };
    capture.numerical_options.set(requested);
    capture.numerical_mode.set(PcuNumericalMode::Strict);
    let inherited = inherited_add::__pcu_capture_entry::<T>(&mut capture, inputs).unwrap();
    let node = capture.graph.node(inherited.value.erase()).unwrap();
    assert_eq!(node.numerical_options, requested);
    // One integer operation already checks its entire domain; compound mode is inapplicable.
    assert_eq!(node.numerical_mode, None);
    assert_eq!(capture.numerical_options.get(), requested);
    assert_eq!(capture.numerical_mode.get(), PcuNumericalMode::Strict);
}

#[test]
fn all_fourteen_owned_integer_types_capture_local_and_inherited_arithmetic_policies() {
    check::<i8>();
    check::<u8>();
    check::<i16>();
    check::<u16>();
    check::<i32>();
    check::<u32>();
    check::<i64>();
    check::<u64>();
    check::<i128>();
    check::<u128>();
    check::<crate::PcuI256>();
    check::<crate::PcuU256>();
    check::<crate::PcuI512>();
    check::<crate::PcuU512>();
}
