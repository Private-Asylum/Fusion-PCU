//! Binding projection needs neither an MLX library image nor a native session.
use super::*;
#[rustfmt::skip]
use crate::{
    global::arguments::{
        PcuCallArgument,
        PcuCallArgumentKind,
    },
    PcuExecutionError,
    PcuF16Bits,
    PcuHostDispatchError,
    PcuReadStorage,
    PcuWriteStorage,
    SliceShape,
};

fn read<T: crate::PcuScalar>(values: &[T], target: PcuBindingRef) -> PcuCallArgument<'_> {
    <[T] as PcuReadStorage<T, SliceShape>>::as_pcu_call_argument(values, target).unwrap()
}
fn write<T: crate::PcuScalar>(values: &mut [T], target: PcuBindingRef) -> PcuCallArgument<'_> {
    <[T] as PcuWriteStorage<T, SliceShape>>::as_pcu_call_argument(values, target).unwrap()
}

#[test]
fn reordered_unused_declarations_are_metadata_without_physical_roles() {
    let input = PcuBindingRef::new(7, 11);
    let output = PcuBindingRef::new(3, 19);
    let unused = [PcuBindingRef::new(2, 5), PcuBindingRef::new(4, 9)];
    let values = [PcuF16Bits::from_bits(0x3c00); 5];
    let mut destination = [PcuF16Bits::from_bits(0x4000); 7];
    {
        let (source, target) = collect(
            input,
            output,
            PcuScalarType::F16,
            &unused,
            [
                write(&mut destination, output),
                PcuCallArgument::unused_read::<PcuF16Bits>(unused[1]),
                read(&values, input),
                read::<PcuF16Bits>(&[], unused[0]),
            ],
        )
        .unwrap();
        let PcuCallArgumentKind::Host(source) = source else {
            panic!("missing source")
        };
        let PcuCallArgumentKind::Host(target) = target else {
            panic!("missing output")
        };
        assert_eq!(source.target(), input);
        assert_eq!(source.bytes().len(), 10);
        assert_eq!(target.target(), output);
        assert_eq!(target.bytes().len(), 14);
    }
    assert_eq!(destination, [PcuF16Bits::from_bits(0x4000); 7]);
}

#[test]
fn declared_unused_inputs_still_require_the_correct_scalar_and_access() {
    let input = PcuBindingRef::new(0, 1);
    let output = PcuBindingRef::new(0, 2);
    let unused = PcuBindingRef::new(0, 3);
    let values = [1.0_f32; 5];
    let mut destination = [91.0_f32; 7];
    let error = collect(
        input,
        output,
        PcuScalarType::F32,
        &[unused],
        [
            read(&values, input),
            write(&mut destination, output),
            PcuCallArgument::unused_read::<f64>(unused),
        ],
    )
    .err()
    .unwrap();
    assert!(
        matches!(error, PcuExecutionError::MlxHostExecution(PcuHostDispatchError::TypeMismatch(target)) if target == unused)
    );
    let error = collect(
        input,
        output,
        PcuScalarType::F32,
        &[unused],
        [
            read(&values, input),
            write::<f32>(&mut [], unused),
            write(&mut destination, output),
        ],
    )
    .err()
    .unwrap();
    assert!(
        matches!(error, PcuExecutionError::MlxHostExecution(PcuHostDispatchError::AccessMismatch(target)) if target == unused)
    );
    assert_eq!(destination.map(f32::to_bits), [91.0_f32.to_bits(); 7]);
}

#[test]
fn duplicate_unknown_and_omitted_declarations_fail_without_publication() {
    let input = PcuBindingRef::new(0, 1);
    let output = PcuBindingRef::new(0, 2);
    let unused = PcuBindingRef::new(0, 3);
    let values = [1.0_f32; 5];
    let mut destination = [91.0_f32; 7];
    let error = collect(
        input,
        output,
        PcuScalarType::F32,
        &[unused],
        [
            read(&values, input),
            write(&mut destination, output),
            read(&values, input),
        ],
    )
    .err()
    .unwrap();
    assert!(
        matches!(error, PcuExecutionError::MlxHostExecution(PcuHostDispatchError::Duplicate(target)) if target == input)
    );
    let unknown = PcuBindingRef::new(0, 4);
    let error = collect(
        input,
        output,
        PcuScalarType::F32,
        &[unused],
        [
            read(&values, input),
            write(&mut destination, output),
            PcuCallArgument::unused_read::<f32>(unknown),
        ],
    )
    .err()
    .unwrap();
    assert!(
        matches!(error, PcuExecutionError::MlxHostExecution(PcuHostDispatchError::Unexpected(target)) if target == unknown)
    );
    let error = collect(
        input,
        output,
        PcuScalarType::F32,
        &[unused],
        [read(&values, input), write(&mut destination, output)],
    )
    .err()
    .unwrap();
    assert!(matches!(error, PcuExecutionError::InvalidTensorSourcePlan));
    assert_eq!(destination.map(f32::to_bits), [91.0_f32.to_bits(); 7]);
}

#[test]
fn exact_two_role_carrier_projection_keeps_zero_extent_and_reordered_bindings() {
    let input = PcuBindingRef::new(2, 11);
    let output = PcuBindingRef::new(0, 1);
    let (source, target) = collect(
        input,
        output,
        PcuScalarType::F64,
        &[],
        [write::<f64>(&mut [], output), read::<f64>(&[], input)],
    )
    .unwrap();
    let PcuCallArgumentKind::Host(source) = source else {
        panic!("missing source")
    };
    let PcuCallArgumentKind::Host(target) = target else {
        panic!("missing output")
    };
    assert_eq!(source.target(), input);
    assert_eq!(target.target(), output);
    assert!(source.bytes().is_empty());
    assert!(target.bytes().is_empty());
}
