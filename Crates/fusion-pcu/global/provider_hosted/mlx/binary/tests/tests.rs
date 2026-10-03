//! Binding preflight is independent of a loaded MLX image or live native session.

use super::*;
#[rustfmt::skip]
use crate::{
    PcuF16Bits,
    PcuReadStorage,
    PcuScalarType,
    PcuWriteStorage,
    SliceShape,
};

fn read(values: &[PcuF16Bits], target: PcuBindingRef) -> PcuCallArgument<'_> {
    <[PcuF16Bits] as PcuReadStorage<PcuF16Bits, SliceShape>>::as_pcu_call_argument(values, target)
        .unwrap()
}
fn write(values: &mut [PcuF16Bits], target: PcuBindingRef) -> PcuCallArgument<'_> {
    <[PcuF16Bits] as PcuWriteStorage<PcuF16Bits, SliceShape>>::as_pcu_call_argument(values, target)
        .unwrap()
}

#[test]
fn unused_empty_input_has_no_physical_role_and_output_order_is_independent() {
    let bindings = [PcuBindingRef::new(0, 7), PcuBindingRef::new(0, 11)];
    let out = PcuBindingRef::new(0, 19);
    let values = [PcuF16Bits::from_bits(0x3c00); 5];
    let mut output = [PcuF16Bits::from_bits(0x4000); 7];
    let (inputs, output) = collect(
        &bindings[..1],
        &bindings,
        out,
        PcuScalarType::F16,
        [
            write(&mut output, out),
            read(&[], bindings[1]),
            read(&values, bindings[0]),
        ],
    )
    .unwrap();
    let Some(PcuCallArgumentKind::Host(input)) = &inputs[0] else {
        panic!("missing used input")
    };
    assert_eq!(input.target(), bindings[0]);
    assert_eq!(input.bytes().len(), 10);
    assert!(inputs[1].is_none());
    let PcuCallArgumentKind::Host(output) = output else {
        panic!("missing output")
    };
    assert_eq!(output.target(), out);
    assert_eq!(output.bytes().len(), 14);
}

#[test]
fn duplicate_or_unknown_binding_fails_before_output_mutation() {
    let input = PcuBindingRef::new(0, 1);
    let out = PcuBindingRef::new(0, 2);
    let values = [PcuF16Bits::from_bits(0x3c00); 5];
    let sentinel = PcuF16Bits::from_bits(0x4000);
    let mut output = [sentinel; 7];
    let result = collect(
        &[input],
        &[input, PcuBindingRef::new(0, 3)],
        out,
        PcuScalarType::F16,
        [
            read(&values, input),
            write(&mut output, out),
            read(&values, input),
        ],
    )
    .err()
    .expect("duplicate binding must fail");
    assert!(
        matches!(result, PcuExecutionError::MlxHostExecution(PcuHostDispatchError::Duplicate(target)) if target == input)
    );
    assert_eq!(output, [sentinel; 7]);
    let result = collect(
        &[input],
        &[input],
        out,
        PcuScalarType::F16,
        [
            write(&mut output, out),
            read(&values, PcuBindingRef::new(0, 9)),
        ],
    )
    .err()
    .expect("unknown binding must fail");
    assert!(
        matches!(result, PcuExecutionError::MlxHostExecution(PcuHostDispatchError::Unexpected(target)) if target == PcuBindingRef::new(0, 9))
    );
    assert_eq!(output, [sentinel; 7]);
}

#[test]
fn unused_declarations_still_require_their_readonly_type_role() {
    let bindings = [PcuBindingRef::new(0, 1), PcuBindingRef::new(0, 2)];
    let out = PcuBindingRef::new(0, 3);
    let values = [PcuF16Bits::from_bits(0x3c00); 5];
    let mut unused = [];
    let mut output = [PcuF16Bits::from_bits(0x4000); 7];
    let result = collect(
        &bindings[..1],
        &bindings,
        out,
        PcuScalarType::F16,
        [
            read(&values, bindings[0]),
            write(&mut unused, bindings[1]),
            write(&mut output, out),
        ],
    );
    assert!(
        matches!(result, Err(PcuExecutionError::MlxHostExecution(PcuHostDispatchError::AccessMismatch(target))) if target == bindings[1])
    );
}

fn integer_roles<T: crate::PcuScalar>(value: T) {
    let input = PcuBindingRef::new(0, 9);
    let unused = PcuBindingRef::new(0, 2);
    let out = PcuBindingRef::new(0, 7);
    let values = [value; 7];
    let mut destination = [value; 9];
    let read =
        <[T] as PcuReadStorage<T, SliceShape>>::as_pcu_call_argument(&values, input).unwrap();
    let write =
        <[T] as PcuWriteStorage<T, SliceShape>>::as_pcu_call_argument(&mut destination, out)
            .unwrap();
    let (inputs, output) = collect(
        &[input],
        &[unused, input],
        out,
        T::TYPE,
        [write, PcuCallArgument::unused_read::<T>(unused), read],
    )
    .unwrap();
    let Some(PcuCallArgumentKind::Host(input)) = &inputs[0] else {
        panic!("missing unique integer input")
    };
    assert_eq!(input.scalar(), T::TYPE);
    assert_eq!(input.bytes().len(), 7 * T::ENCODED_SIZE);
    assert!(inputs[1].is_none());
    let PcuCallArgumentKind::Host(output) = output else {
        panic!("missing integer output")
    };
    assert_eq!(output.scalar(), T::TYPE);
    assert_eq!(output.bytes().len(), 9 * T::ENCODED_SIZE);
}

#[test]
fn narrow_and_wide_integer_roles_share_the_stack_only_publication_preflight() {
    integer_roles(2_u8);
    integer_roles(2_i128);
    integer_roles(crate::PcuU256::from_limbs_le([2, 0, 0, 0]));
    integer_roles(crate::PcuI512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]));
}
