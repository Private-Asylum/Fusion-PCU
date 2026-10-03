//! Joint binding preflight has no native MLX dependency.
use super::*;
#[path = "native/native.rs"]
mod native;
#[rustfmt::skip]
use crate::{
    PcuReadStorage,
    PcuWriteStorage,
    SliceShape,
};

fn read(values: &[i64], target: PcuBindingRef) -> PcuCallArgument<'_> {
    <[i64] as PcuReadStorage<i64, SliceShape>>::as_pcu_call_argument(values, target).unwrap()
}
fn write(values: &mut [i64], target: PcuBindingRef) -> PcuCallArgument<'_> {
    <[i64] as PcuWriteStorage<i64, SliceShape>>::as_pcu_call_argument(values, target).unwrap()
}

#[test]
fn all_four_bindings_reorder_without_conflating_output_extents() {
    let refs = [0, 1, 2, 3].map(|binding| PcuBindingRef::new(7, binding));
    let left = [17_i64; 4];
    let right = [3_i64; 4];
    let mut q = [91_i64; 6];
    let mut r = [92_i64; 9];
    {
        let (inputs, outputs) = collect(
            &[refs[0], refs[1]],
            &[refs[0], refs[1]],
            [refs[2], refs[3]],
            PcuScalarType::I64,
            [
                write(&mut r, refs[3]),
                read(&right, refs[1]),
                write(&mut q, refs[2]),
                read(&left, refs[0]),
            ],
        )
        .unwrap();
        for (slot, input) in inputs.iter().enumerate() {
            let Some(PcuCallArgumentKind::Host(input)) = input else {
                panic!("host input")
            };
            assert_eq!(input.target(), refs[slot]);
            assert_eq!(input.bytes().len(), 4 * size_of::<i64>());
        }
        for (slot, output) in outputs.iter().enumerate() {
            let PcuCallArgumentKind::Host(output) = output else {
                panic!("host output")
            };
            assert_eq!(output.target(), refs[slot + 2]);
            assert_eq!(output.bytes().len(), [6, 9][slot] * size_of::<i64>());
        }
    }
    assert_eq!(q, [91; 6]);
    assert_eq!(r, [92; 9]);
}

#[test]
fn bad_second_output_rejects_before_either_destination_changes() {
    let refs = [0, 1, 2, 3].map(|binding| PcuBindingRef::new(0, binding));
    let values = [17_i64; 4];
    let mut q = [91_i64; 6];
    let mut r = [92_i64; 3];
    {
        let (_, mut outputs) = collect(
            &[refs[0], refs[1]],
            &[refs[0], refs[1]],
            [refs[2], refs[3]],
            PcuScalarType::I64,
            [
                read(&values, refs[0]),
                read(&values, refs[1]),
                write(&mut q, refs[2]),
                write(&mut r, refs[3]),
            ],
        )
        .unwrap();
        {
            let [first, second] = &mut outputs;
            let _destination =
                Destination::prepare(first, PcuScalarType::I64, 32, refs[2], None).unwrap();
            assert!(
                matches!(Destination::prepare(second, PcuScalarType::I64, 32, refs[3], None),
        Err(PcuExecutionError::MlxHostExecution(PcuHostDispatchError::BufferTooSmall(target)))
            if target == refs[3])
            );
        }
    }
    assert_eq!(q, [91; 6]);
    assert_eq!(r, [92; 3]);
}

#[test]
fn duplicate_or_missing_fourth_binding_never_publishes_a_partial_pair() {
    let refs = [0, 1, 2, 3].map(|binding| PcuBindingRef::new(0, binding));
    let values = [17_i64; 4];
    let mut q = [91_i64; 6];
    let error = collect(
        &[refs[0], refs[1]],
        &[refs[0], refs[1]],
        [refs[2], refs[3]],
        PcuScalarType::I64,
        [
            read(&values, refs[0]),
            read(&values, refs[1]),
            write(&mut q, refs[2]),
            read(&values, refs[0]),
        ],
    )
    .err()
    .expect("duplicate must reject");
    assert!(matches!(error,
        PcuExecutionError::MlxHostExecution(PcuHostDispatchError::Duplicate(target))
            if target == refs[0]));
    assert_eq!(q, [91; 6]);
}

#[test]
fn one_actual_read_retains_both_distinct_publication_destinations() {
    let refs = [0, 2, 3].map(|binding| PcuBindingRef::new(7, binding));
    let values = [17_i64; 4];
    let mut q = [91_i64; 6];
    let mut r = [92_i64; 9];
    let (inputs, outputs) = collect(
        &[refs[0]],
        &[refs[0]],
        [refs[1], refs[2]],
        PcuScalarType::I64,
        [
            write(&mut r, refs[2]),
            read(&values, refs[0]),
            write(&mut q, refs[1]),
        ],
    )
    .unwrap();
    assert!(
        matches!(&inputs[0], Some(PcuCallArgumentKind::Host(input)) if input.target() == refs[0])
    );
    assert!(inputs[1].is_none());
    for (slot, output) in outputs.iter().enumerate() {
        assert!(
            matches!(output, PcuCallArgumentKind::Host(output) if output.target() == refs[slot + 1])
        );
    }
}

#[test]
fn unread_source_declaration_requires_no_fabricated_input() {
    let refs = [0, 1, 2, 3].map(|binding| PcuBindingRef::new(0, binding));
    let values = [17_i64; 4];
    let mut q = [91_i64; 6];
    let mut r = [92_i64; 9];
    let (inputs, _) = collect(
        &[refs[1]],
        &[refs[0], refs[1]],
        [refs[2], refs[3]],
        PcuScalarType::I64,
        [
            PcuCallArgument::unused_read::<i64>(refs[0]),
            write(&mut r, refs[3]),
            read(&values, refs[1]),
            write(&mut q, refs[2]),
        ],
    )
    .unwrap();
    assert!(
        matches!(&inputs[0], Some(PcuCallArgumentKind::Host(input)) if input.target() == refs[1])
    );
    assert!(inputs[1].is_none());
}

#[test]
fn unread_declaration_still_rejects_the_wrong_scalar_type() {
    let refs = [0, 1, 2, 3].map(|binding| PcuBindingRef::new(0, binding));
    let values = [17_i64; 4];
    let mut q = [91_i64; 6];
    let mut r = [92_i64; 9];
    assert!(
        collect(
            &[refs[1]],
            &[refs[0], refs[1]],
            [refs[2], refs[3]],
            PcuScalarType::I64,
            [
                PcuCallArgument::unused_read::<u64>(refs[0]),
                read(&values, refs[1]),
                write(&mut q, refs[2]),
                write(&mut r, refs[3])
            ],
        )
        .is_err()
    );
    assert_eq!(q, [91; 6]);
    assert_eq!(r, [92; 9]);
}

#[test]
fn missing_second_destination_after_a_single_read_never_publishes() {
    let refs = [0, 1, 2, 3].map(|binding| PcuBindingRef::new(0, binding));
    let values = [17_i64; 4];
    let mut q = [91_i64; 6];
    assert!(matches!(collect(
        &[refs[0]], &[refs[0], refs[1]], [refs[2], refs[3]], PcuScalarType::I64,
        [read(&values, refs[0]), PcuCallArgument::unused_read::<i64>(refs[1]),
         write(&mut q, refs[2]), read(&values, refs[0])],
    ), Err(PcuExecutionError::MlxHostExecution(PcuHostDispatchError::Duplicate(target))) if target == refs[0]));
    assert_eq!(q, [91; 6]);
}
