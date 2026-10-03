use super::*;

#[test]
fn shapes_belong_to_bindings_and_traversal_order_does_not_specialize_again() {
    let a = PcuBindingRef::new(0, 0);
    let b = PcuBindingRef::new(0, 1);
    let mut first = MlxInputLayout::default();
    first.record(a, 8).unwrap();
    first.record(b, 17).unwrap();
    let mut reversed = MlxInputLayout::default();
    reversed.record(b, 17).unwrap();
    reversed.record(a, 8).unwrap();
    assert!(first == reversed);
    assert_eq!(first.extents(&[b, a], [1, 4]).unwrap(), [17, 8]);
}

#[test]
fn host_inputs_keep_minimum_spans_and_resident_prefixes_keep_full_shapes() {
    let a = PcuBindingRef::new(0, 0);
    let b = PcuBindingRef::new(0, 1);
    let mut layout = MlxInputLayout::default();
    layout.record(a, 8).unwrap();
    assert_eq!(layout.extents(&[a, b], [4, 1]).unwrap(), [8, 1]);
    assert_eq!(layout.extents(&[a], [1, 0]).unwrap(), [8, 0]);
    assert!(layout.extents(&[a], [9, 0]).is_err());
    assert!(layout.extents(&[b], [1, 0]).is_err());
    assert_eq!(
        MlxInputLayout::default().extents(&[a], [4, 0]).unwrap(),
        [4, 0]
    );
}

#[test]
fn invalid_bindings_cannot_overwrite_cold_shape_metadata() {
    let a = PcuBindingRef::new(0, 0);
    let b = PcuBindingRef::new(0, 1);
    let c = PcuBindingRef::new(0, 2);
    let mut layout = MlxInputLayout::default();
    layout.record(a, 8).unwrap();
    assert!(layout.record(a, 99).is_err());
    layout.record(b, 17).unwrap();
    layout.record(c, 99).unwrap();
    layout.record(PcuBindingRef::new(0, 3), 101).unwrap();
    assert!(layout.record(PcuBindingRef::new(0, 4), 103).is_err());
    assert!(layout.extents(&[a, b], [4, 1]).is_err());
    assert_eq!(layout.snapshot_extents(&[a, b], [4, 1]).unwrap(), [8, 17]);
}

#[test]
fn mutable_initial_snapshots_are_distinct_from_readonly_arithmetic_operands() {
    let read = PcuBindingRef::new(0, 0);
    let old_writer = PcuBindingRef::new(0, 1);
    let overwrite = PcuBindingRef::new(0, 2);
    let mut layout = MlxInputLayout::default();
    layout.record_mutable(overwrite, 29).unwrap();
    layout.record(read, 8).unwrap();
    layout.record_mutable(old_writer, 17).unwrap();
    assert_eq!(layout.extents(&[read], [4, 0]).unwrap(), [8, 0]);
    assert_eq!(
        layout
            .snapshot_extents(&[old_writer, read], [4, 1, 0, 0])
            .unwrap(),
        [17, 8, 0, 0]
    );
    assert!(layout.snapshot_extents(&[old_writer], [18]).is_err());
    assert!(layout.snapshot_extents(&[read, old_writer], [4]).is_err());
    assert!(layout.record(old_writer, 101).is_err());
    let mut reversed = MlxInputLayout::default();
    reversed.record_mutable(old_writer, 17).unwrap();
    reversed.record(read, 8).unwrap();
    reversed.record_mutable(overwrite, 29).unwrap();
    assert!(layout == reversed);
}

#[test]
fn four_native_initial_shapes_remain_keyed_to_original_bindings() {
    let bindings = core::array::from_fn::<_, 4, _>(|index| {
        PcuBindingRef::new(2, u32::try_from(index).unwrap())
    });
    let mut layout = MlxInputLayout::default();
    layout.record(bindings[3], 31).unwrap();
    layout.record_mutable(bindings[1], 17).unwrap();
    layout.record(bindings[0], 11).unwrap();
    layout.record_mutable(bindings[2], 23).unwrap();
    assert_eq!(
        layout
            .snapshot_extents(
                &[bindings[2], bindings[0], bindings[3], bindings[1]],
                [1; 4]
            )
            .unwrap(),
        [23, 11, 31, 17]
    );
}

#[test]
fn recording_mutable_shape_does_not_change_old_arithmetic_readonly_projection() {
    let a = PcuBindingRef::new(0, 0);
    let b = PcuBindingRef::new(0, 1);
    let mut layout = MlxInputLayout::default();
    layout.record(a, 8).unwrap();
    layout.record(b, 17).unwrap();
    layout.record_mutable(PcuBindingRef::new(0, 2), 31).unwrap();
    assert_eq!(layout.extents(&[a, b], [4, 1]).unwrap(), [8, 17]);
}
