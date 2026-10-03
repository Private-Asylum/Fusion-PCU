use super::*;

#[test]
fn unequal_output_counts_remain_bound_to_their_own_destinations() {
    let q = PcuBindingRef::new(0, 2);
    let r = PcuBindingRef::new(0, 3);
    let mut layout = MlxOutputLayout::default();
    layout.record(r, 17).unwrap();
    layout.record(q, 11).unwrap();
    assert_eq!(layout.count(q), Some(11));
    assert_eq!(layout.count(r), Some(17));
    layout.validate([Some((q, 7)), Some((r, 7))]).unwrap();
    assert!(layout.validate([Some((q, 7)), None]).is_err());
    assert_eq!(MlxOutputLayout::default().count(q), None);
}

#[test]
fn duplicate_or_fifth_mutable_shape_cannot_overwrite_retained_bindings() {
    let q = PcuBindingRef::new(0, 2);
    let r = PcuBindingRef::new(0, 3);
    let mut layout = MlxOutputLayout::default();
    layout.record(q, 11).unwrap();
    assert!(matches!(layout.record(q, 99),
        Err(PcuExecutionError::MlxHostExecution(crate::PcuHostDispatchError::Duplicate(target)))
            if target == q));
    layout.record(r, 17).unwrap();
    layout.record(PcuBindingRef::new(0, 4), 23).unwrap();
    layout.record(PcuBindingRef::new(0, 5), 29).unwrap();
    assert!(matches!(
        layout.record(PcuBindingRef::new(0, 6), 99),
        Err(PcuExecutionError::InvalidTensorSourcePlan)
    ));
    assert_eq!(layout.count(q), Some(11));
    assert_eq!(layout.count(r), Some(17));
}
