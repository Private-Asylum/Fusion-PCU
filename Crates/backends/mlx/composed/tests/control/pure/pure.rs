//! Cold controls require the exact workload and preserve the original metadata.
use crate::MlxCheckedMapPlan;
use super::source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchDataOp,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuRangePolicy,
    PcuScalarType,
};
fn verify(ir: &PcuDispatchKernelIr<'_>, scalar: PcuScalarType) {
    let plan = MlxCheckedMapPlan::assess(ir, scalar).unwrap();
    let (shared_header, handwritten) = crate::composed::control::sources(&plan).unwrap();
    let (lowered_header, lowered_body) = plan.native_source().unwrap();
    assert_eq!(shared_header, lowered_header);
    assert_ne!(handwritten, lowered_body);
    assert!(handwritten.contains("original"));
    assert!(handwritten.contains("final_value"));
    assert!(!handwritten.contains("v0"));
    assert_eq!(plan.requirements(), ir.numerical_requirements);
    assert_eq!(plan.logical_extent(), 7);
    assert_eq!(plan.resources().len(), 4);
    assert_eq!(plan.declared_bindings().len(), 5);
    for token in [
        "COUNT",
        "CLAMP",
        "FIRST",
        "DEAD",
        "PRODUCT",
        "FINAL",
        "LOADS",
        "STORES",
        "STATUS",
        "NONFINITE",
    ] {
        assert!(!handwritten.contains(token), "unexpanded {token}");
    }
}
#[test]
fn all_twelve_controls_are_distinct_bodies_with_exact_original_metadata() {
    macro_rules! integer { ($($ty:ty),*) => { $(
        source::integer_ir::<$ty, 7>(&source::integer_bindings::<$ty>()).unwrap().with_ir(|ir|
            verify(ir, <$ty as pcu_facade::PcuScalar>::TYPE));
    )* }; }
    integer!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
    macro_rules! floating { ($($ty:ty),*) => { $(
        source::floating_ir::<$ty, 7>(&source::floating_bindings::<$ty>()).unwrap().with_ir(|ir|
            verify(ir, <$ty as pcu_facade::PcuScalar>::TYPE));
    )* }; }
    floating!(f32, f64);
}
#[test]
fn local_policy_is_retained_and_a_different_valid_workload_is_refused() {
    source::integer_ir::<i32, 7>(&source::integer_bindings::<i32>())
        .unwrap()
        .with_ir(|ir| {
            let mut ops = ir.ops.to_vec();
            let mut changed = false;
            for op in &mut ops {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                    range_policy,
                    ..
                }) = op
                {
                    if changed {
                        *range_policy = PcuRangePolicy::Clamp;
                        break;
                    }
                    changed = true;
                }
            }
            let scoped = PcuDispatchKernelIr { ops: &ops, ..*ir };
            let plan = MlxCheckedMapPlan::assess(&scoped, PcuScalarType::I32).unwrap();
            let (_, body) = crate::composed::control::sources(&plan).unwrap();
            assert_eq!(plan.requirements().range_policy, PcuRangePolicy::Reject);
            assert!(body.contains("pcu_integer(first,seed,dead,0u,true)"));
            for op in &mut ops {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { op, .. }) = op
                {
                    *op = PcuDispatchIntegerBinaryOp::Sub;
                    break;
                }
            }
            let different = PcuDispatchKernelIr { ops: &ops, ..*ir };
            let plan = MlxCheckedMapPlan::assess(&different, PcuScalarType::I32).unwrap();
            assert!(crate::composed::control::sources(&plan).is_err());
        });
}
