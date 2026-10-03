//! Eligibility and per-effect status-law tests; native execution stays a separate proof.
use super::MlxCheckedMapPlan;
#[path = "aggregate/aggregate.rs"]
mod aggregate;
#[cfg(feature = "benchmark-control")]
#[path = "control/control.rs"]
mod control;
#[path = "host/host.rs"]
mod host;
#[path = "mixed/mixed.rs"]
mod mixed;
#[path = "native/native.rs"]
mod native;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedScalarFaultLaw,
    PcuDispatchDataOp,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuNumericalMode,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
};

fn assert_profile(ir: &PcuDispatchKernelIr<'_>, scalar: PcuScalarType, floating: bool) {
    let plan = MlxCheckedMapPlan::assess(ir, scalar).unwrap();
    assert_eq!(plan.requirements(), ir.numerical_requirements);
    assert_eq!(plan.logical_extent(), 7);
    assert_eq!(plan.declared_bindings().len(), 5);
    assert_eq!(plan.resources().len(), 4);
    assert_eq!(plan.checked_effects().len(), 4);
    assert_eq!(plan.status_byte_len(), 7 * 4 * 4);
    assert_eq!(plan.instructions().len(), 10);
    let (header, native) = plan.native_source().unwrap();
    assert!(header.contains("struct") || header.contains("pcu_integer"));
    assert!(!native.contains("kernel void"));
    assert!(!native.contains("profile.x"));
    for effect in 0..4 {
        assert!(native.contains(&format!("records[{effect}u*7u+id]=status")));
    }
    assert_eq!(ir.entry.logical_shape[0], if floating { 3 } else { 7 });
    let stage = plan
        .resources()
        .iter()
        .find(|role| role.minimum_write_elements == 7 && role.minimum_read_elements == 7)
        .unwrap();
    assert_eq!(stage.minimum_initial_read_elements, 0);
    assert!(
        plan.checked_effects()
            .windows(2)
            .all(|pair| pair[0].instruction < pair[1].instruction)
    );
    let mut strict = *ir;
    strict.numerical_requirements.numerical_mode = PcuNumericalMode::Strict;
    assert_eq!(
        MlxCheckedMapPlan::assess(&strict, scalar)
            .unwrap()
            .requirements(),
        strict.numerical_requirements
    );
    strict
        .numerical_requirements
        .numerical_options
        .reproducibility = PcuReproducibility::PortableV1;
    assert!(MlxCheckedMapPlan::assess(&strict, scalar).is_err());
}

#[test]
fn all_ten_integer_and_two_float_sources_keep_dead_effects_original_roles_and_grid_extent() {
    macro_rules! integer { ($($ty:ty),*) => { $(
        source::integer_ir::<$ty, 7>(&source::integer_bindings::<$ty>()).unwrap()
            .with_ir(|ir| assert_profile(ir, <$ty as pcu_facade::PcuScalar>::TYPE, false));
    )* }; }
    integer!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
    source::floating_ir::<f32, 7>(&source::floating_bindings::<f32>())
        .unwrap()
        .with_ir(|ir| assert_profile(ir, PcuScalarType::F32, true));
    source::floating_ir::<f64, 7>(&source::floating_bindings::<f64>())
        .unwrap()
        .with_ir(|ir| assert_profile(ir, PcuScalarType::F64, true));
}

#[test]
fn each_local_policy_has_its_own_fault_law_before_arbitration() {
    source::integer_ir::<i32, 7>(&source::integer_bindings::<i32>())
        .unwrap()
        .with_ir(|ir| {
            let mut ops = ir.ops.to_vec();
            let mut changed = false;
            for instruction in &mut ops {
                if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                    range_policy,
                    ..
                }) = instruction
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
            assert_eq!(plan.requirements().range_policy, PcuRangePolicy::Reject);
            let mut fault = PcuExecutionFault {
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                invocation_id: 6,
                recovered: true,
            };
            assert!(!plan.accepts_record(0, fault));
            assert!(plan.accepts_record(1, fault));
            let mut words = vec![0; plan.status_byte_len() / 4];
            words[7] = 0x101;
            words[6] = 1;
            let selected = plan.validate_status_words(&words).unwrap().unwrap();
            assert_eq!(selected.invocation_id, 6);
            assert!(!selected.recovered);
            words[21] = 4;
            assert!(plan.validate_status_words(&words).is_err());
            words[21] = 0xffff_ffff;
            assert!(plan.validate_status_words(&words).is_err());
            assert!(
                plan.validate_status_words(&words[..words.len() - 1])
                    .is_err()
            );

            fault.invocation_id = 7;
            assert!(!plan.accepts_record(1, fault));
            fault.invocation_id = 0;
            fault.kind = PcuExecutionFaultKind::DivideByZero;
            assert!(!plan.accepts_record(1, fault));
            assert!(!plan.accepts_record(4, fault));
            assert_eq!(
                plan.checked_effects()[1].law,
                PcuCheckedScalarFaultLaw::integer_binary(
                    PcuScalarType::I32,
                    PcuDispatchIntegerBinaryOp::Add,
                    PcuRangePolicy::Clamp
                )
                .unwrap()
            );
        });
}
