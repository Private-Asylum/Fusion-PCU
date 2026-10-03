//! Checked-default numerical admission independent of compiler lowering.

#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuScalarType,
    PcuValueType,
};

/// Legacy value ALU opcodes have no explicit permission to suppress numerical faults.
pub fn checked_numeric_contract(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    (kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != fusion_pcu::PcuReproducibility::PortableV1
        || fusion_pcu::describe_portable_v1_map(kernel).is_ok()
        || fusion_pcu::describe_portable_v1_integer_map(kernel).is_ok()
        || fusion_pcu::describe_portable_v1_integer_div_rem_map(kernel).is_ok()
        || fusion_pcu::describe_portable_v1_unary_map(kernel).is_ok())
        && checked_ops(kernel.ops)
}

fn checked_ops(ops: &[PcuDispatchOp<'_>]) -> bool {
    ops.iter().all(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::Alu { .. }) => false,
        PcuDispatchOp::Data(PcuDispatchDataOp::Convert { conversion, .. }) => !matches!(
            conversion.source_type(),
            PcuValueType::Scalar(
                PcuScalarType::F16
                    | PcuScalarType::BF16
                    | PcuScalarType::F8E4M3FN
                    | PcuScalarType::F8E5M2
                    | PcuScalarType::F32
                    | PcuScalarType::F64
            )
        ),
        PcuDispatchOp::GridStrideLoop { body, .. } => checked_ops(body),
        _ => true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuDispatchAluOp,
        PcuDispatchFloatBinaryOp,
        PcuDispatchValueId,
        PcuFloatUnderflowPolicy,
        PcuRangePolicy,
    };

    #[test]
    fn checked_operations_do_not_legalize_legacy_float_alu_in_direct_or_loop_ir() {
        let checked = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type: PcuValueType::f32(),
            op: PcuDispatchFloatBinaryOp::Add,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        });
        for value_type in [PcuValueType::f32(), PcuValueType::f64()] {
            for op in [
                PcuDispatchAluOp::Add,
                PcuDispatchAluOp::Sub,
                PcuDispatchAluOp::Mul,
                PcuDispatchAluOp::Div,
                PcuDispatchAluOp::Min,
                PcuDispatchAluOp::Max,
            ] {
                let legacy = PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    value_type,
                    op,
                    result: PcuDispatchValueId(4),
                    lhs: PcuDispatchValueId(1),
                    rhs: PcuDispatchValueId(2),
                });
                assert!(!checked_ops(&[legacy]));
                assert!(!checked_ops(&[checked, legacy]));
                assert!(!checked_ops(&[PcuDispatchOp::GridStrideLoop {
                    extent: 8,
                    body: &[checked, legacy]
                }]));
            }
        }
        assert!(checked_ops(&[checked]));
        let coordinate = PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            value_type: PcuValueType::Scalar(PcuScalarType::U32),
            op: PcuDispatchAluOp::Add,
            result: PcuDispatchValueId(4),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        });
        assert!(!checked_ops(&[coordinate, checked]));
    }
}
