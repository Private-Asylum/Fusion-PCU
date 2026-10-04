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
    supported_control_shape(kernel.ops)
        && (kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            != fusion_pcu::PcuReproducibility::PortableV1
            || fusion_pcu::describe_portable_v1_map(kernel).is_ok()
            || fusion_pcu::describe_portable_v1_integer_map(kernel).is_ok()
            || fusion_pcu::describe_portable_v1_integer_div_rem_map(kernel).is_ok()
            || fusion_pcu::describe_portable_v1_unary_map(kernel).is_ok()
            || portable_integer_composed_contract(kernel))
        && checked_ops(kernel.ops)
}

/// Borrowed public IR may contain cycles; inspect only the supported outer body.
pub fn supported_control_shape(ops: &[PcuDispatchOp<'_>]) -> bool {
    let body = match ops {
        [
            PcuDispatchOp::GridStrideLoop { body, .. },
            PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return),
        ] => *body,
        _ => ops,
    };
    !body
        .iter()
        .any(|op| matches!(op, PcuDispatchOp::GridStrideLoop { .. }))
}

/// Independently bounded provider opt-in; neutral eligibility alone grants no execution.
pub fn portable_integer_composed_contract(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    fusion_pcu::describe_portable_v1_checked_integer_composed_map::<4>(kernel).is_ok_and(|schema| {
        matches!(
            schema.value_type,
            PcuValueType::Scalar(
                PcuScalarType::I8
                    | PcuScalarType::U8
                    | PcuScalarType::I16
                    | PcuScalarType::U16
                    | PcuScalarType::I32
                    | PcuScalarType::U32
                    | PcuScalarType::I64
                    | PcuScalarType::U64
                    | PcuScalarType::I128
                    | PcuScalarType::U128
                    | PcuScalarType::I256
                    | PcuScalarType::U256
                    | PcuScalarType::I512
                    | PcuScalarType::U512
            )
        ) && !schema
            .resources()
            .iter()
            .any(|resource| resource.has_cross_index_read_write())
    })
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
