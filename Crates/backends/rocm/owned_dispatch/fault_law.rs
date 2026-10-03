//! Cold scalar fault-law capture; bounded structural admission remains separate.

#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedScalarFaultLaw,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuRangePolicy,
    PcuValueType,
};

pub(super) fn capture(kernel: &PcuDispatchKernelIr<'_>) -> Option<PcuCheckedScalarFaultLaw> {
    let ops = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { body, .. },
            PcuDispatchOp::Control(_),
        ] => *body,
        ops => ops,
    };
    let mut law: Option<PcuCheckedScalarFaultLaw> = None;
    for operation in ops {
        let PcuDispatchOp::Data(data) = *operation else {
            continue;
        };
        let range = match data {
            PcuDispatchDataOp::CheckedDivRem { .. } => PcuRangePolicy::Reject,
            PcuDispatchDataOp::CheckedIntegerBinary { range_policy, .. }
            | PcuDispatchDataOp::CheckedFloatBinary { range_policy, .. }
            | PcuDispatchDataOp::CheckedFloatUnary { range_policy, .. }
            | PcuDispatchDataOp::CheckedFloatConvert { range_policy, .. } => range_policy,
            _ => continue,
        };
        // Every emitted range disposition must match the requested contract. The
        // shared packed word can encode recovery, but this is not permission to
        // add recovery to a Reject request. Scoped per-operation UF stays exact.
        if range != kernel.numerical_requirements.range_policy {
            return None;
        }
        let constituent = instruction(data)?;
        law = Some(law.map_or(constituent, |prior| prior.union(constituent)));
    }
    // Existing F32/F64 map admission validates the full body separately. Its
    // lane-only record identifies a fault that may belong to any constituent;
    // this union does not invent an instruction ordinal or a tensor step law.
    law
}

const fn instruction(op: PcuDispatchDataOp) -> Option<PcuCheckedScalarFaultLaw> {
    match op {
        PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(scalar),
            ..
        } => PcuCheckedScalarFaultLaw::integer_div_rem(scalar),
        PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: PcuValueType::Scalar(scalar),
            op,
            range_policy,
            ..
        } => PcuCheckedScalarFaultLaw::integer_binary(scalar, op, range_policy),
        PcuDispatchDataOp::CheckedFloatBinary {
            value_type: PcuValueType::Scalar(scalar),
            op,
            range_policy,
            underflow_policy,
            ..
        } => PcuCheckedScalarFaultLaw::float_binary(scalar, op, range_policy, underflow_policy),
        PcuDispatchDataOp::CheckedFloatUnary {
            value_type: PcuValueType::Scalar(scalar),
            op,
            range_policy,
            underflow_policy,
            ..
        } => PcuCheckedScalarFaultLaw::float_unary(scalar, op, range_policy, underflow_policy),
        PcuDispatchDataOp::CheckedFloatConvert {
            conversion,
            range_policy,
            underflow_policy,
            ..
        } => Some(PcuCheckedScalarFaultLaw::float_conversion(
            conversion,
            range_policy,
            underflow_policy,
        )),
        _ => None,
    }
}

/// Prepared completion metadata; compound steps retain their own semantic domain.
#[derive(Clone, Copy)]
pub(super) enum Retained {
    Scalar(PcuCheckedScalarFaultLaw),
    #[cfg(feature = "tensor")]
    Compound(fusion_pcu::dialect::tensor::TensorStrictFaultDomain),
}
impl Retained {
    pub(super) const fn accepts(self, fault: fusion_pcu::PcuExecutionFault) -> bool {
        match self {
            Self::Scalar(law) => law.allows(fault.kind, fault.recovered),
            #[cfg(feature = "tensor")]
            Self::Compound(domain) => domain.accepts(fault),
        }
    }
}
