//! Explicit native F32 optimizer policy; ordinary scalar arithmetic remains checked.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalRequirement,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    TensorUnsupportedReason,
};

pub(super) fn assess_graph(
    graph: &Graph,
    node: NodeDescriptor<'_>,
) -> Result<(), TensorUnsupportedReason> {
    let OpDescriptor::SgdUpdate { learning_rate, .. } = node.op else {
        return Err(TensorUnsupportedReason::Operation);
    };
    let original = graph
        .node(node.value)
        .map_err(|_| TensorUnsupportedReason::Operation)?;
    let OpDescriptor::SgdUpdate {
        learning_rate: original_rate,
        ..
    } = original.op
    else {
        // A synthetic rewrite of ordinary checked Mul/Sub has no native-compound permission.
        return Err(TensorUnsupportedReason::Operation);
    };
    // Floating descriptor equality aliases +0 and -0, but the frozen parameter ABI does not.
    if original != node || original_rate.to_bits() != learning_rate.to_bits() {
        return Err(TensorUnsupportedReason::Operation);
    }
    assess(node)
}

pub(super) fn assess(node: NodeDescriptor<'_>) -> Result<(), TensorUnsupportedReason> {
    let unsupported = |requirement| TensorUnsupportedReason::NumericalPolicy {
        requirement,
        options: node.numerical_options,
    };
    if node.numerical_options.reproducibility == PcuReproducibility::PortableV1 {
        return Err(unsupported(PcuNumericalRequirement::Reproducibility));
    }
    if node.numerical_mode != Some(PcuNumericalMode::Boundary)
        || node.numerical_options.compound_arithmetic != PcuCompoundArithmeticPolicy::BackendDefined
    {
        return Err(unsupported(PcuNumericalRequirement::CompoundArithmetic));
    }
    if node.float_underflow_policy == Some(PcuFloatUnderflowPolicy::RejectSubnormalResult) {
        return Err(TensorUnsupportedReason::UnderflowPolicy(
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ));
    }
    if node.scalar_type != PcuScalarType::F32 {
        return Err(TensorUnsupportedReason::ElementType);
    }
    if !matches!(node.op, OpDescriptor::SgdUpdate { learning_rate, .. } if learning_rate.is_finite())
    {
        return Err(TensorUnsupportedReason::Other(
            "native SGD requires a finite frozen F32 learning rate".into(),
        ));
    }
    // This private CUDA kernel has no BLAS emulation/TF32 path. Vendor-library environment
    // overrides are irrelevant and must not disqualify its preserved F32 arithmetic.
    Ok(())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
