//! Cold byte-SSA projection to immutable physical integer-limb inputs and outputs.
use std::fmt::Write;
#[rustfmt::skip]
use fusion_pcu::{
    describe_scalar_transport_map,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuImplementationRequirements,
    PcuReproducibility,
    PcuScalarTransportDescription,
    PcuScalarTransportResource,
    PcuScalarType,
};
use crate::MlxError;

/// Detached bounded transport plan; eligibility alone does not advertise execution support.
///
/// Original access roles and numerical requirements remain exact. Initial input snapshots
/// and fresh immutable output arrays are distinct from original caller storage views.
#[derive(Clone, Debug)]
pub struct MlxTransportPlan {
    description: Box<PcuScalarTransportDescription<4>>,
    declarations: Vec<(PcuBindingRef, PcuBindingAccess)>,
    inputs: Vec<PcuScalarTransportResource>,
    input_bindings: Vec<PcuBindingRef>,
    input_counts: [usize; 4],
    outputs: Vec<PcuScalarTransportResource>,
    source: String,
    carrier_width: usize,
}
impl MlxTransportPlan {
    /// Validates original typed SSA, actual resources, complete encodings and bounded shapes.
    /// # Errors
    /// Rejects packed carriers, Portable, cross-index hazards, more than four resources,
    /// more than two writers, excessive instructions or unrepresentable physical extents.
    pub fn assess(
        kernel: &PcuDispatchKernelIr<'_>,
        scalar: PcuScalarType,
    ) -> Result<Self, MlxError> {
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        if kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            != PcuReproducibility::Unspecified
        {
            return Err(unsupported());
        }
        let description =
            describe_scalar_transport_map::<4>(kernel, scalar).map_err(|_| unsupported())?;
        if description
            .resources()
            .iter()
            .any(|resource| resource.has_cross_index_read_write())
        {
            return Err(unsupported());
        }
        let inputs: Vec<_> = description
            .resources()
            .iter()
            .copied()
            .filter(|resource| resource.minimum_initial_read_elements != 0)
            .collect();
        let outputs: Vec<_> = description
            .resources()
            .iter()
            .copied()
            .filter(|resource| resource.minimum_write_elements != 0)
            .collect();
        if inputs.is_empty() || outputs.is_empty() || outputs.len() > 2 {
            return Err(unsupported());
        }
        let mut input_counts = [0; 4];
        for (slot, resource) in inputs.iter().enumerate() {
            input_counts[slot] = usize::try_from(resource.minimum_initial_read_elements)
                .map_err(|_| MlxError::InvalidExtent)?;
        }
        let width = usize::from(scalar.bit_width()) / 8;
        let carrier_width = width.min(4);
        for resource in description.resources() {
            let count = usize::try_from(resource.minimum_elements())
                .map_err(|_| MlxError::InvalidExtent)?;
            validate_count(count, width, carrier_width)?;
        }
        let body = match kernel.ops {
            [PcuDispatchOp::GridStrideLoop { body, .. }, _] => *body,
            ops => &ops[..ops.len() - 1],
        };
        if body.len() > 128 {
            return Err(unsupported());
        }
        let source = lower(
            body,
            &inputs,
            &outputs,
            width / carrier_width,
            description.logical_extent,
        )?;
        Ok(Self {
            description: Box::new(description),
            declarations: kernel
                .bindings
                .iter()
                .map(|binding| (binding.reference(), binding.access))
                .collect(),
            input_bindings: inputs.iter().map(|resource| resource.binding).collect(),
            inputs,
            input_counts,
            outputs,
            source,
            carrier_width,
        })
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.description.scalar
    }
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.description.requirements
    }
    #[must_use]
    pub const fn element_count(&self) -> u32 {
        self.description.logical_extent
    }
    #[must_use]
    pub const fn declared_bindings(&self) -> &[(PcuBindingRef, PcuBindingAccess)] {
        self.declarations.as_slice()
    }
    /// Only original bytes needed before ordered stores, in actual first-access order.
    #[must_use]
    pub const fn inputs(&self) -> &[PcuScalarTransportResource] {
        self.inputs.as_slice()
    }
    /// Only unique original snapshot bindings, in the frozen actual input order.
    #[must_use]
    pub const fn input_bindings(&self) -> &[PcuBindingRef] {
        self.input_bindings.as_slice()
    }
    /// Logical original-byte minima; unused array positions have no native resource.
    #[must_use]
    pub const fn input_element_counts(&self) -> [usize; 4] {
        self.input_counts
    }
    /// Checks full native capacities independently of logical read spans, before SDK work.
    /// # Errors
    /// Rejects wrong actual arity, short capacities or overflowing physical/byte extents.
    pub fn assess_input_extents(&self, extents: &[usize]) -> Result<[usize; 4], MlxError> {
        if extents.len() != self.inputs.len() {
            return Err(MlxError::InvalidExtent);
        }
        let minimum = self.input_element_counts();
        let width = usize::from(self.scalar_type().bit_width()) / 8;
        let mut full = [0; 4];
        for (slot, &count) in extents.iter().enumerate() {
            if count < minimum[slot] {
                return Err(MlxError::InvalidExtent);
            }
            validate_count(count, width, self.carrier_width)?;
            full[slot] = count;
        }
        Ok(full)
    }
    /// Fresh final values, in actual first-access order; never in-place SDK writes.
    #[must_use]
    pub const fn outputs(&self) -> &[PcuScalarTransportResource] {
        self.outputs.as_slice()
    }
    #[must_use]
    pub fn resources(&self) -> &[PcuScalarTransportResource] {
        self.description.resources()
    }
    /// Cold retained custom-kernel body. Names refer only to bounded integer carriers.
    #[must_use]
    pub fn native_source(&self) -> &str {
        &self.source
    }
    #[must_use]
    pub const fn carrier_width(&self) -> usize {
        self.carrier_width
    }
}
fn unsupported() -> MlxError {
    MlxError::InvalidRequest("unsupported MLX scalar transport".into())
}
fn validate_count(count: usize, width: usize, carrier_width: usize) -> Result<(), MlxError> {
    let bytes = count.checked_mul(width).ok_or(MlxError::InvalidExtent)?;
    if count == 0
        || i32::try_from(bytes / carrier_width).is_err()
        || isize::try_from(bytes).is_err()
    {
        return Err(MlxError::InvalidExtent);
    }
    Ok(())
}
fn lower(
    body: &[PcuDispatchOp<'_>],
    inputs: &[PcuScalarTransportResource],
    outputs: &[PcuScalarTransportResource],
    limbs: usize,
    count: u32,
) -> Result<String, MlxError> {
    let mut source = format!(
        "uint i=thread_position_in_grid.x;\nif(i>={}u) return;\nuint lane=i/{}u; uint limb=i%{}u;\n",
        u64::from(count) * limbs as u64,
        limbs,
        limbs
    );
    let mut stored: Vec<(PcuBindingRef, u16)> = Vec::new();
    for op in body {
        match *op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                binding,
                index,
                result,
                ..
            }) => {
                if let Some((_, value)) = stored.iter().find(|(target, _)| *target == binding) {
                    writeln!(source, "auto v{}=v{value};", result.0).map_err(|_| unsupported())?;
                } else {
                    let slot = inputs
                        .iter()
                        .position(|r| r.binding == binding)
                        .ok_or_else(unsupported)?;
                    let index = if index == PcuDispatchIndex::BindingElementZero {
                        "limb"
                    } else {
                        "i"
                    };
                    writeln!(source, "auto v{}=input{slot}[{index}];", result.0)
                        .map_err(|_| unsupported())?;
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) => {
                if let Some((_, prior)) = stored.iter_mut().find(|(target, _)| *target == binding) {
                    *prior = value.0;
                } else {
                    stored.push((binding, value.0));
                }
            }
            _ => return Err(unsupported()),
        }
    }
    for (slot, resource) in outputs.iter().enumerate() {
        let (_, value) = stored
            .iter()
            .find(|(binding, _)| *binding == resource.binding)
            .ok_or_else(unsupported)?;
        writeln!(source, "output{slot}[i]=v{value};").map_err(|_| unsupported())?;
    }
    Ok(source)
}
