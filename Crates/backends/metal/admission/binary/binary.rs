//! Complete structural admission for one checked six-format floating binary operation.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuRangePolicy,
    PcuValueType,
    PcuValueTypeCaps,
    assess_checked_float_binary_operands,
};
#[rustfmt::skip]
use crate::{
    MetalBuffer,
    MetalError,
    MetalPreparedFloatBinary,
    MetalSession,
};

/// One snapshotted F16/BF16/F32/F64/OFP8 Add/Sub/Mul/Div map, with exact neutral operand bindings.
pub struct MetalPreparedFloatBinaryKernel {
    map: MetalPreparedFloatBinary,
    inputs: [PcuBindingRef; 2],
    output: PcuBindingRef,
    extent: usize,
    bytes: usize,
    loads: [PcuBindingRef; 2],
    read_count: usize,
    requirements: fusion_pcu::PcuImplementationRequirements,
    operand_slots: [usize; 2],
    input_bytes: [usize; 2],
}
/// Compatibility name for exact F64 binary preparation.
pub type MetalPreparedF64BinaryKernel = MetalPreparedFloatBinaryKernel;
/// Exact F32 binary preparation.
pub type MetalPreparedF32BinaryKernel = MetalPreparedFloatBinaryKernel;
impl MetalPreparedFloatBinaryKernel {
    /// The full original numerical request, frozen independently of resource roles.
    #[must_use]
    pub const fn requirements(&self) -> fusion_pcu::PcuImplementationRequirements {
        self.requirements
    }
    #[must_use]
    pub const fn scalar_type(&self) -> fusion_pcu::PcuScalarType {
        self.map.scalar_type()
    }

    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.extent
    }
    #[must_use]
    pub const fn input_bindings(&self) -> [PcuBindingRef; 2] {
        self.inputs
    }
    /// Distinct bindings actually loaded, in original load order.
    /// Mathematical operand order remains available through `input_bindings`.
    #[must_use]
    pub fn actual_input_bindings(&self) -> &[PcuBindingRef] {
        &self.loads[..self.read_count]
    }
    /// Per-binding minimum byte spans for the actual unique read table.
    #[must_use]
    pub fn actual_input_byte_lengths(&self) -> &[usize] {
        &self.input_bytes[..self.read_count]
    }
    pub(crate) const fn actual_input_pair(&self) -> [PcuBindingRef; 2] {
        [
            self.loads[0],
            self.loads[if self.read_count == 1 { 0 } else { 1 }],
        ]
    }
    pub(crate) fn is_unread_declaration(&self, binding: PcuBindingRef) -> bool {
        self.binding_bytes(binding) == 0
    }
    #[must_use]
    pub const fn output_binding(&self) -> PcuBindingRef {
        self.output
    }
    pub(crate) fn binding_bytes(&self, binding: PcuBindingRef) -> usize {
        if binding == self.output {
            return self.bytes;
        }
        self.loads[..self.read_count]
            .iter()
            .zip(self.input_bytes)
            .filter_map(|(&load, bytes)| (load == binding).then_some(bytes))
            .max()
            .unwrap_or(0)
    }
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) const fn test_map(&self) -> &MetalPreparedFloatBinary {
        &self.map
    }
    /// Executes exact bound inputs with a fresh terminal output owner.
    ///
    /// # Errors
    /// Returns extent/session, checked arithmetic or terminal runtime failure.
    pub fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        if inputs
            .iter()
            .zip(self.inputs)
            .any(|(input, binding)| input.byte_len() != self.binding_bytes(binding))
        {
            return Err(MetalError::InvalidExtent);
        }
        self.map.execute_prefix(inputs, self.bytes)
    }
    pub(crate) fn execute_completed(
        &self,
        inputs: [&MetalBuffer; 2],
    ) -> Result<(MetalBuffer, Option<crate::MetalFault>), MetalError> {
        self.map
            .execute_completed(self.operand_slots.map(|slot| inputs[slot]), self.bytes)
    }
    pub(crate) fn execute_reads_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        self.map.execute_into(
            self.operand_slots.map(|slot| inputs[slot]),
            output,
            self.bytes,
        )
    }
    /// Runs into an exact same-session output; completed Clamp reports recovery after publication.
    ///
    /// # Errors
    /// Returns extent/affinity, runtime or structured checked arithmetic failure.
    pub fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        self.map.execute_into(inputs, output, self.bytes)
    }
}
struct Profile {
    scalar: fusion_pcu::PcuScalarType,
    operation: fusion_pcu::PcuDispatchFloatBinaryOp,
    underflow: fusion_pcu::PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    inputs: [PcuBindingRef; 2],
    output: PcuBindingRef,
    extent: usize,
    bytes: usize,
    loads: [PcuBindingRef; 2],
    read_count: usize,
    requirements: fusion_pcu::PcuImplementationRequirements,
    operand_slots: [usize; 2],
    input_bytes: [usize; 2],
    broadcast: [bool; 2],
}
impl Profile {
    #[allow(clippy::too_many_lines)] // Authentic node/SSA/schema/shape admission forms one cold gate.
    fn admit(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MetalError> {
        if kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            == fusion_pcu::PcuReproducibility::PortableV1
        {
            return Self::admit_portable(kernel);
        }
        super::require_scalar_numerics(kernel)?;
        if kernel.entry.logical_shape[1..] != [1, 1] {
            return Err(MetalError::Unsupported);
        }
        let (body, extent) = match kernel.ops {
            [
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (*body, *extent),
            [
                body @ ..,
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (body, kernel.entry.logical_shape[0]),
            _ => return Err(MetalError::Unsupported),
        };
        let [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                value_type: PcuValueType::Scalar(scalar),
                op,
                underflow_policy,
                range_policy,
                ..
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. }),
        ] = body
        else {
            return Err(MetalError::Unsupported);
        };
        if !matches!(
            scalar,
            fusion_pcu::PcuScalarType::F32
                | fusion_pcu::PcuScalarType::F64
                | fusion_pcu::PcuScalarType::F16
                | fusion_pcu::PcuScalarType::BF16
                | fusion_pcu::PcuScalarType::F8E4M3FN
                | fusion_pcu::PcuScalarType::F8E5M2
        ) {
            return Err(MetalError::Unsupported);
        }
        let schema = assess_checked_float_binary_operands(
            kernel,
            PcuValueType::Scalar(*scalar),
            *op,
            *underflow_policy,
            PcuValueTypeCaps::for_scalar(*scalar),
        )
        .map_err(|_| MetalError::Unsupported)?;
        let width = usize::from(scalar.bit_width()) / 8;
        let extent = usize::try_from(extent).map_err(|_| MetalError::InvalidExtent)?;
        let bytes = extent.checked_mul(width).ok_or(MetalError::InvalidExtent)?;
        let reads = schema.input_bindings();
        let loads = [reads[0], *reads.get(1).unwrap_or(&reads[0])];
        let operand_slots = schema.operand_inputs();
        let inputs = operand_slots.map(|slot| loads[slot]);
        let counts = schema.input_element_counts(extent);
        let input_bytes = [
            counts[0]
                .checked_mul(width)
                .ok_or(MetalError::InvalidExtent)?,
            counts[1]
                .checked_mul(width)
                .ok_or(MetalError::InvalidExtent)?,
        ];
        Ok(Self {
            scalar: *scalar,
            operation: *op,
            underflow: *underflow_policy,
            range: *range_policy,
            inputs,
            output: schema.output_binding(),
            extent,
            bytes,
            loads,
            read_count: reads.len(),
            requirements: kernel.numerical_requirements,
            operand_slots,
            input_bytes,
            broadcast: schema
                .operand_indices()
                .map(|index| index == PcuDispatchIndex::BindingElementZero),
        })
    }

    fn admit_portable(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MetalError> {
        // Eligibility is not conformance: this provider explicitly opts into only the four
        // independently qualified integer-synthesized encodings and this frozen map realization.
        let description =
            fusion_pcu::describe_portable_v1_map(kernel).map_err(|_| MetalError::Unsupported)?;
        if kernel.bindings.len() != 3
            || kernel.bindings[0].access != PcuBindingAccess::ReadOnly
            || kernel.bindings[1].access != PcuBindingAccess::ReadOnly
            || kernel.bindings[2].access == PcuBindingAccess::ReadOnly
            || description.load_bindings[0] == description.load_bindings[1]
            || description.load_bindings.iter().any(|&reference| {
                !kernel.bindings[..2]
                    .iter()
                    .any(|binding| binding.reference() == reference)
            })
            || description.output_binding != kernel.bindings[2].reference()
        {
            return Err(MetalError::Unsupported);
        }
        let width = usize::from(description.scalar.bit_width()) / 8;
        let extent = description.logical_extent as usize;
        let bytes = extent.checked_mul(width).ok_or(MetalError::InvalidExtent)?;
        let inputs = description
            .operands
            .map(|operand| description.load_bindings[usize::from(operand)]);
        let broadcast = description
            .operands
            .map(|operand| description.broadcast_loads[usize::from(operand)]);
        Ok(Self {
            scalar: description.scalar,
            operation: description.operation,
            underflow: description.underflow,
            range: PcuRangePolicy::Reject,
            inputs,
            output: description.output_binding,
            extent,
            bytes,
            loads: description.load_bindings,
            read_count: 2,
            requirements: kernel.numerical_requirements,
            operand_slots: description.operands.map(usize::from),
            input_bytes: description
                .broadcast_loads
                .map(|broadcast| if broadcast { width } else { bytes }),
            broadcast,
        })
    }
}
impl MetalSession {
    /// Admits one neutral checked six-format floating binary map before compiling its integer realization.
    /// Readonly scalar broadcast retains exact byte extents and actual SSA operand order.
    /// Extra graph nodes and incompatible schemas remain unsupported.
    ///
    /// # Errors
    /// Returns unsupported structure or a native compilation/pipeline failure.
    pub fn prepare_float_binary_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedFloatBinaryKernel, MetalError> {
        let profile = Profile::admit(kernel)?;
        Ok(MetalPreparedFloatBinaryKernel {
            map: self
                .prepare_checked_float_binary_with_range(
                    profile.scalar,
                    profile.operation,
                    profile.underflow,
                    profile.range,
                )?
                .with_broadcast(profile.broadcast),
            inputs: profile.inputs,
            output: profile.output,
            extent: profile.extent,
            bytes: profile.bytes,
            loads: profile.loads,
            read_count: profile.read_count,
            requirements: profile.requirements,
            operand_slots: profile.operand_slots,
            input_bytes: profile.input_bytes,
        })
    }
    /// Prepares a width-restricted F64 binary profile.
    ///
    /// # Errors
    /// Rejects other widths before compiling, or returns structural/native failure.
    pub fn prepare_f64_binary_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedF64BinaryKernel, MetalError> {
        if Profile::admit(kernel)?.scalar != fusion_pcu::PcuScalarType::F64 {
            return Err(MetalError::Unsupported);
        }
        self.prepare_float_binary_kernel(kernel)
    }
    /// Prepares a width-restricted F32 binary profile.
    ///
    /// # Errors
    /// Rejects other widths before compiling, or returns structural/native failure.
    pub fn prepare_f32_binary_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedF32BinaryKernel, MetalError> {
        if Profile::admit(kernel)?.scalar != fusion_pcu::PcuScalarType::F32 {
            return Err(MetalError::Unsupported);
        }
        self.prepare_float_binary_kernel(kernel)
    }
}
#[cfg(test)]
#[path = "tests.rs"]
pub mod tests;

/// Pure canonical eligibility; a native error after this gate must not be hidden by another factory.
pub fn is_checked_binary_kernel(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    Profile::admit(kernel).is_ok()
}
