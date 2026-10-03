//! Structural admission of bounded neutral checked fourteen-width integer map profiles.

#[path = "binary/binary.rs"]
pub mod binary;
#[path = "carrier/carrier.rs"]
pub mod carrier;
#[path = "unary/unary.rs"]
mod unary;
#[rustfmt::skip]
pub use unary::{
    MetalCheckedUnaryPlan,
    MetalPortableUnaryPlan,
};
pub use carrier::MetalPreparedCarrierKernel;
#[rustfmt::skip]
pub use binary::{
    MetalPreparedF32BinaryKernel,
    MetalPreparedF64BinaryKernel,
    MetalPreparedFloatBinaryKernel,
};

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuValueType,
    PcuValueTypeCaps,
    assess_checked_integer_binary_operands,
    validate_scalar_identity_kernel,
};
#[rustfmt::skip]
use crate::{
    MetalBuffer,
    MetalError,
    MetalIntegerOp,
    MetalSession,
};

pub fn require_scalar_numerics(kernel: &PcuDispatchKernelIr<'_>) -> Result<(), MetalError> {
    // Scalar instruction checks remain exact under compound/precision permissions. Portable
    // reproducibility requires its separate qualified binary admission; this gate has no profile.
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != fusion_pcu::PcuReproducibility::Unspecified
    {
        return Err(MetalError::Unsupported);
    }
    Ok(())
}

/// A snapshotted, structurally admitted neutral checked fourteen-width integer map.
///
/// This bounded adapter admits one fourteen-width Add/Sub/Mul, Reject or Clamp, direct/grid/broadcast.
/// General graphs, raw ALU, floating operations, `MatMul` and training reject before
/// compilation. No caller IR or parameter borrow survives preparation.
pub struct MetalPreparedIntegerKernel {
    map: crate::runtime::integer::IntegerMap,
    input_bindings: [PcuBindingRef; 2],
    output_binding: PcuBindingRef,
    extent: usize,
    scalar: fusion_pcu::PcuScalarType,
    bytes: usize,
    input_bytes: [usize; 2],
    loads: [PcuBindingRef; 2],
    read_count: usize,
    read_bytes: [usize; 2],
    operand_slots: [usize; 2],
    requirements: fusion_pcu::PcuImplementationRequirements,
}
/// Compatibility name retained for existing U32 callers.
pub type MetalPreparedU32Kernel = MetalPreparedIntegerKernel;

impl MetalPreparedIntegerKernel {
    pub(crate) fn execute_completed(
        &self,
        inputs: [&MetalBuffer; 2],
    ) -> Result<(MetalBuffer, Option<crate::MetalFault>), MetalError> {
        self.map
            .execute_completed(self.operand_slots.map(|slot| inputs[slot]))
    }
    /// Full original numerical request retained independently of actual resource roles.
    #[must_use]
    pub const fn requirements(&self) -> fusion_pcu::PcuImplementationRequirements {
        self.requirements
    }
    /// Actual unique loaded resources, in original `BindingLoad` order.
    #[must_use]
    pub fn actual_input_bindings(&self) -> &[PcuBindingRef] {
        &self.loads[..self.read_count]
    }
    /// Minimum physical spans for the actual unique read table.
    #[must_use]
    pub fn actual_input_byte_lengths(&self) -> &[usize] {
        &self.read_bytes[..self.read_count]
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
    pub(crate) fn binding_bytes(&self, binding: PcuBindingRef) -> usize {
        if binding == self.output_binding {
            return self.bytes;
        }
        self.loads[..self.read_count]
            .iter()
            .zip(self.read_bytes)
            .filter_map(|(&load, bytes)| (load == binding).then_some(bytes))
            .max()
            .unwrap_or(0)
    }

    #[must_use]
    pub const fn scalar_type(&self) -> fusion_pcu::PcuScalarType {
        self.scalar
    }
    pub(crate) fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        self.map
            .execute_into(self.operand_slots.map(|slot| inputs[slot]), output)
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.extent
    }
    #[must_use]
    pub const fn input_bindings(&self) -> [PcuBindingRef; 2] {
        self.input_bindings
    }
    #[must_use]
    pub const fn output_binding(&self) -> PcuBindingRef {
        self.output_binding
    }
    /// Executes inputs in the order returned by `input_bindings` with fresh owned output.
    ///
    /// # Errors
    /// Returns extent/affinity, terminal runtime or checked arithmetic failure.
    pub fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        if inputs
            .iter()
            .zip(self.input_bytes)
            .any(|(input, required)| input.byte_len() != required)
        {
            return Err(MetalError::InvalidExtent);
        }
        self.map.execute(inputs)
    }
}
impl MetalSession {
    /// Validates the existing neutral checked-u32 kernel and compiles its fixed profile.
    ///
    /// Grid-stride maps preserve logical extent and index fault priority; this implementation
    /// launches one Metal thread per logical element rather than replaying the physical stride.
    ///
    /// # Errors
    /// Unsupported IR rejects before native compilation; native compiler errors remain visible.
    pub fn prepare_u32_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedIntegerKernel, MetalError> {
        if kernel.bindings.first().is_none_or(|binding| {
            binding.binding_type != fusion_pcu::PcuBindingType::Value(PcuValueType::u32())
        }) {
            return Err(MetalError::Unsupported);
        }
        self.prepare_integer_kernel(kernel)
    }
    /// Prepares checked fourteen-width Add/Sub/Mul with exact byte extents and observable Clamp.
    ///
    /// # Errors
    /// Returns unsupported structure before native compilation, or a native error.
    pub fn prepare_integer_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedIntegerKernel, MetalError> {
        let profile = Profile::admit(kernel)?;
        Ok(MetalPreparedIntegerKernel {
            map: self.prepare_checked_integer_map(
                profile.scalar,
                profile.operation,
                profile.range,
                profile.extent,
                profile.broadcast,
            )?,
            input_bindings: profile.inputs,
            output_binding: profile.output,
            extent: profile.extent,
            scalar: profile.scalar,
            bytes: profile
                .extent
                .checked_mul(usize::from(profile.scalar.bit_width()) / 8)
                .ok_or(MetalError::InvalidExtent)?,
            input_bytes: profile.operand_slots.map(|slot| profile.read_bytes[slot]),
            loads: profile.loads,
            read_count: profile.read_count,
            read_bytes: profile.read_bytes,
            operand_slots: profile.operand_slots,
            requirements: kernel.numerical_requirements,
        })
    }
}
struct Profile {
    scalar: fusion_pcu::PcuScalarType,
    operation: MetalIntegerOp,
    range: fusion_pcu::PcuRangePolicy,
    broadcast: [bool; 2],
    inputs: [PcuBindingRef; 2],
    output: PcuBindingRef,
    extent: usize,
    loads: [PcuBindingRef; 2],
    read_count: usize,
    read_bytes: [usize; 2],
    operand_slots: [usize; 2],
}
impl Profile {
    fn identity_scalar(kernel: &PcuDispatchKernelIr<'_>) -> Option<fusion_pcu::PcuScalarType> {
        [
            fusion_pcu::PcuScalarType::I32,
            fusion_pcu::PcuScalarType::U32,
            fusion_pcu::PcuScalarType::I64,
            fusion_pcu::PcuScalarType::U64,
        ]
        .into_iter()
        .find(|&scalar| validate_scalar_identity_kernel(kernel, scalar).is_ok())
    }
    #[allow(clippy::too_many_lines)] // One cold gate checks numerical, schema, SSA and opcode admission.
    fn admit(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MetalError> {
        require_scalar_numerics(kernel)?;
        if kernel.entry.logical_shape[1..] != [1, 1] {
            return Err(MetalError::Unsupported);
        }
        let (body, extent) = if let [PcuDispatchOp::GridStrideLoop { extent, body }, _] = kernel.ops
        {
            (*body, *extent)
        } else {
            if kernel.entry.logical_shape[1..] != [1, 1] {
                return Err(MetalError::Unsupported);
            }
            (kernel.ops, kernel.entry.logical_shape[0])
        };
        if extent == 0 {
            return Err(MetalError::InvalidExtent);
        }
        if let Some(scalar) = Self::identity_scalar(kernel) {
            let input = kernel
                .bindings
                .iter()
                .find(|binding| binding.access == fusion_pcu::PcuBindingAccess::ReadOnly)
                .ok_or(MetalError::Unsupported)?
                .reference();
            let output = kernel
                .bindings
                .iter()
                .find(|binding| binding.access != fusion_pcu::PcuBindingAccess::ReadOnly)
                .ok_or(MetalError::Unsupported)?
                .reference();
            return Ok(Self {
                scalar,
                operation: MetalIntegerOp::Identity,
                range: kernel.numerical_requirements.range_policy,
                broadcast: [false; 2],
                inputs: [input, input],
                output,
                extent: extent as usize,
                loads: [input; 2],
                read_count: 1,
                read_bytes: [
                    usize::try_from(extent)
                        .ok()
                        .and_then(|extent| extent.checked_mul(usize::from(scalar.bit_width()) / 8))
                        .ok_or(MetalError::InvalidExtent)?,
                    0,
                ],
                operand_slots: [0, 0],
            });
        }
        let Some(PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            range_policy,
            op,
            ..
        })) = body.get(2)
        else {
            return Err(MetalError::Unsupported);
        };
        let scalar = match kernel.bindings.first().map(|binding| binding.binding_type) {
            Some(fusion_pcu::PcuBindingType::Value(PcuValueType::Scalar(scalar)))
                if matches!(
                    scalar,
                    fusion_pcu::PcuScalarType::I8
                        | fusion_pcu::PcuScalarType::U8
                        | fusion_pcu::PcuScalarType::I16
                        | fusion_pcu::PcuScalarType::U16
                        | fusion_pcu::PcuScalarType::I32
                        | fusion_pcu::PcuScalarType::U32
                        | fusion_pcu::PcuScalarType::I64
                        | fusion_pcu::PcuScalarType::U64
                        | fusion_pcu::PcuScalarType::I128
                        | fusion_pcu::PcuScalarType::U128
                        | fusion_pcu::PcuScalarType::I256
                        | fusion_pcu::PcuScalarType::U256
                        | fusion_pcu::PcuScalarType::I512
                        | fusion_pcu::PcuScalarType::U512
                ) =>
            {
                scalar
            }
            _ => return Err(MetalError::Unsupported),
        };
        let scalar_caps = PcuValueTypeCaps::for_scalar(scalar);
        let schema = assess_checked_integer_binary_operands(
            kernel,
            PcuValueType::Scalar(scalar),
            *op,
            scalar_caps,
        )
        .map_err(|_| MetalError::Unsupported)?;
        let operation = match op {
            PcuDispatchIntegerBinaryOp::Add => MetalIntegerOp::Add,
            PcuDispatchIntegerBinaryOp::Sub => MetalIntegerOp::Subtract,
            PcuDispatchIntegerBinaryOp::Mul => MetalIntegerOp::Multiply,
        };
        let reads = schema.input_bindings();
        let loads = [reads[0], *reads.get(1).unwrap_or(&reads[0])];
        let operand_slots = schema.operand_inputs();
        let extent = usize::try_from(extent).map_err(|_| MetalError::InvalidExtent)?;
        let counts = schema.input_element_counts(extent);
        let width = usize::from(scalar.bit_width()) / 8;
        let read_bytes = [
            counts[0]
                .checked_mul(width)
                .ok_or(MetalError::InvalidExtent)?,
            counts[1]
                .checked_mul(width)
                .ok_or(MetalError::InvalidExtent)?,
        ];
        Ok(Self {
            scalar,
            operation,
            range: *range_policy,
            broadcast: schema
                .operand_indices()
                .map(|index| index == PcuDispatchIndex::BindingElementZero),
            inputs: operand_slots.map(|slot| loads[slot]),
            output: schema.output_binding(),
            extent,
            loads,
            read_count: reads.len(),
            read_bytes,
            operand_slots,
        })
    }
}

/// A bounded neutral encoding-only six-format unary map.
pub struct MetalPreparedFloatKernel {
    map: crate::MetalPreparedFloatUnary,
    input: PcuBindingRef,
    output: PcuBindingRef,
    extent: usize,
    scalar: fusion_pcu::PcuScalarType,
    bytes: usize,
    input_bytes: usize,
    requirements: fusion_pcu::PcuImplementationRequirements,
}
/// Compatibility name for the original F32 encoding map.
pub type MetalPreparedF32Kernel = MetalPreparedFloatKernel;
/// F64 paired-limb encoding map.
pub type MetalPreparedF64Kernel = MetalPreparedFloatKernel;
impl MetalPreparedFloatKernel {
    /// Complete original numerical requirements retained by this detached unary program.
    #[must_use]
    pub const fn requirements(&self) -> fusion_pcu::PcuImplementationRequirements {
        self.requirements
    }
    pub(crate) const fn is_unread_declaration(&self, binding: PcuBindingRef) -> bool {
        !(binding.set == self.input.set && binding.binding == self.input.binding
            || binding.set == self.output.set && binding.binding == self.output.binding)
    }
    #[must_use]
    pub const fn scalar_type(&self) -> fusion_pcu::PcuScalarType {
        self.scalar
    }
    pub(crate) const fn binding_bytes(&self, binding: PcuBindingRef) -> usize {
        if binding.set == self.input.set && binding.binding == self.input.binding {
            self.input_bytes
        } else if binding.set == self.output.set && binding.binding == self.output.binding {
            self.bytes
        } else {
            0
        }
    }
    pub(crate) fn execute_completed(
        &self,
        input: &MetalBuffer,
    ) -> Result<(MetalBuffer, Option<crate::MetalFault>), MetalError> {
        self.map.execute_completed(input, self.bytes)
    }
    /// Writes the admitted prefix into a same-session borrowed owner.
    ///
    /// A recovered range error retains a fully completed payload. Fatal errors may write the
    /// physical prefix; callers must discard that logical value rather than read partial bytes.
    ///
    /// # Errors
    /// Returns affinity, extent, terminal execution or checked arithmetic failure.
    pub fn execute_into(
        &self,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        self.map.execute_into(input, output, self.bytes)
    }
    #[must_use]
    pub const fn input_binding(&self) -> PcuBindingRef {
        self.input
    }
    #[must_use]
    pub const fn output_binding(&self) -> PcuBindingRef {
        self.output
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.extent
    }
    /// Executes an exact encoding map with the admitted shape and policy.
    ///
    /// # Errors
    /// Returns extent/affinity, numerical fault or operational failure.
    pub fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        if input.byte_len() != self.input_bytes {
            return Err(MetalError::InvalidExtent);
        }
        self.map.execute_prefix(input, self.bytes)
    }
}
impl MetalSession {
    /// Prepares exactly one neutral checked F32 unary operation.
    ///
    /// # Errors
    /// Rejects unsupported widths, policies, broadcast, graphs and resource schemas before work.
    pub fn prepare_f32_unary_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedFloatKernel, MetalError> {
        if kernel.bindings.first().is_none_or(|binding| {
            binding.binding_type != fusion_pcu::PcuBindingType::Value(PcuValueType::f32())
        }) {
            return Err(MetalError::Unsupported);
        }
        self.prepare_float_unary_kernel(kernel)
    }
    /// Prepares one checked six-format encoding unary map with complete structural admission.
    ///
    /// # Errors
    /// Rejects unsupported policy, shape, scalar, broadcast or interface before compilation.
    #[allow(clippy::too_many_lines)] // One cold six-format schema/SSA/policy gate precedes compilation and detached execution.
    pub fn prepare_float_unary_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedFloatKernel, MetalError> {
        if kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            == fusion_pcu::PcuReproducibility::PortableV1
        {
            return unary::prepare(self, kernel);
        }
        unary::prepare_normal(self, kernel)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingStorageClass,
        PcuDispatchControlOp,
        PcuDispatchEntryPoint,
        PcuDispatchValueId,
        PcuKernelId,
    };

    pub fn fixture(grid: bool, broadcast: bool, visit: impl FnOnce(&PcuDispatchKernelIr<'_>)) {
        fixture_typed::<u32>(grid, broadcast, visit);
    }
    pub(super) fn fixture_typed<T: fusion_pcu::PcuScalar>(
        grid: bool,
        broadcast: bool,
        visit: impl FnOnce(&PcuDispatchKernelIr<'_>),
    ) {
        let bindings = [
            PcuBinding::scalar::<T>(
                Some("a"),
                2,
                3,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<T>(
                Some("b"),
                2,
                7,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<T>(
                Some("out"),
                4,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let index = if grid {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: bindings[0].reference(),
                index: if broadcast {
                    PcuDispatchIndex::BindingElementZero
                } else {
                    index
                },
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: bindings[1].reference(),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                range_policy: fusion_pcu::PcuRangePolicy::Reject,
                value_type: PcuValueType::Scalar(T::TYPE),
                op: PcuDispatchIntegerBinaryOp::Sub,
                result: PcuDispatchValueId(3),
                lhs: PcuDispatchValueId(2),
                rhs: PcuDispatchValueId(1),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: bindings[2].reference(),
                index,
                value: PcuDispatchValueId(3),
            }),
        ];
        let direct = [
            body[0],
            body[1],
            body[2],
            body[3],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 3,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: PcuKernelId(7),
            entry: PcuDispatchEntryPoint {
                name: "checked-map",
                logical_shape: [3, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if grid { &grid_ops } else { &direct },
            type_caps: PcuValueTypeCaps::for_value_type(PcuValueType::Scalar(T::TYPE)),
            feature_caps: fusion_pcu::PcuDispatchFeatureCaps::empty(),
        };
        visit(&kernel);
    }

    #[test]
    fn neutral_admission_preserves_operands_and_broadcast_roles() {
        for grid in [false, true] {
            fixture(grid, false, |kernel| {
                let profile = Profile::admit(kernel).unwrap();
                assert_eq!(
                    profile.inputs,
                    [PcuBindingRef::new(2, 7), PcuBindingRef::new(2, 3)]
                );
                assert_eq!(profile.extent, 3);
                assert_eq!(profile.operation, MetalIntegerOp::Subtract);
            });
            fixture(grid, true, |kernel| {
                let profile = Profile::admit(kernel).unwrap();
                assert_eq!(
                    profile.inputs,
                    [PcuBindingRef::new(2, 7), PcuBindingRef::new(2, 3)]
                );
                assert_eq!(profile.broadcast, [false, true]);
            });
        }
    }

    #[test]
    fn signed_profile_preserves_operand_order_widths_and_broadcast() {
        for grid in [false, true] {
            fixture_typed::<i32>(grid, false, |kernel| {
                let profile = Profile::admit(kernel).unwrap();
                assert_eq!(profile.scalar, fusion_pcu::PcuScalarType::I32);
                assert_eq!(
                    profile.inputs,
                    [PcuBindingRef::new(2, 7), PcuBindingRef::new(2, 3)]
                );
                assert_eq!(profile.operation, MetalIntegerOp::Subtract);
            });
            fixture_typed::<i32>(grid, true, |kernel| {
                let profile = Profile::admit(kernel).unwrap();
                assert_eq!(
                    profile.inputs,
                    [PcuBindingRef::new(2, 7), PcuBindingRef::new(2, 3)]
                );
                assert_eq!(profile.broadcast, [false, true]);
            });
            fixture_typed::<i128>(grid, false, |kernel| {
                assert_eq!(
                    Profile::admit(kernel).unwrap().scalar,
                    fusion_pcu::PcuScalarType::I128
                );
            });
        }
    }
    #[test]
    fn qword_admission_preserves_scalar_and_swapped_operands() {
        for grid in [false, true] {
            fixture_typed::<i64>(grid, false, |kernel| {
                let profile = Profile::admit(kernel).unwrap();
                assert_eq!(profile.scalar, fusion_pcu::PcuScalarType::I64);
                assert_eq!(
                    profile.inputs,
                    [PcuBindingRef::new(2, 7), PcuBindingRef::new(2, 3)]
                );
            });
            fixture_typed::<u64>(grid, false, |kernel| {
                assert_eq!(
                    Profile::admit(kernel).unwrap().scalar,
                    fusion_pcu::PcuScalarType::U64
                );
            });
        }
    }
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Requires actual Metal I32 direct/grid hardware qualification."]
    fn signed_neutral_direct_grid_preserve_swapped_operands_and_fault_kind() {
        let session = MetalSession::open(0).unwrap();
        for grid in [false, true] {
            fixture_typed::<i32>(grid, false, |kernel| {
                let prepared = session.prepare_integer_kernel(kernel).unwrap();
                let left = session
                    .upload_u32(&[9, 15, i32::MIN.cast_unsigned()])
                    .unwrap();
                let right = session
                    .upload_u32(&[1, 4, i32::MIN.cast_unsigned()])
                    .unwrap();
                assert_eq!(
                    prepared
                        .execute([&left, &right])
                        .unwrap()
                        .download_u32()
                        .unwrap(),
                    [8, 11, 0]
                );
                let bad = session
                    .upload_u32(&[i32::MIN.cast_unsigned(), 15, 1])
                    .unwrap();
                let Err(MetalError::Arithmetic(fault)) = prepared.execute([&bad, &right]) else {
                    panic!("missing signed underflow");
                };
                assert_eq!(
                    fault.kind,
                    fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow
                );
                assert_eq!(fault.invocation_id, 0);
                assert_eq!(
                    prepared
                        .execute([&left, &right])
                        .unwrap()
                        .download_u32()
                        .unwrap(),
                    [8, 11, 0]
                );
            });
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Requires actual macOS Metal device."]
    fn neutral_prepared_direct_and_grid_maps_execute_on_metal() {
        let session = MetalSession::open(0).unwrap();
        for grid in [false, true] {
            fixture(grid, false, |kernel| {
                let prepared = session.prepare_u32_kernel(kernel).unwrap();
                let left = session.upload_u32(&[9, 15, u32::MAX]).unwrap();
                let right = session.upload_u32(&[1, 4, u32::MAX]).unwrap();
                assert_eq!(
                    prepared
                        .execute([&left, &right])
                        .unwrap()
                        .download_u32()
                        .unwrap(),
                    [8, 11, 0]
                );
                assert!(matches!(
                    prepared.execute([&session.upload_u32(&[1]).unwrap(), &right]),
                    Err(MetalError::InvalidExtent)
                ));
            });
        }
    }
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Requires actual macOS Metal device."]
    fn host_checked_fault_keeps_output_and_retry_preserves_tail() {
        use fusion_pcu::{PcuHostArgument, PcuHostKernelBackend, PcuPreparedHostKernel};
        let session = MetalSession::open(0).unwrap();
        for grid in [false, true] {
            fixture(grid, false, |kernel| {
                let mut prepared = session.prepare_host_kernel(kernel).unwrap();
                let mut output = [91_u32; 5];
                {
                    let mut args = [
                        PcuHostArgument::read(PcuBindingRef::new(2, 3), &[1_u32, 4, 9]),
                        PcuHostArgument::read(PcuBindingRef::new(2, 7), &[9_u32, 15, 10]),
                        PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut output),
                    ];
                    prepared.call(&mut args).unwrap();
                }
                assert_eq!(output, [8, 11, 1, 91, 91]);
                {
                    let mut args = [
                        PcuHostArgument::read(PcuBindingRef::new(2, 3), &[1_u32, 16, 11]),
                        PcuHostArgument::read(PcuBindingRef::new(2, 7), &[9_u32, 15, 10]),
                        PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut output),
                    ];
                    assert!(
                        matches!(prepared.call(&mut args), Err(fusion_pcu::PcuHostDispatchError::Backend(MetalError::Arithmetic(fault))) if fault.kind == fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow && fault.invocation_id == 1)
                    );
                }
                assert_eq!(output, [8, 11, 1, 91, 91]);
                {
                    let mut args = [
                        PcuHostArgument::read(PcuBindingRef::new(2, 3), &[1_u32, 2, 3]),
                        PcuHostArgument::read(PcuBindingRef::new(2, 7), &[7_u32, 8, 9]),
                        PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut output),
                    ];
                    prepared.call(&mut args).unwrap();
                }
                assert_eq!(output, [6, 6, 6, 91, 91]);
            });
        }
    }
}

#[cfg(test)]
#[path = "integer_roles/integer_roles.rs"]
mod integer_roles;

/// Retain the canonical integer path before any separately admitted composition.
pub fn is_checked_integer_kernel(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    Profile::admit(kernel).is_ok()
}
