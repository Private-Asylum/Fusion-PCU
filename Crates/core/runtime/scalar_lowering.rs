//! Bounded scalar helper lowering used by generated PCU companions.
//!
//! This context exists only while a kernel IR builder is assembled. Generated helper companions
//! call it directly, so helper composition has no dispatch-time callback or allocation.

#[rustfmt::skip]
use crate::{
    PcuDispatchDataOp,
    PcuDispatchCheckedFloatConversion,
    PcuDispatchFloatBinaryOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchValueId,
    PcuError,
    PcuParameterValue,
    PcuRangePolicy,
    PcuScalar,
    PcuFloatUnderflowPolicy,
    PcuValueType,
};
use crate::model::PcuDispatchKernelBuilder;
use core::marker::PhantomData;

/// Maximum nested helper depth admitted during generated IR construction.
pub const PCU_SCALAR_HELPER_MAX_DEPTH: u8 = 32;

/// Floating scalar types admitted by the generated arithmetic-helper profile.
pub trait PcuFloatScalar: PcuScalar {
    /// IR value type used by arithmetic on this scalar.
    const VALUE_TYPE: PcuValueType;
}

impl PcuFloatScalar for f32 {
    const VALUE_TYPE: PcuValueType = PcuValueType::f32();
}

impl PcuFloatScalar for f64 {
    const VALUE_TYPE: PcuValueType = PcuValueType::f64();
}

/// A value id paired with its Rust scalar type for generated helper boundaries.
///
/// The phantom type prevents a companion lowered for one scalar profile from accepting a value
/// produced for another profile. The value remains only an SSA id; it does not own runtime data.
#[derive(Clone, Copy)]
pub struct PcuScalarValue<T: PcuScalar> {
    id: PcuDispatchValueId,
    scalar: PhantomData<T>,
}

impl<T: PcuScalar> PcuScalarValue<T> {
    const fn from_id(id: PcuDispatchValueId) -> Self {
        Self {
            id,
            scalar: PhantomData,
        }
    }

    /// Returns the underlying SSA id for use by the same lowering context.
    #[must_use]
    pub const fn id(self) -> PcuDispatchValueId {
        self.id
    }
}

/// IR construction context shared by generated per-function helper companions.
pub struct PcuScalarLowering<'a, const MAX_OPS: usize> {
    builder: Option<PcuDispatchKernelBuilder<'a, MAX_OPS>>,
    next_value: u16,
    depth: u8,
    float_underflow_policy: PcuFloatUnderflowPolicy,
    range_policy: PcuRangePolicy,
}

impl<'a, const MAX_OPS: usize> PcuScalarLowering<'a, MAX_OPS> {
    /// Starts scalar lowering after `next_value` has been reserved by the caller.
    #[must_use]
    pub const fn new(builder: PcuDispatchKernelBuilder<'a, MAX_OPS>, next_value: u16) -> Self {
        Self {
            builder: Some(builder),
            next_value,
            depth: 0,
            float_underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: PcuRangePolicy::Reject,
        }
    }

    /// Starts scalar lowering with the requested checked-float underflow policy.
    #[must_use]
    pub const fn with_float_underflow_policy(
        builder: PcuDispatchKernelBuilder<'a, MAX_OPS>,
        next_value: u16,
        policy: PcuFloatUnderflowPolicy,
    ) -> Self {
        Self {
            builder: Some(builder),
            next_value,
            depth: 0,
            float_underflow_policy: policy,
            range_policy: PcuRangePolicy::Reject,
        }
    }

    /// Sets the checked floating range-recovery policy for subsequently emitted operations.
    #[must_use]
    pub const fn with_range_policy(mut self, policy: PcuRangePolicy) -> Self {
        self.range_policy = policy;
        self
    }

    /// Returns the checked floating range-recovery policy for this lowering context.
    #[must_use]
    pub const fn range_policy(&self) -> PcuRangePolicy {
        self.range_policy
    }

    /// Enters one generated helper. Excessive nesting, including recursion, is rejected.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the bounded helper depth is exceeded.
    pub fn enter_helper(&mut self) -> Result<(), PcuError> {
        self.depth = self
            .depth
            .checked_add(1)
            .filter(|depth| *depth <= PCU_SCALAR_HELPER_MAX_DEPTH)
            .ok_or_else(PcuError::resource_exhausted)?;
        Ok(())
    }

    /// Leaves one generated helper scope.
    pub const fn leave_helper(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Emits an `f32` constant as a type-tagged value.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when values or builder operations are exhausted.
    pub fn constant_f32_value(&mut self, bits: u32) -> Result<PcuScalarValue<f32>, PcuError> {
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::Constant {
            result,
            value: PcuParameterValue::F32(bits),
        })?;
        Ok(PcuScalarValue::from_id(result))
    }

    /// Emits an `f64` constant as a type-tagged value.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when values or builder operations are exhausted.
    pub fn constant_f64_value(&mut self, bits: u64) -> Result<PcuScalarValue<f64>, PcuError> {
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::Constant {
            result,
            value: PcuParameterValue::F64(bits),
        })?;
        Ok(PcuScalarValue::from_id(result))
    }

    /// Emits checked floating Add/Sub/Mul/Div, preserving type and both arithmetic policies.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when values or builder operations are exhausted.
    pub fn checked_binary_value<T: PcuFloatScalar>(
        &mut self,
        op: PcuDispatchFloatBinaryOp,
        lhs: PcuScalarValue<T>,
        rhs: PcuScalarValue<T>,
    ) -> Result<PcuScalarValue<T>, PcuError> {
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::CheckedFloatBinary {
            value_type: T::VALUE_TYPE,
            op,
            underflow_policy: self.float_underflow_policy,
            range_policy: self.range_policy,
            result,
            lhs: lhs.id(),
            rhs: rhs.id(),
        })?;
        Ok(PcuScalarValue::from_id(result))
    }

    /// Emits checked floating `ReLU` with the context's arithmetic policies.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when values or builder operations are exhausted.
    pub fn checked_unary_value<T: PcuFloatScalar>(
        &mut self,
        op: PcuDispatchFloatUnaryOp,
        value: PcuScalarValue<T>,
    ) -> Result<PcuScalarValue<T>, PcuError> {
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: T::VALUE_TYPE,
            op,
            underflow_policy: self.float_underflow_policy,
            range_policy: self.range_policy,
            result,
            value: value.id(),
        })?;
        Ok(PcuScalarValue::from_id(result))
    }

    /// Emits a checked F64-to-F32 conversion with the context's arithmetic policies.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when values or builder operations are exhausted.
    pub fn checked_f64_to_f32_value(
        &mut self,
        value: PcuScalarValue<f64>,
    ) -> Result<PcuScalarValue<f32>, PcuError> {
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::CheckedFloatConvert {
            conversion: PcuDispatchCheckedFloatConversion::F64ToF32,
            underflow_policy: self.float_underflow_policy,
            range_policy: self.range_policy,
            result,
            value: value.id(),
        })?;
        Ok(PcuScalarValue::from_id(result))
    }

    /// Emits a checked exact F32-to-F64 widening, inheriting the context's policy metadata.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when values or builder operations are exhausted.
    pub fn checked_f32_to_f64_value(
        &mut self,
        value: PcuScalarValue<f32>,
    ) -> Result<PcuScalarValue<f64>, PcuError> {
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::CheckedFloatConvert {
            conversion: PcuDispatchCheckedFloatConversion::F32ToF64,
            underflow_policy: self.float_underflow_policy,
            range_policy: self.range_policy,
            result,
            value: value.id(),
        })?;
        Ok(PcuScalarValue::from_id(result))
    }

    /// Emits an indexed floating binding load with a compile-time scalar type.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when values or builder operations are exhausted.
    pub fn load_value<T: PcuFloatScalar>(
        &mut self,
        binding: crate::PcuBindingRef,
        index: PcuDispatchIndex,
    ) -> Result<PcuScalarValue<T>, PcuError> {
        self.load_typed(binding, index).map(PcuScalarValue::from_id)
    }

    /// Emits an indexed store of a type-tagged floating value.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder operation capacity is exhausted.
    pub fn store_value<T: PcuFloatScalar>(
        &mut self,
        binding: crate::PcuBindingRef,
        index: PcuDispatchIndex,
        value: PcuScalarValue<T>,
    ) -> Result<(), PcuError> {
        self.store_typed(binding, index, value.id())
    }

    /// Returns the completed builder.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` if the builder was already consumed.
    pub fn finish(mut self) -> Result<PcuDispatchKernelBuilder<'a, MAX_OPS>, PcuError> {
        self.builder.take().ok_or_else(PcuError::invalid)
    }

    fn fresh_value(&mut self) -> Result<PcuDispatchValueId, PcuError> {
        let id = self.next_value;
        self.next_value = self
            .next_value
            .checked_add(1)
            .ok_or_else(PcuError::resource_exhausted)?;
        Ok(PcuDispatchValueId(id))
    }

    fn load_typed(
        &mut self,
        binding: crate::PcuBindingRef,
        index: PcuDispatchIndex,
    ) -> Result<PcuDispatchValueId, PcuError> {
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index,
        })?;
        Ok(result)
    }

    fn store_typed(
        &mut self,
        binding: crate::PcuBindingRef,
        index: PcuDispatchIndex,
        value: PcuDispatchValueId,
    ) -> Result<(), PcuError> {
        self.push(PcuDispatchDataOp::BindingStore {
            binding,
            index,
            value,
        })
    }

    fn push(&mut self, operation: PcuDispatchDataOp) -> Result<(), PcuError> {
        let builder = self.builder.take().ok_or_else(PcuError::invalid)?;
        match builder.with_data_op(operation) {
            Ok(builder) => {
                self.builder = Some(builder);
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{PcuFloatScalar, PcuScalarLowering};
    #[rustfmt::skip]
    use crate::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchDataOp,
        PcuDispatchCheckedFloatConversion,
        PcuDispatchFloatBinaryOp,
        PcuDispatchFloatUnaryOp,
        PcuDispatchControlOp,
        PcuDispatchIndex,
        PcuFloatUnderflowPolicy,
        PcuRangePolicy,
        PcuValueType,
        validate_typed_dispatch_value_flow,
    };
    use crate::model::PcuDispatchKernelBuilder;

    #[allow(clippy::too_many_lines)] // Keeps typed construction and IR assertions in one fixture.
    fn assert_checked_ops_for_type<T: PcuFloatScalar>(
        expected_type: PcuValueType,
        policy: Option<PcuFloatUnderflowPolicy>,
    ) {
        let bindings = [
            PcuBinding::scalar::<T>(
                Some("lhs"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<T>(
                Some("rhs"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<T>(
                Some("output"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadWrite,
            ),
        ];
        for op in [
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchFloatBinaryOp::Div,
        ] {
            let make_builder = || {
                PcuDispatchKernelBuilder::<5>::new(1, "main", [4, 1, 1]).with_bindings(&bindings)
            };
            let mut lowering = policy.map_or_else(
                || PcuScalarLowering::new(make_builder(), 1),
                |policy| PcuScalarLowering::with_float_underflow_policy(make_builder(), 1, policy),
            );
            let lhs = lowering
                .load_value::<T>(
                    PcuBindingRef::new(0, 0),
                    PcuDispatchIndex::BindingElementZero,
                )
                .expect("typed lhs load");
            let rhs = lowering
                .load_value::<T>(
                    PcuBindingRef::new(0, 1),
                    PcuDispatchIndex::BindingElementZero,
                )
                .expect("typed rhs load");
            let result = lowering
                .checked_binary_value(op, lhs, rhs)
                .expect("typed checked-float binary");
            assert_eq!(result.id().0, 3);
            lowering
                .store_value(
                    PcuBindingRef::new(0, 2),
                    PcuDispatchIndex::InvocationId,
                    result,
                )
                .expect("typed result store");
            let builder = lowering
                .finish()
                .expect("builder returned")
                .with_control_op(PcuDispatchControlOp::Return)
                .expect("return added");
            builder.with_ir(|ir| {
                assert_eq!(
                    ir.bindings[0].binding_type.value_type(),
                    Some(expected_type)
                );
                let checked = ir.ops.iter().find_map(|instruction| match instruction {
                    crate::PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                        value_type,
                        op: actual_op,
                        underflow_policy,
                        range_policy,
                        result,
                        lhs,
                        rhs,
                    }) => Some((
                        *value_type,
                        *actual_op,
                        *underflow_policy,
                        *range_policy,
                        *result,
                        *lhs,
                        *rhs,
                    )),
                    _ => None,
                });
                assert_eq!(
                    checked,
                    Some((
                        expected_type,
                        op,
                        policy.unwrap_or_default(),
                        PcuRangePolicy::Reject,
                        crate::PcuDispatchValueId(3),
                        crate::PcuDispatchValueId(1),
                        crate::PcuDispatchValueId(2),
                    ))
                );
                assert_eq!(validate_typed_dispatch_value_flow(ir), Ok(()));
            });
        }
    }

    #[test]
    fn checked_typed_helper_values_preserve_type_operations_and_underflow_policy() {
        for policy in [
            None,
            Some(PcuFloatUnderflowPolicy::IeeeAfterRounding),
            Some(PcuFloatUnderflowPolicy::RejectSubnormalResult),
            Some(PcuFloatUnderflowPolicy::AllowGradualUnderflow),
        ] {
            assert_checked_ops_for_type::<f32>(PcuValueType::f32(), policy);
            assert_checked_ops_for_type::<f64>(PcuValueType::f64(), policy);
        }
    }

    #[test]
    fn checked_unary_helper_stamps_type_and_both_policies() {
        let bindings = [
            PcuBinding::scalar::<f32>(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<f32>(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let builder =
            PcuDispatchKernelBuilder::<4>::new(1, "relu", [4, 1, 1]).with_bindings(&bindings);
        let mut lowering = PcuScalarLowering::with_float_underflow_policy(
            builder,
            1,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .with_range_policy(PcuRangePolicy::Clamp);
        let input = lowering
            .load_value::<f32>(PcuBindingRef::new(0, 0), PcuDispatchIndex::InvocationId)
            .expect("typed input load");
        let output = lowering
            .checked_unary_value(PcuDispatchFloatUnaryOp::Relu, input)
            .expect("checked ReLU");
        assert_eq!(output.id().0, 2);
        lowering
            .store_value(
                PcuBindingRef::new(0, 1),
                PcuDispatchIndex::InvocationId,
                output,
            )
            .expect("typed output store");
        let builder = lowering
            .finish()
            .expect("builder returned")
            .with_control_op(PcuDispatchControlOp::Return)
            .expect("return added");
        builder.with_ir(|ir| {
            assert!(ir.ops.iter().any(|instruction| matches!(
                instruction,
                crate::PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                    value_type: PcuValueType::Scalar(crate::PcuScalarType::F32),
                    op: PcuDispatchFloatUnaryOp::Relu,
                    underflow_policy: PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    range_policy: PcuRangePolicy::Clamp,
                    result: crate::PcuDispatchValueId(2),
                    value: crate::PcuDispatchValueId(1),
                })
            )));
            assert_eq!(validate_typed_dispatch_value_flow(ir), Ok(()));
        });
    }

    #[test]
    fn checked_f64_to_f32_helper_preserves_conversion_types_policy_and_capability() {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let bindings = [
                PcuBinding::scalar::<f64>(
                    Some("input"),
                    0,
                    0,
                    PcuBindingStorageClass::Storage,
                    PcuBindingAccess::ReadOnly,
                ),
                PcuBinding::scalar::<f32>(
                    Some("output"),
                    0,
                    1,
                    PcuBindingStorageClass::Storage,
                    PcuBindingAccess::WriteOnly,
                ),
            ];
            let builder = PcuDispatchKernelBuilder::<4>::new(1, "convert", [1, 1, 1])
                .with_bindings(&bindings);
            let mut lowering = PcuScalarLowering::with_float_underflow_policy(builder, 1, policy);
            let value = lowering
                .load_value::<f64>(
                    PcuBindingRef::new(0, 0),
                    PcuDispatchIndex::BindingElementZero,
                )
                .expect("typed f64 load");
            let converted = lowering
                .checked_f64_to_f32_value(value)
                .expect("checked typed conversion");
            assert_eq!(converted.id().0, 2);
            lowering
                .store_value(
                    PcuBindingRef::new(0, 1),
                    PcuDispatchIndex::InvocationId,
                    converted,
                )
                .expect("typed f32 store");
            let builder = lowering.finish().expect("builder returned");
            builder.with_ir(|ir| {
                assert_eq!(validate_typed_dispatch_value_flow(ir), Ok(()));
                let operation = ir.ops.iter().find_map(|op| match op {
                    crate::PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                        conversion,
                        underflow_policy,
                        range_policy,
                        result,
                        value,
                    }) => Some((
                        *conversion,
                        *underflow_policy,
                        *range_policy,
                        *result,
                        *value,
                    )),
                    _ => None,
                });
                assert_eq!(
                    operation,
                    Some((
                        PcuDispatchCheckedFloatConversion::F64ToF32,
                        policy,
                        PcuRangePolicy::Reject,
                        crate::PcuDispatchValueId(2),
                        crate::PcuDispatchValueId(1),
                    ))
                );
                assert!(
                    ir.required_instruction_support()
                        .contains(crate::PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
                );
                assert!(
                    crate::PcuDispatchOpCaps::all()
                        .contains(crate::PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
                );
                assert!(
                    ir.required_type_support().contains(
                        crate::PcuValueTypeCaps::for_value_type(PcuValueType::f64())
                            .union(crate::PcuValueTypeCaps::for_value_type(PcuValueType::f32()))
                    )
                );
            });
        }
    }

    #[test]
    fn checked_f32_to_f64_helper_preserves_width_policy_and_capability() {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let bindings = [
                PcuBinding::scalar::<f32>(
                    Some("input"),
                    0,
                    0,
                    PcuBindingStorageClass::Storage,
                    PcuBindingAccess::ReadOnly,
                ),
                PcuBinding::scalar::<f64>(
                    Some("output"),
                    0,
                    1,
                    PcuBindingStorageClass::Storage,
                    PcuBindingAccess::WriteOnly,
                ),
            ];
            let builder =
                PcuDispatchKernelBuilder::<4>::new(1, "widen", [1, 1, 1]).with_bindings(&bindings);
            let mut lowering = PcuScalarLowering::with_float_underflow_policy(builder, 1, policy);
            let value = lowering
                .load_value::<f32>(
                    PcuBindingRef::new(0, 0),
                    PcuDispatchIndex::BindingElementZero,
                )
                .expect("typed f32 load");
            let widened = lowering
                .checked_f32_to_f64_value(value)
                .expect("checked exact widening");
            assert_eq!(widened.id().0, 2);
            lowering
                .store_value(
                    PcuBindingRef::new(0, 1),
                    PcuDispatchIndex::InvocationId,
                    widened,
                )
                .expect("typed f64 store");
            let builder = lowering.finish().expect("builder returned");
            builder.with_ir(|ir| {
                assert_eq!(validate_typed_dispatch_value_flow(ir), Ok(()));
                let operation = ir.ops.iter().find_map(|op| match op {
                    crate::PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                        conversion,
                        underflow_policy,
                        range_policy,
                        result,
                        value,
                    }) => Some((
                        *conversion,
                        *underflow_policy,
                        *range_policy,
                        *result,
                        *value,
                    )),
                    _ => None,
                });
                assert_eq!(
                    operation,
                    Some((
                        PcuDispatchCheckedFloatConversion::F32ToF64,
                        policy,
                        PcuRangePolicy::Reject,
                        crate::PcuDispatchValueId(2),
                        crate::PcuDispatchValueId(1),
                    ))
                );
                assert!(
                    ir.required_instruction_support()
                        .contains(crate::PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
                );
                assert!(
                    ir.required_type_support().contains(
                        crate::PcuValueTypeCaps::for_value_type(PcuValueType::f32())
                            .union(crate::PcuValueTypeCaps::for_value_type(PcuValueType::f64()))
                    )
                );
            });
        }
    }

    #[test]
    fn range_policy_defaults_to_reject_and_is_emitted_independently_of_underflow_policy() {
        let bindings = [
            PcuBinding::scalar::<f64>(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<f32>(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let builder = PcuDispatchKernelBuilder::<4>::new(1, "clamped_convert", [1, 1, 1])
            .with_bindings(&bindings);
        {
            let default_lowering = PcuScalarLowering::new(builder, 1);
            assert_eq!(default_lowering.range_policy(), PcuRangePolicy::Reject);
        }

        let builder = PcuDispatchKernelBuilder::<4>::new(1, "clamped_convert", [1, 1, 1])
            .with_bindings(&bindings);
        let mut lowering = PcuScalarLowering::with_float_underflow_policy(
            builder,
            1,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .with_range_policy(PcuRangePolicy::Clamp);
        assert_eq!(lowering.range_policy(), PcuRangePolicy::Clamp);
        let input = lowering
            .load_value::<f64>(
                PcuBindingRef::new(0, 0),
                PcuDispatchIndex::BindingElementZero,
            )
            .expect("typed input load");
        let converted = lowering
            .checked_f64_to_f32_value(input)
            .expect("checked conversion");
        lowering
            .store_value(
                PcuBindingRef::new(0, 1),
                PcuDispatchIndex::InvocationId,
                converted,
            )
            .expect("typed output store");
        let builder = lowering.finish().expect("builder returned");
        builder.with_ir(|ir| {
            let policies = ir.ops.iter().find_map(|op| match op {
                crate::PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                    underflow_policy,
                    range_policy,
                    ..
                }) => Some((*underflow_policy, *range_policy)),
                _ => None,
            });
            assert_eq!(
                policies,
                Some((
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuRangePolicy::Clamp,
                ))
            );
            assert!(
                ir.required_feature_support()
                    .contains(crate::PcuDispatchFeatureCaps::RANGE_CLAMP)
            );
        });
    }

    #[test]
    fn range_policy_is_emitted_for_checked_float_binary_operations() {
        let binary_bindings = [
            PcuBinding::scalar::<f32>(
                Some("lhs"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<f32>(
                Some("rhs"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<f32>(
                Some("output"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let builder = PcuDispatchKernelBuilder::<5>::new(1, "clamped_add", [1, 1, 1])
            .with_bindings(&binary_bindings);
        let mut lowering =
            PcuScalarLowering::new(builder, 1).with_range_policy(PcuRangePolicy::Clamp);
        let lhs = lowering
            .load_value::<f32>(
                PcuBindingRef::new(0, 0),
                PcuDispatchIndex::BindingElementZero,
            )
            .expect("typed lhs load");
        let rhs = lowering
            .load_value::<f32>(
                PcuBindingRef::new(0, 1),
                PcuDispatchIndex::BindingElementZero,
            )
            .expect("typed rhs load");
        let sum = lowering
            .checked_binary_value(PcuDispatchFloatBinaryOp::Add, lhs, rhs)
            .expect("checked addition");
        lowering
            .store_value(
                PcuBindingRef::new(0, 2),
                PcuDispatchIndex::InvocationId,
                sum,
            )
            .expect("typed sum store");
        let builder = lowering.finish().expect("builder returned");
        builder.with_ir(|ir| {
            assert!(ir.ops.iter().any(|op| matches!(
                op,
                crate::PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    range_policy: PcuRangePolicy::Clamp,
                    ..
                })
            )));
            assert!(
                ir.required_feature_support()
                    .contains(crate::PcuDispatchFeatureCaps::RANGE_CLAMP)
            );
        });
    }
}
