//! Bounded scalar helper lowering used by generated PCU companions.
//!
//! This context exists only while a kernel IR builder is assembled. Generated helper companions
//! call it directly, so helper composition has no dispatch-time callback or allocation.

#[rustfmt::skip]
use crate::{
    PcuDispatchAluOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchValueId,
    PcuError,
    PcuParameterValue,
    PcuScalar,
    PcuValueType,
};
use core::marker::PhantomData;
use crate::model::PcuDispatchKernelBuilder;

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
}

impl<'a, const MAX_OPS: usize> PcuScalarLowering<'a, MAX_OPS> {
    /// Starts scalar lowering after `next_value` has been reserved by the caller.
    #[must_use]
    pub const fn new(builder: PcuDispatchKernelBuilder<'a, MAX_OPS>, next_value: u16) -> Self {
        Self {
            builder: Some(builder),
            next_value,
            depth: 0,
        }
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

    /// Emits a floating arithmetic operation, preserving the scalar type through the helper ABI.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when values or builder operations are exhausted.
    pub fn alu_value<T: PcuFloatScalar>(
        &mut self,
        op: PcuDispatchAluOp,
        lhs: PcuScalarValue<T>,
        rhs: PcuScalarValue<T>,
    ) -> Result<PcuScalarValue<T>, PcuError> {
        self.alu_typed(T::VALUE_TYPE, op, lhs.id(), rhs.id())
            .map(PcuScalarValue::from_id)
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

    fn alu_typed(
        &mut self,
        value_type: PcuValueType,
        op: PcuDispatchAluOp,
        lhs: PcuDispatchValueId,
        rhs: PcuDispatchValueId,
    ) -> Result<PcuDispatchValueId, PcuError> {
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::Alu {
            value_type,
            result,
            op,
            lhs,
            rhs,
        })?;
        Ok(result)
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
    use super::PcuScalarLowering;
    #[rustfmt::skip]
    use crate::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchAluOp,
        PcuDispatchControlOp,
        PcuDispatchIndex,
        PcuValueType,
        validate_typed_dispatch_value_flow,
    };
    use crate::model::PcuDispatchKernelBuilder;

    #[test]
    fn f64_typed_helper_values_preserve_scalar_type_through_ir() {
        let bindings = [
            PcuBinding::scalar::<f64>(
                Some("seed"),
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
                PcuBindingAccess::ReadWrite,
            ),
        ];
        let builder =
            PcuDispatchKernelBuilder::<5>::new(1, "main", [4, 1, 1]).with_bindings(&bindings);
        let mut lowering = PcuScalarLowering::new(builder, 1);
        let seed = lowering
            .load_value::<f64>(
                PcuBindingRef::new(0, 0),
                PcuDispatchIndex::BindingElementZero,
            )
            .expect("f64 seed load");
        let one = lowering
            .constant_f64_value(1.0_f64.to_bits())
            .expect("f64 constant");
        let result = lowering
            .alu_value(PcuDispatchAluOp::Add, seed, one)
            .expect("f64 arithmetic");
        lowering
            .store_value(
                PcuBindingRef::new(0, 1),
                PcuDispatchIndex::InvocationId,
                result,
            )
            .expect("f64 store");
        let builder = lowering
            .finish()
            .expect("builder returned")
            .with_control_op(PcuDispatchControlOp::Return)
            .expect("return added");
        builder.with_ir(|ir| {
            assert_eq!(
                ir.bindings[0].binding_type.value_type(),
                Some(PcuValueType::f64())
            );
            assert_eq!(validate_typed_dispatch_value_flow(ir), Ok(()),);
        });
    }
}
