//! Small typed construction API for the current scalar `f32` indexed-map profile.
//!
//! This is a builder for an existing bounded PCU IR subset. It does not compile Rust
//! functions, infer arbitrary Rust types, or model control flow. Builders are move-only so
//! callers cannot branch them and allocate duplicate IDs. Value handles are scoped by kernel
//! ID; callers must use distinct IDs when building independent kernels.

use core::marker::PhantomData;

use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchAluOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchControlOp,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuError,
    PcuParameterValue,
    PcuScalarType,
    PcuValueType,
};
use crate::model::PcuDispatchKernelBuilder;

/// A typed value produced within one [`F32MapBuilder`].
///
/// Its identifier is private so callers cannot forge an IR reference. Values are SSA-like
/// handles scoped to the kernel ID supplied to the builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct F32Value {
    id: u16,
    kernel_id: u32,
    _type: PhantomData<fn() -> f32>,
}

/// Bounded builder for the scalar `f32` indexed-map subset.
///
/// Binding metadata is borrowed and validated as operations are appended. Operation storage
/// lives inline in the underlying dispatch builder; no heap allocation is used.
#[derive(Debug)]
pub struct F32MapBuilder<'a, const MAX_OPS: usize = 32> {
    inner: PcuDispatchKernelBuilder<'a, MAX_OPS>,
    bindings: &'a [PcuBinding<'a>],
    kernel_id: u32,
    next_value: u16,
}

impl<'a, const MAX_OPS: usize> F32MapBuilder<'a, MAX_OPS> {
    /// Start a kernel with a finite logical invocation shape and its resource bindings.
    #[must_use]
    pub const fn new(
        kernel_id: u32,
        entry_point: &'a str,
        logical_shape: [u32; 3],
        bindings: &'a [PcuBinding<'a>],
    ) -> Self {
        Self {
            inner: PcuDispatchKernelBuilder::new(kernel_id, entry_point, logical_shape)
                .with_bindings(bindings),
            bindings,
            kernel_id,
            // Value ID zero is reserved by the current SPIR-V dataflow profile.
            next_value: 1,
        }
    }

    /// Load a scalar `f32` binding at the current invocation ID.
    ///
    /// # Errors
    ///
    /// Returns an error when the binding is missing, incompatible, or operation storage is full.
    pub fn load_f32(mut self, binding: PcuBindingRef) -> Result<(Self, F32Value), PcuError> {
        self.check_binding(binding, false)?;
        let value = self.fresh_value()?;
        self.push(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(value.id),
            binding,
            index: PcuDispatchIndex::InvocationId,
        })?;
        Ok((self, value))
    }

    /// Construct a scalar `f32` constant.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation buffer is full or the value ID space is exhausted.
    pub fn constant(mut self, value: f32) -> Result<(Self, F32Value), PcuError> {
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::Constant {
            result: PcuDispatchValueId(result.id),
            value: PcuParameterValue::F32(value.to_bits()),
        })?;
        Ok((self, result))
    }

    /// Add two values produced by this builder.
    ///
    /// # Errors
    ///
    /// Returns an error when either value belongs to another kernel or operation/value capacity
    /// is exhausted.
    pub fn add(mut self, lhs: F32Value, rhs: F32Value) -> Result<(Self, F32Value), PcuError> {
        self.check_value(lhs)?;
        self.check_value(rhs)?;
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::Alu {
            result: PcuDispatchValueId(result.id),
            op: PcuDispatchAluOp::Add,
            lhs: PcuDispatchValueId(lhs.id),
            rhs: PcuDispatchValueId(rhs.id),
        })?;
        Ok((self, result))
    }

    /// Multiply two values produced by this builder.
    ///
    /// # Errors
    ///
    /// Returns an error when either value belongs to another kernel or operation/value capacity
    /// is exhausted.
    pub fn mul(mut self, lhs: F32Value, rhs: F32Value) -> Result<(Self, F32Value), PcuError> {
        self.check_value(lhs)?;
        self.check_value(rhs)?;
        let result = self.fresh_value()?;
        self.push(PcuDispatchDataOp::Alu {
            result: PcuDispatchValueId(result.id),
            op: PcuDispatchAluOp::Mul,
            lhs: PcuDispatchValueId(lhs.id),
            rhs: PcuDispatchValueId(rhs.id),
        })?;
        Ok((self, result))
    }

    /// Store one value to a writable scalar `f32` binding at the current invocation ID.
    ///
    /// # Errors
    ///
    /// Returns an error when the value or binding is incompatible, or operation storage is full.
    pub fn store_f32(mut self, binding: PcuBindingRef, value: F32Value) -> Result<Self, PcuError> {
        self.check_value(value)?;
        self.check_binding(binding, true)?;
        self.push(PcuDispatchDataOp::BindingStore {
            binding,
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(value.id),
        })?;
        self.inner = self
            .inner
            .with_op(PcuDispatchOp::Control(PcuDispatchControlOp::Return))?;
        Ok(self)
    }

    /// Finish construction and borrow the resulting backend-neutral kernel IR.
    #[must_use]
    pub fn ir(&self) -> PcuDispatchKernelIr<'_> {
        self.inner.ir()
    }

    fn fresh_value(&mut self) -> Result<F32Value, PcuError> {
        let id = self.next_value;
        self.next_value = self
            .next_value
            .checked_add(1)
            .ok_or_else(PcuError::resource_exhausted)?;
        Ok(F32Value {
            id,
            kernel_id: self.kernel_id,
            _type: PhantomData,
        })
    }

    fn push(&mut self, op: PcuDispatchDataOp) -> Result<(), PcuError> {
        self.inner = self.inner.with_data_op(op)?;
        Ok(())
    }

    fn check_binding(&self, reference: PcuBindingRef, write: bool) -> Result<(), PcuError> {
        let binding = self
            .bindings
            .iter()
            .find(|binding| binding.set == reference.set && binding.binding == reference.binding)
            .ok_or_else(PcuError::invalid)?;
        if binding.binding_type != PcuBindingType::Value(PcuValueType::Scalar(PcuScalarType::F32))
            || binding.storage != crate::PcuBindingStorageClass::Storage
            || (write && matches!(binding.access, PcuBindingAccess::ReadOnly))
            || (!write && matches!(binding.access, PcuBindingAccess::WriteOnly))
        {
            return Err(PcuError::invalid());
        }
        Ok(())
    }

    const fn check_value(&self, value: F32Value) -> Result<(), PcuError> {
        if value.kernel_id != self.kernel_id {
            return Err(PcuError::invalid());
        }
        if value.id >= self.next_value {
            return Err(PcuError::invalid());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PcuBindingStorageClass,
        PcuDispatchDataOp,
        PcuDispatchIndex,
        PcuDispatchOp,
        PcuDispatchValueId,
    };

    #[test]
    fn builds_f32_indexed_map_with_automatic_value_ids() {
        let bindings = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let (builder, input) = F32MapBuilder::<8>::new(7, "map", [65, 1, 1], &bindings)
            .load_f32(PcuBindingRef::new(0, 0))
            .unwrap();
        let (builder, bias) = builder.constant(0.5).unwrap();
        let (builder, sum) = builder.add(input, bias).unwrap();
        let builder = builder.store_f32(PcuBindingRef::new(0, 1), sum).unwrap();
        let ir = builder.ir();

        assert_eq!(ir.entry.logical_shape, [65, 1, 1]);
        assert_eq!(ir.ops.len(), 5);
        assert!(matches!(
            ir.ops[0],
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                index: PcuDispatchIndex::InvocationId,
                ..
            })
        ));
        assert!(
            matches!(ir.ops[1], PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
            result: PcuDispatchValueId(2), value: PcuParameterValue::F32(raw_bits),
        }) if raw_bits == 0.5f32.to_bits())
        );
        assert!(matches!(
            ir.ops[2],
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result: PcuDispatchValueId(3),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
                ..
            })
        ));
        assert!(matches!(
            ir.ops[3],
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                value: PcuDispatchValueId(3),
                index: PcuDispatchIndex::InvocationId,
                ..
            })
        ));
        assert!(matches!(
            ir.ops[4],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return)
        ));
    }

    #[test]
    fn rejects_wrong_binding_access_and_type() {
        let bindings = [
            PcuBinding::value(
                None,
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                None,
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadWrite,
                PcuValueType::u32(),
            ),
        ];
        assert!(
            F32MapBuilder::<4>::new(1, "bad", [1, 1, 1], &bindings)
                .store_f32(
                    PcuBindingRef::new(0, 0),
                    F32Value {
                        id: 0,
                        kernel_id: 1,
                        _type: PhantomData,
                    }
                )
                .is_err()
        );
        assert!(
            F32MapBuilder::<4>::new(1, "bad", [1, 1, 1], &bindings)
                .load_f32(PcuBindingRef::new(9, 9))
                .is_err()
        );
        assert!(
            F32MapBuilder::<4>::new(1, "bad", [1, 1, 1], &bindings)
                .load_f32(PcuBindingRef::new(0, 1))
                .is_err()
        );
    }

    #[test]
    fn rejects_non_storage_bindings() {
        let bindings = [PcuBinding::value(
            None,
            0,
            0,
            PcuBindingStorageClass::Uniform,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        )];
        let builder = F32MapBuilder::<2>::new(1, "uniform", [1, 1, 1], &bindings);
        assert!(builder.load_f32(PcuBindingRef::new(0, 0)).is_err());
    }

    #[test]
    fn rejects_values_from_another_kernel() {
        let bindings = [];
        let (_, value) = F32MapBuilder::<2>::new(1, "a", [1, 1, 1], &bindings)
            .constant(1.0)
            .unwrap();
        let other = F32MapBuilder::<2>::new(2, "b", [1, 1, 1], &bindings);
        assert!(other.constant(2.0).unwrap().0.add(value, value).is_err());
    }

    #[test]
    fn bounded_operation_capacity_is_enforced() {
        let bindings = [];
        let (builder, _) = F32MapBuilder::<1>::new(1, "capacity", [1, 1, 1], &bindings)
            .constant(1.0)
            .unwrap();
        assert!(builder.constant(2.0).is_err());
    }
}
