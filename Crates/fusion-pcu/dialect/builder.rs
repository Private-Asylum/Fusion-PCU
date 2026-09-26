//! Bounded typed construction helpers for the open dialect protocol.
//!
//! The builder only creates local typed SSA-like operation sequences. The current protocol does
//! not describe constants, named ports, or external bindings, so this module deliberately does
//! not invent those concepts.

use core::marker::PhantomData;

use crate::{
    PcuDialectEffects,
    PcuDialectFragment,
    PcuDialectId,
    PcuDialectOperand,
    PcuDialectOperation,
    PcuDialectSupport,
    PcuDialectValidationError,
    PcuDialectValueId,
    PcuDialectVersion,
    PcuValueType,
    validate_dialect_fragment,
};

/// Maps a Rust marker type to one scalar PCU value type.
pub trait DialectValueType {
    /// Scalar type represented by this marker.
    const PCU_TYPE: PcuValueType;
}

impl DialectValueType for bool {
    const PCU_TYPE: PcuValueType = PcuValueType::bool();
}

impl DialectValueType for u32 {
    const PCU_TYPE: PcuValueType = PcuValueType::u32();
}

impl DialectValueType for f32 {
    const PCU_TYPE: PcuValueType = PcuValueType::f32();
}

/// A typed, copyable reference to a preceding result in one builder.
///
/// SSA references are not owned resources: copying a handle represents ordinary fan-out and does
/// not duplicate the underlying operation or value storage.
pub struct DialectValue<T> {
    id: PcuDialectValueId,
    scope_id: u64,
    marker: PhantomData<fn() -> T>,
}

impl<T> Copy for DialectValue<T> {}

impl<T> Clone for DialectValue<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> core::fmt::Debug for DialectValue<T> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DialectValue")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

/// Errors reported while constructing a dialect fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialectBuilderError {
    /// Caller-provided operation storage has no free slot.
    OperationStorageFull,
    /// The next result ID would collide with the protocol's `UNUSED` sentinel.
    ValueIdExhausted,
    /// A typed handle has a different caller-assigned scope ID.
    ForeignValue,
    /// The operation is not admitted with exactly this signature and effect set.
    InvalidOperation(PcuDialectValidationError),
}

/// Caller-storage-backed builder for one dialect fragment.
pub struct PcuDialectBuilder<'a> {
    dialect: PcuDialectId<'a>,
    version: PcuDialectVersion,
    support: PcuDialectSupport<'a>,
    operations: &'a mut [PcuDialectOperation<'a>],
    scope_id: u64,
    len: usize,
    next_value_id: u16,
}

impl<'a> PcuDialectBuilder<'a> {
    /// Starts building a fragment in caller-owned operation storage.
    ///
    /// `scope_id` brands handles from this builder. Callers should assign a distinct value when
    /// creating another builder whose handles must not interoperate, including when reusing the
    /// same operation storage.
    #[must_use]
    pub const fn new(
        dialect: PcuDialectId<'a>,
        version: PcuDialectVersion,
        support: PcuDialectSupport<'a>,
        operations: &'a mut [PcuDialectOperation<'a>],
        scope_id: u64,
    ) -> Self {
        Self {
            dialect,
            version,
            support,
            operations,
            scope_id,
            len: 0,
            next_value_id: 0,
        }
    }

    /// Appends a zero-input operation that produces a typed value.
    ///
    /// # Errors
    ///
    /// Returns an error if storage is full, the value namespace is exhausted, or the support
    /// declaration does not admit this exact operation signature and effect set.
    pub fn operation0<T: DialectValueType>(
        &mut self,
        name: &'a str,
        effects: PcuDialectEffects,
    ) -> Result<DialectValue<T>, DialectBuilderError> {
        self.append_with_result::<T>(name, [PcuDialectOperand::UNUSED; 4], 0, effects)
    }

    /// Appends a one-input operation that consumes one typed value and produces another.
    ///
    /// The input handle can be copied when the same SSA value must feed multiple operations.
    ///
    /// # Errors
    ///
    /// Returns an error if storage is full, the value namespace is exhausted, the input belongs
    /// to another builder, or support rejects the exact operation signature/effects.
    pub fn operation1<A: DialectValueType, R: DialectValueType>(
        &mut self,
        name: &'a str,
        effects: PcuDialectEffects,
        input: DialectValue<A>,
    ) -> Result<DialectValue<R>, DialectBuilderError> {
        let operands = [
            Self::operand(&input, self.scope_id)?,
            PcuDialectOperand::UNUSED,
            PcuDialectOperand::UNUSED,
            PcuDialectOperand::UNUSED,
        ];
        self.append_with_result::<R>(name, operands, 1, effects)
    }

    /// Appends a two-input operation that consumes two typed values and produces another.
    ///
    /// Input handles can be copied for ordinary SSA fan-out.
    ///
    /// # Errors
    ///
    /// Returns an error if storage is full, the value namespace is exhausted, either input
    /// belongs to another builder, or support rejects the exact operation signature/effects.
    pub fn operation2<A: DialectValueType, B: DialectValueType, R: DialectValueType>(
        &mut self,
        name: &'a str,
        effects: PcuDialectEffects,
        first: DialectValue<A>,
        second: DialectValue<B>,
    ) -> Result<DialectValue<R>, DialectBuilderError> {
        let operands = [
            Self::operand(&first, self.scope_id)?,
            Self::operand(&second, self.scope_id)?,
            PcuDialectOperand::UNUSED,
            PcuDialectOperand::UNUSED,
        ];
        self.append_with_result::<R>(name, operands, 2, effects)
    }

    /// Appends a one-input effect operation with no result.
    ///
    /// # Errors
    ///
    /// Returns an error if storage is full, the input belongs to another builder, or support
    /// rejects the exact operation signature/effects.
    pub fn effect1<T: DialectValueType>(
        &mut self,
        name: &'a str,
        effects: PcuDialectEffects,
        input: DialectValue<T>,
    ) -> Result<(), DialectBuilderError> {
        let operands = [
            Self::operand(&input, self.scope_id)?,
            PcuDialectOperand::UNUSED,
            PcuDialectOperand::UNUSED,
            PcuDialectOperand::UNUSED,
        ];
        self.append(PcuDialectOperation {
            name,
            inputs: operands,
            input_count: 1,
            result: None,
            effects,
        })
    }

    /// Finishes construction and returns the borrowed immutable fragment.
    ///
    /// # Errors
    ///
    /// Returns an error if the final fragment fails protocol validation.
    pub fn finish(self) -> Result<PcuDialectFragment<'a>, DialectBuilderError> {
        let fragment = PcuDialectFragment {
            dialect: self.dialect,
            version: self.version,
            operations: &self.operations[..self.len],
        };
        validate_dialect_fragment(&fragment, &self.support)
            .map_err(DialectBuilderError::InvalidOperation)?;
        Ok(fragment)
    }

    const fn operand<T: DialectValueType>(
        value: &DialectValue<T>,
        scope_id: u64,
    ) -> Result<PcuDialectOperand, DialectBuilderError> {
        if value.scope_id != scope_id {
            return Err(DialectBuilderError::ForeignValue);
        }
        Ok(PcuDialectOperand {
            id: value.id,
            value_type: T::PCU_TYPE,
        })
    }

    fn append_with_result<T: DialectValueType>(
        &mut self,
        name: &'a str,
        inputs: [PcuDialectOperand; 4],
        input_count: u8,
        effects: PcuDialectEffects,
    ) -> Result<DialectValue<T>, DialectBuilderError> {
        if self.next_value_id == u16::MAX {
            return Err(DialectBuilderError::ValueIdExhausted);
        }
        let id = PcuDialectValueId(self.next_value_id);
        self.append(PcuDialectOperation {
            name,
            inputs,
            input_count,
            result: Some(PcuDialectOperand {
                id,
                value_type: T::PCU_TYPE,
            }),
            effects,
        })?;
        self.next_value_id += 1;
        Ok(DialectValue {
            id,
            scope_id: self.scope_id,
            marker: PhantomData,
        })
    }

    fn append(&mut self, operation: PcuDialectOperation<'a>) -> Result<(), DialectBuilderError> {
        if self.len == self.operations.len() {
            return Err(DialectBuilderError::OperationStorageFull);
        }
        self.operations[self.len] = operation;
        let candidate = PcuDialectFragment {
            dialect: self.dialect,
            version: self.version,
            operations: &self.operations[..=self.len],
        };
        validate_dialect_fragment(&candidate, &self.support)
            .map_err(DialectBuilderError::InvalidOperation)?;
        self.len += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compose_dialect_fragments,
        PcuDialectOperationSpec,
    };

    const INPUT: &str = "vm.input.u32";
    const ADD: &str = "vm.add.u32";
    const EMIT: &str = "vm.emit.u32";
    const U32: PcuValueType = PcuValueType::u32();
    const NO_INPUTS: &[PcuValueType] = &[];
    const ONE_U32: &[PcuValueType] = &[U32];
    const TWO_U32: &[PcuValueType] = &[U32, U32];
    const SPECS: &[PcuDialectOperationSpec<'static>] = &[
        PcuDialectOperationSpec {
            name: INPUT,
            input_types: NO_INPUTS,
            result_type: Some(U32),
            effects: PcuDialectEffects::READ_MEMORY,
        },
        PcuDialectOperationSpec {
            name: ADD,
            input_types: TWO_U32,
            result_type: Some(U32),
            effects: PcuDialectEffects::PURE,
        },
        PcuDialectOperationSpec {
            name: EMIT,
            input_types: ONE_U32,
            result_type: None,
            effects: PcuDialectEffects::WRITE_MEMORY,
        },
    ];
    const DIALECT: PcuDialectId<'static> = PcuDialectId("org.fusion.builder-test");
    const VERSION: PcuDialectVersion = PcuDialectVersion { major: 1, minor: 0 };
    const SUPPORT: PcuDialectSupport<'static> = PcuDialectSupport {
        dialect: DIALECT,
        major: 1,
        min_minor: 0,
        max_minor: 0,
        operations: SPECS,
        effects: PcuDialectEffects::READ_MEMORY.union(PcuDialectEffects::WRITE_MEMORY),
    };

    fn empty_operation() -> PcuDialectOperation<'static> {
        PcuDialectOperation {
            name: "",
            inputs: [PcuDialectOperand::UNUSED; 4],
            input_count: 0,
            result: None,
            effects: PcuDialectEffects::PURE,
        }
    }

    fn build_sequence<'a>(storage: &'a mut [PcuDialectOperation<'a>]) -> PcuDialectFragment<'a> {
        let mut builder = PcuDialectBuilder::new(DIALECT, VERSION, SUPPORT, storage, 1);
        let lhs = builder
            .operation0::<u32>(INPUT, PcuDialectEffects::READ_MEMORY)
            .unwrap();
        let rhs = builder
            .operation0::<u32>(INPUT, PcuDialectEffects::READ_MEMORY)
            .unwrap();
        let sum = builder
            .operation2::<u32, u32, u32>(ADD, PcuDialectEffects::PURE, lhs, rhs)
            .unwrap();
        builder
            .effect1(EMIT, PcuDialectEffects::WRITE_MEMORY, sum)
            .unwrap();
        builder.finish().unwrap()
    }

    #[test]
    fn typed_builder_builds_and_composes_valid_operation_sequence() {
        let mut first_storage = [empty_operation(); 4];
        let first = build_sequence(&mut first_storage);
        assert_eq!(validate_dialect_fragment(&first, &SUPPORT), Ok(()));

        let mut second_storage = [empty_operation(); 4];
        let second = build_sequence(&mut second_storage);
        let mut composed_storage = [empty_operation(); 8];
        let count = compose_dialect_fragments(&first, &second, &mut composed_storage).unwrap();
        let composed = PcuDialectFragment {
            dialect: DIALECT,
            version: VERSION,
            operations: &composed_storage[..count],
        };
        assert_eq!(count, 8);
        assert_eq!(composed_storage[4].result.unwrap().id, PcuDialectValueId(3));
        assert_eq!(composed_storage[6].result.unwrap().id, PcuDialectValueId(5));
        assert_eq!(composed_storage[7].inputs[0].id, PcuDialectValueId(5));
        assert_eq!(validate_dialect_fragment(&composed, &SUPPORT), Ok(()));
    }

    #[test]
    fn builder_rejects_wrong_signature_effect_and_storage_exhaustion() {
        let mut storage = [empty_operation(); 2];
        let mut builder = PcuDialectBuilder::new(DIALECT, VERSION, SUPPORT, &mut storage, 2);
        assert!(matches!(
            builder.operation0::<u32>(ADD, PcuDialectEffects::PURE),
            Err(DialectBuilderError::InvalidOperation(
                PcuDialectValidationError::SignatureMismatch
            ))
        ));
        assert!(matches!(
            builder.operation0::<u32>(INPUT, PcuDialectEffects::PURE),
            Err(DialectBuilderError::InvalidOperation(
                PcuDialectValidationError::UnsupportedEffects
            ))
        ));
        let lhs = builder
            .operation0::<u32>(INPUT, PcuDialectEffects::READ_MEMORY)
            .unwrap();
        let rhs = builder
            .operation0::<u32>(INPUT, PcuDialectEffects::READ_MEMORY)
            .unwrap();
        assert!(matches!(
            builder.operation0::<u32>(INPUT, PcuDialectEffects::READ_MEMORY),
            Err(DialectBuilderError::OperationStorageFull)
        ));
        assert!(matches!(
            builder.operation2::<u32, u32, u32>(ADD, PcuDialectEffects::PURE, lhs, rhs),
            Err(DialectBuilderError::OperationStorageFull)
        ));
    }

    #[test]
    fn builder_rejects_handle_from_another_builder() {
        let mut first_storage = [empty_operation(); 2];
        let mut first = PcuDialectBuilder::new(DIALECT, VERSION, SUPPORT, &mut first_storage, 3);
        let value = first
            .operation0::<u32>(INPUT, PcuDialectEffects::READ_MEMORY)
            .unwrap();
        let mut second_storage = [empty_operation(); 2];
        let mut second = PcuDialectBuilder::new(DIALECT, VERSION, SUPPORT, &mut second_storage, 4);
        assert!(matches!(
            second.operation1::<u32, u32>(EMIT, PcuDialectEffects::WRITE_MEMORY, value),
            Err(DialectBuilderError::ForeignValue)
        ));
    }

    #[test]
    fn typed_value_handle_supports_ssa_fan_out() {
        let mut storage = [empty_operation(); 3];
        let mut builder = PcuDialectBuilder::new(DIALECT, VERSION, SUPPORT, &mut storage, 5);
        let input = builder
            .operation0::<u32>(INPUT, PcuDialectEffects::READ_MEMORY)
            .unwrap();
        let doubled = builder
            .operation2::<u32, u32, u32>(ADD, PcuDialectEffects::PURE, input, input)
            .unwrap();
        builder
            .effect1(EMIT, PcuDialectEffects::WRITE_MEMORY, doubled)
            .unwrap();
        let fragment = builder.finish().unwrap();
        assert_eq!(validate_dialect_fragment(&fragment, &SUPPORT), Ok(()));
    }
}
