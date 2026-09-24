//! Open, versioned extension vocabulary for backend-specific PCU instructions.
//!
//! This module describes instruction identity, typed inputs/results, and declared effects. It
//! deliberately does not assign executable meaning to an operation name. A backend may admit a
//! dialect only when its consumer-specific support table recognizes that exact operation and
//! version.

use crate::PcuValueType;

/// Stable, namespaced identity supplied by a dialect author (for example `org.example.vm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDialectId<'a>(pub &'a str);

/// Explicit dialect version. Compatibility is exact-major and bounded-minor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDialectVersion {
    pub major: u16,
    pub minor: u16,
}

/// Declared externally visible effects of an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuDialectEffects(u16);

impl PcuDialectEffects {
    pub const PURE: Self = Self(0);
    pub const READ_MEMORY: Self = Self(1 << 0);
    pub const WRITE_MEMORY: Self = Self(1 << 1);
    pub const CONTROL: Self = Self(1 << 2);
    pub const SYNCHRONIZE: Self = Self(1 << 3);

    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, effects: Self) -> bool {
        self.0 & effects.0 == effects.0
    }

    #[must_use]
    pub const fn union(self, effects: Self) -> Self {
        Self(self.0 | effects.0)
    }
}

/// Local SSA-like value identity. IDs are scoped to one fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDialectValueId(pub u16);

/// A typed value reference used as an operation input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDialectOperand {
    pub id: PcuDialectValueId,
    pub value_type: PcuValueType,
}

impl PcuDialectOperand {
    pub const UNUSED: Self = Self {
        id: PcuDialectValueId(u16::MAX),
        value_type: PcuValueType::bool(),
    };
}

/// One external instruction descriptor; name plus contract, never implicit executable semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDialectOperation<'a> {
    pub name: &'a str,
    pub inputs: [PcuDialectOperand; 4],
    pub input_count: u8,
    pub result: Option<PcuDialectOperand>,
    pub effects: PcuDialectEffects,
}

/// A small immutable dialect fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuDialectFragment<'a> {
    pub dialect: PcuDialectId<'a>,
    pub version: PcuDialectVersion,
    pub operations: &'a [PcuDialectOperation<'a>],
}

/// Consumer declaration for one dialect it understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuDialectSupport<'a> {
    pub dialect: PcuDialectId<'a>,
    pub major: u16,
    pub min_minor: u16,
    pub max_minor: u16,
    pub operations: &'a [PcuDialectOperationSpec<'a>],
    pub effects: PcuDialectEffects,
}

/// Consumer-owned signature for one admitted external operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuDialectOperationSpec<'a> {
    pub name: &'a str,
    pub input_types: &'a [PcuValueType],
    pub result_type: Option<PcuValueType>,
    pub effects: PcuDialectEffects,
}

/// First contract failure while validating an extension fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuDialectValidationError {
    EmptyDialectId,
    EmptyOperationName,
    UnsupportedDialect,
    UnsupportedVersion,
    UnsupportedOperation,
    UnsupportedEffects,
    SignatureMismatch,
    UseBeforeDefinition(PcuDialectValueId),
    DuplicateResult(PcuDialectValueId),
    OperandTypeMismatch(PcuDialectValueId),
}

/// Validate identity/version, operation admission, effects, and local typed value flow.
///
/// # Errors
///
/// Returns the first contract violation, including unsupported operations or malformed local
/// value flow.
pub fn validate_dialect_fragment(
    fragment: &PcuDialectFragment<'_>,
    support: &PcuDialectSupport<'_>,
) -> Result<(), PcuDialectValidationError> {
    if fragment.dialect.0.is_empty() {
        return Err(PcuDialectValidationError::EmptyDialectId);
    }
    if fragment.dialect != support.dialect {
        return Err(PcuDialectValidationError::UnsupportedDialect);
    }
    if fragment.version.major != support.major
        || fragment.version.minor < support.min_minor
        || fragment.version.minor > support.max_minor
    {
        return Err(PcuDialectValidationError::UnsupportedVersion);
    }

    for (index, op) in fragment.operations.iter().enumerate() {
        if op.name.is_empty() {
            return Err(PcuDialectValidationError::EmptyOperationName);
        }
        if op.input_count as usize > op.inputs.len() {
            return Err(PcuDialectValidationError::UnsupportedOperation);
        }
        let Some(spec) = support
            .operations
            .iter()
            .find(|known| known.name == op.name)
        else {
            return Err(PcuDialectValidationError::UnsupportedOperation);
        };
        if spec.input_types.len() != op.input_count as usize
            || spec
                .input_types
                .iter()
                .zip(&op.inputs[..op.input_count as usize])
                .any(|(expected, found)| *expected != found.value_type)
            || spec.result_type != op.result.map(|result| result.value_type)
        {
            return Err(PcuDialectValidationError::SignatureMismatch);
        }
        if spec.effects != op.effects || !support.effects.contains(op.effects) {
            return Err(PcuDialectValidationError::UnsupportedEffects);
        }
        for input in &op.inputs[..op.input_count as usize] {
            let Some(definition) = fragment.operations[..index]
                .iter()
                .find_map(|prior| prior.result.filter(|r| r.id == input.id))
            else {
                return Err(PcuDialectValidationError::UseBeforeDefinition(input.id));
            };
            if definition.value_type != input.value_type {
                return Err(PcuDialectValidationError::OperandTypeMismatch(input.id));
            }
        }
        if let Some(result) = op.result
            && (fragment.operations[..index]
                .iter()
                .any(|prior| prior.result.is_some_and(|r| r.id == result.id))
                || op.inputs[..op.input_count as usize]
                    .iter()
                    .any(|input| input.id == result.id))
        {
            return Err(PcuDialectValidationError::DuplicateResult(result.id));
        }
    }
    Ok(())
}

/// Errors when composing two fragments into caller-owned operation storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuDialectComposeError {
    IdentityMismatch,
    VersionMismatch,
    OutputTooSmall,
    ValueIdOverflow,
    InvalidFragmentFlow,
}

/// Compose two same-dialect fragments, renumbering every result in the second fragment.
///
/// Inputs in the second fragment are remapped by the same offset, so its internal dataflow is
/// preserved and no accidental capture of first-fragment IDs occurs. Cross-fragment value
/// references are not part of this deliberately small composition API.
///
/// # Errors
///
/// Returns an error when fragment identity or version differs, output storage is too small,
/// value IDs overflow, or the second fragment has invalid local dataflow.
pub fn compose_dialect_fragments<'a>(
    first: &PcuDialectFragment<'a>,
    second: &PcuDialectFragment<'a>,
    output: &mut [PcuDialectOperation<'a>],
) -> Result<usize, PcuDialectComposeError> {
    if first.dialect != second.dialect {
        return Err(PcuDialectComposeError::IdentityMismatch);
    }
    if first.version != second.version {
        return Err(PcuDialectComposeError::VersionMismatch);
    }
    let count = first
        .operations
        .len()
        .checked_add(second.operations.len())
        .ok_or(PcuDialectComposeError::OutputTooSmall)?;
    if output.len() < count {
        return Err(PcuDialectComposeError::OutputTooSmall);
    }
    let offset = first
        .operations
        .iter()
        .filter_map(|op| op.result.map(|r| r.id.0))
        .max()
        .map_or(0u32, |id| u32::from(id) + 1);
    for (index, op) in second.operations.iter().enumerate() {
        if op.input_count as usize > op.inputs.len() {
            return Err(PcuDialectComposeError::InvalidFragmentFlow);
        }
        for input in &op.inputs[..op.input_count as usize] {
            let Some(definition) = second.operations[..index]
                .iter()
                .find_map(|prior| prior.result.filter(|result| result.id == input.id))
            else {
                return Err(PcuDialectComposeError::InvalidFragmentFlow);
            };
            if definition.value_type != input.value_type {
                return Err(PcuDialectComposeError::InvalidFragmentFlow);
            }
        }
        if let Some(result) = op.result
            && second.operations[..index].iter().any(|prior| {
                prior
                    .result
                    .is_some_and(|prior_result| prior_result.id == result.id)
            })
        {
            return Err(PcuDialectComposeError::InvalidFragmentFlow);
        }
        if op
            .result
            .into_iter()
            .chain(op.inputs[..op.input_count as usize].iter().copied())
            .any(|v| u32::from(v.id.0) + offset > u32::from(u16::MAX))
        {
            return Err(PcuDialectComposeError::ValueIdOverflow);
        }
    }
    output[..first.operations.len()].copy_from_slice(first.operations);
    for (index, op) in second.operations.iter().enumerate() {
        let mut remapped = *op;
        for input in &mut remapped.inputs[..usize::from(remapped.input_count)] {
            let remapped_id = u32::from(input.id.0) + offset;
            input.id = PcuDialectValueId(
                u16::try_from(remapped_id).map_err(|_| PcuDialectComposeError::ValueIdOverflow)?,
            );
        }
        if let Some(result) = &mut remapped.result {
            let remapped_id = u32::from(result.id.0) + offset;
            result.id = PcuDialectValueId(
                u16::try_from(remapped_id).map_err(|_| PcuDialectComposeError::ValueIdOverflow)?,
            );
        }
        output[first.operations.len() + index] = remapped;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn no_inputs() -> [PcuDialectOperand; 4] {
        [PcuDialectOperand::UNUSED; 4]
    }

    #[test]
    #[allow(clippy::too_many_lines)] // End-to-end fragment composition and rejection vector.
    fn vm_consumer_admits_typed_versioned_instructions_and_composes_without_capture() {
        let dialect = PcuDialectId("org.example.stack-vm");
        let add = PcuDialectOperation {
            name: "stack.add.u32",
            inputs: no_inputs(),
            input_count: 0,
            result: Some(PcuDialectOperand {
                id: PcuDialectValueId(0),
                value_type: PcuValueType::u32(),
            }),
            effects: PcuDialectEffects::PURE,
        };
        let store = PcuDialectOperation {
            name: "stack.store.u32",
            inputs: [
                PcuDialectOperand {
                    id: PcuDialectValueId(0),
                    value_type: PcuValueType::u32(),
                },
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
            ],
            input_count: 1,
            result: None,
            effects: PcuDialectEffects::WRITE_MEMORY,
        };
        let a_ops = [add];
        let b_ops = [store];
        let a = PcuDialectFragment {
            dialect,
            version: PcuDialectVersion { major: 1, minor: 2 },
            operations: &a_ops,
        };
        let b = PcuDialectFragment {
            dialect,
            version: a.version,
            operations: &b_ops,
        };
        let u32_type = [PcuValueType::u32()];
        let specs = [
            PcuDialectOperationSpec {
                name: "stack.add.u32",
                input_types: &[],
                result_type: Some(PcuValueType::u32()),
                effects: PcuDialectEffects::PURE,
            },
            PcuDialectOperationSpec {
                name: "stack.store.u32",
                input_types: &u32_type,
                result_type: None,
                effects: PcuDialectEffects::WRITE_MEMORY,
            },
        ];
        let vm = PcuDialectSupport {
            dialect,
            major: 1,
            min_minor: 1,
            max_minor: 3,
            operations: &specs,
            effects: PcuDialectEffects::WRITE_MEMORY,
        };
        assert_eq!(validate_dialect_fragment(&a, &vm), Ok(()));
        assert_eq!(
            validate_dialect_fragment(&b, &vm),
            Err(PcuDialectValidationError::UseBeforeDefinition(
                PcuDialectValueId(0)
            ))
        );

        // A second self-contained fragment defines and then consumes its own value. Composition
        // moves that local value namespace beyond the first fragment's namespace.
        let producer = PcuDialectOperation {
            name: "stack.add.u32",
            inputs: no_inputs(),
            input_count: 0,
            result: Some(PcuDialectOperand {
                id: PcuDialectValueId(0),
                value_type: PcuValueType::u32(),
            }),
            effects: PcuDialectEffects::PURE,
        };
        let consumer = PcuDialectOperation {
            name: "stack.store.u32",
            inputs: [
                PcuDialectOperand {
                    id: PcuDialectValueId(0),
                    value_type: PcuValueType::u32(),
                },
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
            ],
            input_count: 1,
            result: None,
            effects: PcuDialectEffects::WRITE_MEMORY,
        };
        let second_ops = [producer, consumer];
        let second = PcuDialectFragment {
            operations: &second_ops,
            ..b
        };
        let mut composed = [add; 3];
        assert_eq!(compose_dialect_fragments(&a, &second, &mut composed), Ok(3));
        assert_eq!(composed[1].result.unwrap().id, PcuDialectValueId(1));
        assert_eq!(composed[2].inputs[0].id, PcuDialectValueId(1));
        let composed_fragment = PcuDialectFragment {
            operations: &composed,
            ..a
        };
        assert_eq!(validate_dialect_fragment(&composed_fragment, &vm), Ok(()));
    }

    #[test]
    fn stream_consumer_uses_same_open_protocol_and_checks_effect_contract() {
        let dialect = PcuDialectId("org.example.stream-extension");
        let op = PcuDialectOperation {
            name: "stream.rotate.u32",
            inputs: no_inputs(),
            input_count: 0,
            result: Some(PcuDialectOperand {
                id: PcuDialectValueId(7),
                value_type: PcuValueType::u32(),
            }),
            effects: PcuDialectEffects::PURE,
        };
        let operations = [op];
        let fragment = PcuDialectFragment {
            dialect,
            version: PcuDialectVersion { major: 2, minor: 0 },
            operations: &operations,
        };
        let stream_spec = [PcuDialectOperationSpec {
            name: "stream.rotate.u32",
            input_types: &[],
            result_type: Some(PcuValueType::u32()),
            effects: PcuDialectEffects::PURE,
        }];
        let stream = PcuDialectSupport {
            dialect,
            major: 2,
            min_minor: 0,
            max_minor: 0,
            operations: &stream_spec,
            effects: PcuDialectEffects::PURE,
        };
        assert_eq!(validate_dialect_fragment(&fragment, &stream), Ok(()));
        let denied = PcuDialectSupport {
            effects: PcuDialectEffects::READ_MEMORY,
            ..stream
        };
        assert_eq!(validate_dialect_fragment(&fragment, &denied), Ok(()));
        let mut writes = op;
        writes.effects = PcuDialectEffects::WRITE_MEMORY;
        let bad_operations = [writes];
        let bad_fragment = PcuDialectFragment {
            operations: &bad_operations,
            ..fragment
        };
        assert_eq!(
            validate_dialect_fragment(&bad_fragment, &stream),
            Err(PcuDialectValidationError::UnsupportedEffects)
        );
    }

    #[test]
    fn rejects_signature_lies_and_malformed_input_counts() {
        let dialect = PcuDialectId("org.example.check");
        let operation = PcuDialectOperation {
            name: "vm.load",
            inputs: no_inputs(),
            input_count: 0,
            result: Some(PcuDialectOperand {
                id: PcuDialectValueId(0),
                value_type: PcuValueType::u32(),
            }),
            effects: PcuDialectEffects::READ_MEMORY,
        };
        let operations = [operation];
        let fragment = PcuDialectFragment {
            dialect,
            version: PcuDialectVersion { major: 1, minor: 0 },
            operations: &operations,
        };
        let spec = [PcuDialectOperationSpec {
            name: "vm.load",
            input_types: &[],
            result_type: Some(PcuValueType::u64()),
            effects: PcuDialectEffects::READ_MEMORY,
        }];
        let consumer = PcuDialectSupport {
            dialect,
            major: 1,
            min_minor: 0,
            max_minor: 0,
            operations: &spec,
            effects: PcuDialectEffects::READ_MEMORY,
        };
        assert_eq!(
            validate_dialect_fragment(&fragment, &consumer),
            Err(PcuDialectValidationError::SignatureMismatch)
        );
        let mut malformed = operation;
        malformed.input_count = u8::MAX;
        let malformed_ops = [malformed];
        let malformed_fragment = PcuDialectFragment {
            operations: &malformed_ops,
            ..fragment
        };
        assert_eq!(
            validate_dialect_fragment(&malformed_fragment, &consumer),
            Err(PcuDialectValidationError::UnsupportedOperation)
        );
    }

    #[test]
    fn composition_refuses_unbound_cross_fragment_references() {
        let dialect = PcuDialectId("org.example.compose");
        let producer = PcuDialectOperation {
            name: "produce",
            inputs: no_inputs(),
            input_count: 0,
            result: Some(PcuDialectOperand {
                id: PcuDialectValueId(0),
                value_type: PcuValueType::u32(),
            }),
            effects: PcuDialectEffects::PURE,
        };
        let first_ops = [producer];
        let first = PcuDialectFragment {
            dialect,
            version: PcuDialectVersion { major: 1, minor: 0 },
            operations: &first_ops,
        };
        let dependent = PcuDialectOperation {
            name: "consume",
            inputs: [
                PcuDialectOperand {
                    id: PcuDialectValueId(0),
                    value_type: PcuValueType::u32(),
                },
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
            ],
            input_count: 1,
            result: None,
            effects: PcuDialectEffects::PURE,
        };
        let second_ops = [dependent];
        let second = PcuDialectFragment {
            operations: &second_ops,
            ..first
        };
        let mut output = [producer; 2];
        assert_eq!(
            compose_dialect_fragments(&first, &second, &mut output),
            Err(PcuDialectComposeError::InvalidFragmentFlow)
        );
    }
}
