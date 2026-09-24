//! Open, versioned extension vocabulary for backend-specific PCU instructions.
//!
//! This module describes instruction identity, typed inputs/results, and declared effects. It
//! deliberately does not assign executable meaning to an operation name. A backend may admit a
//! dialect only when its consumer-specific support table recognizes that exact operation and
//! version.

use crate::{
    PcuScalarType,
    PcuValueType,
};

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

/// One named program-boundary value contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDialectPortSpec<'a> {
    pub name: &'a str,
    pub value_type: PcuValueType,
}

/// One named program-boundary binding to a local value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDialectPort<'a> {
    pub name: &'a str,
    pub value: PcuDialectOperand,
}

/// Scalar literal stored using its exact typed representation.
///
/// Floating-point variants carry IEEE bit patterns; this keeps the representation deterministic
/// for `no_std` consumers and preserves NaN payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuDialectImmediateValue {
    Bool(bool),
    I4(i8),
    U4(u8),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F16(u16),
    BF16(u16),
    F32(u32),
    F64(u64),
}

impl PcuDialectImmediateValue {
    const fn scalar_type(self) -> PcuScalarType {
        match self {
            Self::Bool(_) => PcuScalarType::Bool,
            Self::I4(_) => PcuScalarType::I4,
            Self::U4(_) => PcuScalarType::U4,
            Self::I8(_) => PcuScalarType::I8,
            Self::U8(_) => PcuScalarType::U8,
            Self::I16(_) => PcuScalarType::I16,
            Self::U16(_) => PcuScalarType::U16,
            Self::I32(_) => PcuScalarType::I32,
            Self::U32(_) => PcuScalarType::U32,
            Self::I64(_) => PcuScalarType::I64,
            Self::U64(_) => PcuScalarType::U64,
            Self::F16(_) => PcuScalarType::F16,
            Self::BF16(_) => PcuScalarType::BF16,
            Self::F32(_) => PcuScalarType::F32,
            Self::F64(_) => PcuScalarType::F64,
        }
    }

    const fn has_valid_encoding(self) -> bool {
        match self {
            Self::I4(value) => value >= -8 && value <= 7,
            Self::U4(value) => value <= 15,
            _ => true,
        }
    }
}

/// A named, typed immediate attribute on one external operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDialectImmediate<'a> {
    pub name: &'a str,
    pub value: PcuDialectImmediateValue,
}

/// Explicit immediate metadata for one operation index in a program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuDialectOperationAttributes<'a> {
    pub operation_index: u16,
    pub attributes: &'a [PcuDialectImmediate<'a>],
}

/// Program-level ports and immediate attributes around an existing local fragment.
///
/// Input port IDs are initial typed definitions. Output ports name values defined by those inputs
/// or by operations. This wrapper leaves the source-compatible fragment and operation literals
/// unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuDialectProgram<'a> {
    pub fragment: PcuDialectFragment<'a>,
    pub input_ports: &'a [PcuDialectPort<'a>],
    pub output_ports: &'a [PcuDialectPort<'a>],
    /// Exactly one entry per operation, identified by its explicit zero-based index.
    pub operation_attributes: &'a [PcuDialectOperationAttributes<'a>],
}

/// Consumer-owned immediate requirement for one operation attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDialectImmediateSpec<'a> {
    pub name: &'a str,
    pub scalar_type: PcuScalarType,
}

/// Consumer-owned immediate schema for one operation name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuDialectOperationImmediateSpec<'a> {
    pub operation_name: &'a str,
    pub attributes: &'a [PcuDialectImmediateSpec<'a>],
}

/// Consumer-owned whole-program contract layered over operation support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuDialectProgramSupport<'a> {
    pub dialect_support: PcuDialectSupport<'a>,
    pub input_ports: &'a [PcuDialectPortSpec<'a>],
    pub output_ports: &'a [PcuDialectPortSpec<'a>],
    pub operation_immediates: &'a [PcuDialectOperationImmediateSpec<'a>],
}

/// An explicit connection from one first-program output to one second-program input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDialectProgramBinding<'a> {
    pub first_output: &'a str,
    pub second_input: &'a str,
}

/// The three consumer contracts checked while composing a program pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuDialectProgramComposeSupports<'a> {
    pub first: &'a PcuDialectProgramSupport<'a>,
    pub second: &'a PcuDialectProgramSupport<'a>,
    pub composed: &'a PcuDialectProgramSupport<'a>,
}

/// Caller-owned storage for the composed operations and program boundary metadata.
pub struct PcuDialectProgramComposeStorage<'out, 'data: 'out> {
    pub operations: &'out mut [PcuDialectOperation<'data>],
    pub input_ports: &'out mut [PcuDialectPort<'data>],
    pub output_ports: &'out mut [PcuDialectPort<'data>],
    pub operation_attributes: &'out mut [PcuDialectOperationAttributes<'data>],
}

/// Failure while explicitly composing two port-bearing programs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuDialectProgramComposeError {
    FirstProgramInvalid(PcuDialectValidationError),
    SecondProgramInvalid(PcuDialectValidationError),
    ComposedSupportInvalid(PcuDialectValidationError),
    IdentityMismatch,
    VersionMismatch,
    MissingBinding,
    DuplicateBinding,
    UnknownBindingPort,
    BindingTypeMismatch,
    OperationStorageTooSmall,
    InputPortStorageTooSmall,
    OutputPortStorageTooSmall,
    AttributeStorageTooSmall,
    ValueIdOverflow,
    OperationIndexOverflow,
}

/// Compose two programs as a straight-line pipeline using explicit named port bindings.
///
/// Every second-program input must be bound exactly once to a first-program output. One first
/// output may feed multiple second inputs. The composed public inputs are the first program's
/// inputs, and the composed public outputs are the second program's outputs. A second output that
/// forwards one of its input values therefore forwards the corresponding first output value.
/// First outputs not exposed by the second program are dropped from the composed boundary.
///
/// Second-program input IDs are replaced by the IDs of their explicitly connected first outputs.
/// Other second-program definitions are shifted above the first program's input and result IDs.
/// Attribute metadata retains first-operation indexes and shifts second-operation indexes by the
/// number of first operations. No operation semantics or control flow are inferred.
///
/// The first, second, and composed programs are checked against their respective support schemas
/// before any output buffer is written. Capacity and remapping bounds are also preflighted, so an
/// error leaves all output buffers unchanged.
///
/// # Errors
///
/// Returns an error for invalid source programs, incompatible identities or versions, missing,
/// duplicate, unknown, or mistyped bindings, a composed support mismatch, insufficient output
/// capacity, or an ID/index overflow.
pub fn compose_dialect_programs<'data: 'out, 'out>(
    first: &PcuDialectProgram<'data>,
    second: &PcuDialectProgram<'data>,
    bindings: &[PcuDialectProgramBinding<'_>],
    supports: PcuDialectProgramComposeSupports<'_>,
    storage: PcuDialectProgramComposeStorage<'out, 'data>,
) -> Result<PcuDialectProgram<'out>, PcuDialectProgramComposeError> {
    let PcuDialectProgramComposeStorage {
        operations: output_operations,
        input_ports: output_inputs,
        output_ports: output_outputs,
        operation_attributes: output_attributes,
    } = storage;
    let plan = preflight_program_composition(
        first,
        second,
        bindings,
        supports,
        ComposeCapacities {
            operations: output_operations.len(),
            inputs: output_inputs.len(),
            outputs: output_outputs.len(),
            attributes: output_attributes.len(),
        },
    )?;

    let first_operation_count = first.fragment.operations.len();
    let operation_count = first_operation_count + second.fragment.operations.len();
    let input_count = first.input_ports.len();
    let output_count = second.output_ports.len();
    let attribute_count = first.operation_attributes.len() + second.operation_attributes.len();
    let value_offset = plan.value_offset;
    let second_index_offset = plan.second_index_offset;

    output_operations[..first_operation_count].copy_from_slice(first.fragment.operations);
    for (index, operation) in second.fragment.operations.iter().enumerate() {
        let mut mapped = *operation;
        for input in &mut mapped.inputs[..usize::from(mapped.input_count)] {
            input.id = remap_second_value(input.id, second, first, bindings, value_offset);
        }
        if let Some(result) = &mut mapped.result {
            result.id = shifted_value_id(result.id, value_offset);
        }
        output_operations[first_operation_count + index] = mapped;
    }
    output_inputs[..input_count].copy_from_slice(first.input_ports);
    for (index, port) in second.output_ports.iter().enumerate() {
        let mut mapped = *port;
        mapped.value.id =
            remap_second_value(mapped.value.id, second, first, bindings, value_offset);
        output_outputs[index] = mapped;
    }
    output_attributes[..first.operation_attributes.len()]
        .copy_from_slice(first.operation_attributes);
    for (index, attributes) in second.operation_attributes.iter().enumerate() {
        output_attributes[first.operation_attributes.len() + index] =
            PcuDialectOperationAttributes {
                operation_index: attributes.operation_index + second_index_offset,
                attributes: attributes.attributes,
            };
    }

    Ok(PcuDialectProgram {
        fragment: PcuDialectFragment {
            dialect: first.fragment.dialect,
            version: first.fragment.version,
            operations: &output_operations[..operation_count],
        },
        input_ports: &output_inputs[..input_count],
        output_ports: &output_outputs[..output_count],
        operation_attributes: &output_attributes[..attribute_count],
    })
}

fn preflight_program_composition(
    first: &PcuDialectProgram<'_>,
    second: &PcuDialectProgram<'_>,
    bindings: &[PcuDialectProgramBinding<'_>],
    supports: PcuDialectProgramComposeSupports<'_>,
    capacities: ComposeCapacities,
) -> Result<CompositionPlan, PcuDialectProgramComposeError> {
    validate_dialect_program(first, supports.first)
        .map_err(PcuDialectProgramComposeError::FirstProgramInvalid)?;
    validate_dialect_program(second, supports.second)
        .map_err(PcuDialectProgramComposeError::SecondProgramInvalid)?;
    if first.fragment.dialect != second.fragment.dialect {
        return Err(PcuDialectProgramComposeError::IdentityMismatch);
    }
    if first.fragment.version != second.fragment.version {
        return Err(PcuDialectProgramComposeError::VersionMismatch);
    }

    validate_program_bindings(first, second, bindings)?;
    validate_composed_program_support(first, second, supports.composed)?;

    let operation_count = first
        .fragment
        .operations
        .len()
        .checked_add(second.fragment.operations.len())
        .ok_or(PcuDialectProgramComposeError::OperationStorageTooSmall)?;
    let attribute_count = first
        .operation_attributes
        .len()
        .checked_add(second.operation_attributes.len())
        .ok_or(PcuDialectProgramComposeError::AttributeStorageTooSmall)?;
    if capacities.operations < operation_count {
        return Err(PcuDialectProgramComposeError::OperationStorageTooSmall);
    }
    if capacities.inputs < first.input_ports.len() {
        return Err(PcuDialectProgramComposeError::InputPortStorageTooSmall);
    }
    if capacities.outputs < second.output_ports.len() {
        return Err(PcuDialectProgramComposeError::OutputPortStorageTooSmall);
    }
    if capacities.attributes < attribute_count {
        return Err(PcuDialectProgramComposeError::AttributeStorageTooSmall);
    }

    let second_index_offset = if second.operation_attributes.is_empty() {
        0
    } else {
        u16::try_from(first.fragment.operations.len())
            .map_err(|_| PcuDialectProgramComposeError::OperationIndexOverflow)?
    };
    for attributes in second.operation_attributes {
        attributes
            .operation_index
            .checked_add(second_index_offset)
            .ok_or(PcuDialectProgramComposeError::OperationIndexOverflow)?;
    }
    let value_offset = next_program_value_id(first);
    validate_composition_value_remaps(second, bindings, value_offset)?;
    Ok(CompositionPlan {
        value_offset,
        second_index_offset,
    })
}

#[derive(Clone, Copy)]
struct ComposeCapacities {
    operations: usize,
    inputs: usize,
    outputs: usize,
    attributes: usize,
}

struct CompositionPlan {
    value_offset: u32,
    second_index_offset: u16,
}

fn validate_program_bindings(
    first: &PcuDialectProgram<'_>,
    second: &PcuDialectProgram<'_>,
    bindings: &[PcuDialectProgramBinding<'_>],
) -> Result<(), PcuDialectProgramComposeError> {
    for (index, binding) in bindings.iter().enumerate() {
        let Some(source) = first
            .output_ports
            .iter()
            .find(|port| port.name == binding.first_output)
        else {
            return Err(PcuDialectProgramComposeError::UnknownBindingPort);
        };
        let Some(target) = second
            .input_ports
            .iter()
            .find(|port| port.name == binding.second_input)
        else {
            return Err(PcuDialectProgramComposeError::UnknownBindingPort);
        };
        if binding.first_output.is_empty() || binding.second_input.is_empty() {
            return Err(PcuDialectProgramComposeError::UnknownBindingPort);
        }
        if bindings[..index]
            .iter()
            .any(|prior| prior.second_input == binding.second_input)
        {
            return Err(PcuDialectProgramComposeError::DuplicateBinding);
        }
        if source.value.value_type != target.value.value_type {
            return Err(PcuDialectProgramComposeError::BindingTypeMismatch);
        }
    }
    for input in second.input_ports {
        let matches = bindings
            .iter()
            .filter(|binding| binding.second_input == input.name)
            .count();
        if matches == 0 {
            return Err(PcuDialectProgramComposeError::MissingBinding);
        }
        if matches > 1 {
            return Err(PcuDialectProgramComposeError::DuplicateBinding);
        }
    }
    if bindings.len() != second.input_ports.len() {
        return Err(PcuDialectProgramComposeError::UnknownBindingPort);
    }
    Ok(())
}

fn validate_composed_program_support(
    first: &PcuDialectProgram<'_>,
    second: &PcuDialectProgram<'_>,
    support: &PcuDialectProgramSupport<'_>,
) -> Result<(), PcuDialectProgramComposeError> {
    validate_port_schema(first.input_ports, support.input_ports)
        .map_err(PcuDialectProgramComposeError::ComposedSupportInvalid)?;
    validate_port_schema(second.output_ports, support.output_ports)
        .map_err(PcuDialectProgramComposeError::ComposedSupportInvalid)?;
    validate_program_support_schemas(support)
        .map_err(PcuDialectProgramComposeError::ComposedSupportInvalid)?;
    validate_dialect_identity_and_operations(&first.fragment, &support.dialect_support)
        .map_err(PcuDialectProgramComposeError::ComposedSupportInvalid)?;
    validate_dialect_identity_and_operations(&second.fragment, &support.dialect_support)
        .map_err(PcuDialectProgramComposeError::ComposedSupportInvalid)?;
    validate_program_immediates_against_support(first, support)
        .map_err(PcuDialectProgramComposeError::ComposedSupportInvalid)?;
    validate_program_immediates_against_support(second, support)
        .map_err(PcuDialectProgramComposeError::ComposedSupportInvalid)
}

fn validate_program_immediates_against_support(
    program: &PcuDialectProgram<'_>,
    support: &PcuDialectProgramSupport<'_>,
) -> Result<(), PcuDialectValidationError> {
    for (index, operation) in program.fragment.operations.iter().enumerate() {
        let operation_index = u16::try_from(index)
            .map_err(|_| PcuDialectValidationError::OperationIndexOutOfRange(u16::MAX))?;
        let attributes = program
            .operation_attributes
            .iter()
            .find(|entry| entry.operation_index == operation_index)
            .ok_or(PcuDialectValidationError::MissingOperationAttributes(
                operation_index,
            ))?;
        let Some(schema) = support
            .operation_immediates
            .iter()
            .find(|schema| schema.operation_name == operation.name)
        else {
            if attributes.attributes.is_empty() {
                continue;
            }
            return Err(PcuDialectValidationError::UnknownImmediate);
        };
        validate_immediates(attributes.attributes, schema.attributes)?;
    }
    Ok(())
}

fn validate_composition_value_remaps(
    second: &PcuDialectProgram<'_>,
    bindings: &[PcuDialectProgramBinding<'_>],
    offset: u32,
) -> Result<(), PcuDialectProgramComposeError> {
    for input in second.input_ports {
        if second_input_is_bound(input.name, bindings) {
            continue;
        }
        check_shifted_value_id(input.value.id, offset)?;
    }
    for operation in second.fragment.operations {
        if let Some(result) = operation.result {
            check_shifted_value_id(result.id, offset)?;
        }
        for input in &operation.inputs[..usize::from(operation.input_count)] {
            if !second_value_is_input(input.id, second)
                || !second_input_is_bound(
                    second
                        .input_ports
                        .iter()
                        .find(|port| port.value.id == input.id)
                        .map_or("", |port| port.name),
                    bindings,
                )
            {
                check_shifted_value_id(input.id, offset)?;
            }
        }
    }
    for output in second.output_ports {
        if !second_value_is_input(output.value.id, second)
            || !second_input_is_bound(
                second
                    .input_ports
                    .iter()
                    .find(|port| port.value.id == output.value.id)
                    .map_or("", |port| port.name),
                bindings,
            )
        {
            check_shifted_value_id(output.value.id, offset)?;
        }
    }
    Ok(())
}

fn next_program_value_id(program: &PcuDialectProgram<'_>) -> u32 {
    program
        .input_ports
        .iter()
        .map(|port| port.value.id.0)
        .chain(
            program
                .fragment
                .operations
                .iter()
                .filter_map(|operation| operation.result.map(|result| result.id.0)),
        )
        .max()
        .map_or(0, |id| u32::from(id) + 1)
}

fn check_shifted_value_id(
    id: PcuDialectValueId,
    offset: u32,
) -> Result<(), PcuDialectProgramComposeError> {
    if u32::from(id.0) + offset > u32::from(u16::MAX) {
        return Err(PcuDialectProgramComposeError::ValueIdOverflow);
    }
    Ok(())
}

fn shifted_value_id(id: PcuDialectValueId, offset: u32) -> PcuDialectValueId {
    // Preflight checks every shifted definition and reference before output buffers are written.
    let shifted = u16::try_from(u32::from(id.0) + offset).unwrap_or(u16::MAX);
    PcuDialectValueId(shifted)
}

fn second_input_is_bound(name: &str, bindings: &[PcuDialectProgramBinding<'_>]) -> bool {
    bindings.iter().any(|binding| binding.second_input == name)
}

fn second_value_is_input(id: PcuDialectValueId, program: &PcuDialectProgram<'_>) -> bool {
    program.input_ports.iter().any(|port| port.value.id == id)
}

fn remap_second_value(
    id: PcuDialectValueId,
    second: &PcuDialectProgram<'_>,
    first: &PcuDialectProgram<'_>,
    bindings: &[PcuDialectProgramBinding<'_>],
    offset: u32,
) -> PcuDialectValueId {
    let Some(input) = second.input_ports.iter().find(|port| port.value.id == id) else {
        return shifted_value_id(id, offset);
    };
    let Some(binding) = bindings
        .iter()
        .find(|binding| binding.second_input == input.name)
    else {
        return shifted_value_id(id, offset);
    };
    first
        .output_ports
        .iter()
        .find(|port| port.name == binding.first_output)
        .map_or_else(|| shifted_value_id(id, offset), |port| port.value.id)
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
    EmptyPortName,
    DuplicatePortName,
    DuplicatePortId(PcuDialectValueId),
    MissingPort,
    UnknownPort,
    PortTypeMismatch,
    MissingOutputValue(PcuDialectValueId),
    MissingOperationAttributes(u16),
    DuplicateOperationAttributes(u16),
    OperationIndexOutOfRange(u16),
    EmptyImmediateName,
    DuplicateImmediateName,
    DuplicateImmediateOperationSchema,
    MissingImmediate,
    UnknownImmediate,
    ImmediateTypeMismatch,
    InvalidImmediateEncoding,
}

/// Validate a whole open dialect program against consumer-owned ports and immediate schemas.
///
/// Ports are matched by name and exact type. Inputs seed local value definitions, while outputs
/// must refer to a value available after the operation sequence. Every operation has one
/// explicitly indexed attribute set, including operations with no attributes. This checks a
/// straight-line local dataflow fragment only; it does not define control-flow or region meaning.
///
/// # Errors
///
/// Returns the first mismatch in dialect support, ports, operation flow, attribute metadata, or
/// the consumer-owned schemas.
pub fn validate_dialect_program(
    program: &PcuDialectProgram<'_>,
    support: &PcuDialectProgramSupport<'_>,
) -> Result<(), PcuDialectValidationError> {
    validate_port_schema(program.input_ports, support.input_ports)?;
    validate_port_schema(program.output_ports, support.output_ports)?;
    validate_dialect_identity_and_operations(&program.fragment, &support.dialect_support)?;
    validate_program_support_schemas(support)?;
    validate_program_input_ids(program)?;
    validate_program_operations(program, support)?;
    validate_program_dataflow(program)
}

fn validate_program_support_schemas(
    support: &PcuDialectProgramSupport<'_>,
) -> Result<(), PcuDialectValidationError> {
    for (index, schema) in support.operation_immediates.iter().enumerate() {
        if schema.operation_name.is_empty() {
            return Err(PcuDialectValidationError::EmptyOperationName);
        }
        if support.operation_immediates[..index]
            .iter()
            .any(|prior| prior.operation_name == schema.operation_name)
        {
            return Err(PcuDialectValidationError::DuplicateImmediateOperationSchema);
        }
        if !support
            .dialect_support
            .operations
            .iter()
            .any(|op| op.name == schema.operation_name)
        {
            return Err(PcuDialectValidationError::UnsupportedOperation);
        }
        for (attribute_index, attribute) in schema.attributes.iter().enumerate() {
            if attribute.name.is_empty() {
                return Err(PcuDialectValidationError::EmptyImmediateName);
            }
            if schema.attributes[..attribute_index]
                .iter()
                .any(|prior| prior.name == attribute.name)
            {
                return Err(PcuDialectValidationError::DuplicateImmediateName);
            }
        }
    }
    Ok(())
}

fn validate_program_input_ids(
    program: &PcuDialectProgram<'_>,
) -> Result<(), PcuDialectValidationError> {
    for port in program.input_ports {
        if program
            .input_ports
            .iter()
            .filter(|other| other.value.id == port.value.id)
            .count()
            > 1
        {
            return Err(PcuDialectValidationError::DuplicatePortId(port.value.id));
        }
        if program
            .fragment
            .operations
            .iter()
            .any(|op| op.result.is_some_and(|result| result.id == port.value.id))
        {
            return Err(PcuDialectValidationError::DuplicateResult(port.value.id));
        }
    }
    Ok(())
}

fn validate_program_operations(
    program: &PcuDialectProgram<'_>,
    support: &PcuDialectProgramSupport<'_>,
) -> Result<(), PcuDialectValidationError> {
    for (index, operation) in program.fragment.operations.iter().enumerate() {
        if let Some(result) = operation.result
            && (program.fragment.operations[..index].iter().any(|prior| {
                prior
                    .result
                    .is_some_and(|prior_result| prior_result.id == result.id)
            }) || operation.inputs[..usize::from(operation.input_count)]
                .iter()
                .any(|input| input.id == result.id))
        {
            return Err(PcuDialectValidationError::DuplicateResult(result.id));
        }
        let op_index = u16::try_from(index)
            .map_err(|_| PcuDialectValidationError::OperationIndexOutOfRange(u16::MAX))?;
        let Some(attributes) = program
            .operation_attributes
            .iter()
            .find(|entry| entry.operation_index == op_index)
        else {
            return Err(PcuDialectValidationError::MissingOperationAttributes(
                op_index,
            ));
        };
        if program
            .operation_attributes
            .iter()
            .filter(|entry| entry.operation_index == op_index)
            .count()
            > 1
        {
            return Err(PcuDialectValidationError::DuplicateOperationAttributes(
                op_index,
            ));
        }

        let Some(op_schema) = support
            .operation_immediates
            .iter()
            .find(|schema| schema.operation_name == operation.name)
        else {
            if attributes.attributes.is_empty() {
                continue;
            }
            return Err(PcuDialectValidationError::UnknownImmediate);
        };
        validate_immediates(attributes.attributes, op_schema.attributes)?;
    }
    for entry in program.operation_attributes {
        let index = usize::from(entry.operation_index);
        if index >= program.fragment.operations.len() {
            return Err(PcuDialectValidationError::OperationIndexOutOfRange(
                entry.operation_index,
            ));
        }
    }
    if program.operation_attributes.len() != program.fragment.operations.len() {
        for index in 0..program.fragment.operations.len() {
            let op_index = u16::try_from(index)
                .map_err(|_| PcuDialectValidationError::OperationIndexOutOfRange(u16::MAX))?;
            if !program
                .operation_attributes
                .iter()
                .any(|entry| entry.operation_index == op_index)
            {
                return Err(PcuDialectValidationError::MissingOperationAttributes(
                    op_index,
                ));
            }
        }
    }
    Ok(())
}

fn validate_program_dataflow(
    program: &PcuDialectProgram<'_>,
) -> Result<(), PcuDialectValidationError> {
    for (index, operation) in program.fragment.operations.iter().enumerate() {
        for input in &operation.inputs[..usize::from(operation.input_count)] {
            let defined = program
                .input_ports
                .iter()
                .any(|port| port.value.id == input.id)
                || program.fragment.operations[..index]
                    .iter()
                    .any(|prior| prior.result.is_some_and(|result| result.id == input.id));
            if !defined {
                return Err(PcuDialectValidationError::UseBeforeDefinition(input.id));
            }
            let found_type = program
                .input_ports
                .iter()
                .find(|port| port.value.id == input.id)
                .map(|port| port.value.value_type)
                .or_else(|| {
                    program.fragment.operations[..index]
                        .iter()
                        .find_map(|prior| {
                            prior
                                .result
                                .filter(|result| result.id == input.id)
                                .map(|result| result.value_type)
                        })
                });
            if found_type != Some(input.value_type) {
                return Err(PcuDialectValidationError::OperandTypeMismatch(input.id));
            }
        }
    }
    for port in program.output_ports {
        let Some(found_type) = program
            .input_ports
            .iter()
            .find(|input| input.value.id == port.value.id)
            .map(|input| input.value.value_type)
            .or_else(|| {
                program.fragment.operations.iter().find_map(|op| {
                    op.result
                        .filter(|result| result.id == port.value.id)
                        .map(|result| result.value_type)
                })
            })
        else {
            return Err(PcuDialectValidationError::MissingOutputValue(port.value.id));
        };
        if found_type != port.value.value_type {
            return Err(PcuDialectValidationError::PortTypeMismatch);
        }
    }
    Ok(())
}

fn validate_dialect_identity_and_operations(
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
    for op in fragment.operations {
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
    }
    Ok(())
}

fn validate_port_schema(
    ports: &[PcuDialectPort<'_>],
    schema: &[PcuDialectPortSpec<'_>],
) -> Result<(), PcuDialectValidationError> {
    for (index, expected) in schema.iter().enumerate() {
        if expected.name.is_empty() {
            return Err(PcuDialectValidationError::EmptyPortName);
        }
        if schema[..index]
            .iter()
            .any(|prior| prior.name == expected.name)
        {
            return Err(PcuDialectValidationError::DuplicatePortName);
        }
    }
    for (index, port) in ports.iter().enumerate() {
        if port.name.is_empty() {
            return Err(PcuDialectValidationError::EmptyPortName);
        }
        if ports[..index].iter().any(|prior| prior.name == port.name) {
            return Err(PcuDialectValidationError::DuplicatePortName);
        }
        if ports[..index]
            .iter()
            .any(|prior| prior.value.id == port.value.id)
        {
            return Err(PcuDialectValidationError::DuplicatePortId(port.value.id));
        }
        let Some(expected) = schema.iter().find(|expected| expected.name == port.name) else {
            return Err(PcuDialectValidationError::UnknownPort);
        };
        if expected.value_type != port.value.value_type {
            return Err(PcuDialectValidationError::PortTypeMismatch);
        }
    }
    for expected in schema {
        if !ports.iter().any(|port| port.name == expected.name) {
            return Err(PcuDialectValidationError::MissingPort);
        }
    }
    Ok(())
}

fn validate_immediates(
    attributes: &[PcuDialectImmediate<'_>],
    schema: &[PcuDialectImmediateSpec<'_>],
) -> Result<(), PcuDialectValidationError> {
    for (index, expected) in schema.iter().enumerate() {
        if expected.name.is_empty() {
            return Err(PcuDialectValidationError::EmptyImmediateName);
        }
        if schema[..index]
            .iter()
            .any(|prior| prior.name == expected.name)
        {
            return Err(PcuDialectValidationError::DuplicateImmediateName);
        }
    }
    for (index, attribute) in attributes.iter().enumerate() {
        if attribute.name.is_empty() {
            return Err(PcuDialectValidationError::EmptyImmediateName);
        }
        if attributes[..index]
            .iter()
            .any(|prior| prior.name == attribute.name)
        {
            return Err(PcuDialectValidationError::DuplicateImmediateName);
        }
        let Some(expected) = schema
            .iter()
            .find(|expected| expected.name == attribute.name)
        else {
            return Err(PcuDialectValidationError::UnknownImmediate);
        };
        if expected.scalar_type != attribute.value.scalar_type() {
            return Err(PcuDialectValidationError::ImmediateTypeMismatch);
        }
        if !attribute.value.has_valid_encoding() {
            return Err(PcuDialectValidationError::InvalidImmediateEncoding);
        }
    }
    for expected in schema {
        if !attributes
            .iter()
            .any(|attribute| attribute.name == expected.name)
        {
            return Err(PcuDialectValidationError::MissingImmediate);
        }
    }
    Ok(())
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
/// references are not part of this deliberately small composition API. This function composes
/// operations only; it does not compose `PcuDialectProgram` ports or immediate metadata.
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

    #[test]
    #[allow(clippy::too_many_lines)] // Validates one accepted program and its rejection cases together.
    fn program_verifier_checks_named_ports_and_typed_immediates() {
        let dialect = PcuDialectId("org.example.math");
        let operation = PcuDialectOperation {
            name: "scale",
            inputs: [
                PcuDialectOperand {
                    id: PcuDialectValueId(7),
                    value_type: PcuValueType::u32(),
                },
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
            ],
            input_count: 1,
            result: Some(PcuDialectOperand {
                id: PcuDialectValueId(8),
                value_type: PcuValueType::u32(),
            }),
            effects: PcuDialectEffects::PURE,
        };
        let operations = [operation];
        let fragment = PcuDialectFragment {
            dialect,
            version: PcuDialectVersion { major: 2, minor: 1 },
            operations: &operations,
        };
        let inputs = [PcuDialectPort {
            name: "source",
            value: PcuDialectOperand {
                id: PcuDialectValueId(7),
                value_type: PcuValueType::u32(),
            },
        }];
        let outputs = [PcuDialectPort {
            name: "scaled",
            value: PcuDialectOperand {
                id: PcuDialectValueId(8),
                value_type: PcuValueType::u32(),
            },
        }];
        let attributes = [PcuDialectImmediate {
            name: "factor",
            value: PcuDialectImmediateValue::U32(3),
        }];
        let operation_attributes = [PcuDialectOperationAttributes {
            operation_index: 0,
            attributes: &attributes,
        }];
        let program = PcuDialectProgram {
            fragment,
            input_ports: &inputs,
            output_ports: &outputs,
            operation_attributes: &operation_attributes,
        };
        let operation_specs = [PcuDialectOperationSpec {
            name: "scale",
            input_types: &[PcuValueType::u32()],
            result_type: Some(PcuValueType::u32()),
            effects: PcuDialectEffects::PURE,
        }];
        let dialect_support = PcuDialectSupport {
            dialect,
            major: 2,
            min_minor: 0,
            max_minor: 1,
            operations: &operation_specs,
            effects: PcuDialectEffects::PURE,
        };
        let input_specs = [PcuDialectPortSpec {
            name: "source",
            value_type: PcuValueType::u32(),
        }];
        let output_specs = [PcuDialectPortSpec {
            name: "scaled",
            value_type: PcuValueType::u32(),
        }];
        let immediate_specs = [PcuDialectImmediateSpec {
            name: "factor",
            scalar_type: PcuScalarType::U32,
        }];
        let op_immediate_specs = [PcuDialectOperationImmediateSpec {
            operation_name: "scale",
            attributes: &immediate_specs,
        }];
        let support = PcuDialectProgramSupport {
            dialect_support,
            input_ports: &input_specs,
            output_ports: &output_specs,
            operation_immediates: &op_immediate_specs,
        };

        assert_eq!(validate_dialect_program(&program, &support), Ok(()));

        let no_inputs = [];
        assert_eq!(
            validate_dialect_program(
                &PcuDialectProgram {
                    input_ports: &no_inputs,
                    ..program
                },
                &support
            ),
            Err(PcuDialectValidationError::MissingPort)
        );
        let invalid_output = [PcuDialectPort {
            value: PcuDialectOperand {
                id: PcuDialectValueId(99),
                ..outputs[0].value
            },
            ..outputs[0]
        }];
        assert_eq!(
            validate_dialect_program(
                &PcuDialectProgram {
                    output_ports: &invalid_output,
                    ..program
                },
                &support
            ),
            Err(PcuDialectValidationError::MissingOutputValue(
                PcuDialectValueId(99)
            ))
        );
        let duplicate_inputs = [
            inputs[0],
            PcuDialectPort {
                name: "source",
                ..inputs[0]
            },
        ];
        assert_eq!(
            validate_dialect_program(
                &PcuDialectProgram {
                    input_ports: &duplicate_inputs,
                    ..program
                },
                &support
            ),
            Err(PcuDialectValidationError::DuplicatePortName)
        );
        let wrong_input = [PcuDialectPort {
            value: PcuDialectOperand {
                value_type: PcuValueType::i32(),
                ..inputs[0].value
            },
            ..inputs[0]
        }];
        assert_eq!(
            validate_dialect_program(
                &PcuDialectProgram {
                    input_ports: &wrong_input,
                    ..program
                },
                &support
            ),
            Err(PcuDialectValidationError::PortTypeMismatch)
        );
        let no_attributes = [PcuDialectOperationAttributes {
            operation_index: 0,
            attributes: &[],
        }];
        assert_eq!(
            validate_dialect_program(
                &PcuDialectProgram {
                    operation_attributes: &no_attributes,
                    ..program
                },
                &support
            ),
            Err(PcuDialectValidationError::MissingImmediate)
        );
        let wrong_attributes = [PcuDialectImmediate {
            name: "factor",
            value: PcuDialectImmediateValue::I32(3),
        }];
        let wrong_operation_attributes = [PcuDialectOperationAttributes {
            operation_index: 0,
            attributes: &wrong_attributes,
        }];
        assert_eq!(
            validate_dialect_program(
                &PcuDialectProgram {
                    operation_attributes: &wrong_operation_attributes,
                    ..program
                },
                &support
            ),
            Err(PcuDialectValidationError::ImmediateTypeMismatch)
        );
        let duplicate_attributes = [attributes[0], attributes[0]];
        let duplicate_operation_attributes = [PcuDialectOperationAttributes {
            operation_index: 0,
            attributes: &duplicate_attributes,
        }];
        assert_eq!(
            validate_dialect_program(
                &PcuDialectProgram {
                    operation_attributes: &duplicate_operation_attributes,
                    ..program
                },
                &support
            ),
            Err(PcuDialectValidationError::DuplicateImmediateName)
        );
        let no_operation_attributes = [];
        assert_eq!(
            validate_dialect_program(
                &PcuDialectProgram {
                    operation_attributes: &no_operation_attributes,
                    ..program
                },
                &support
            ),
            Err(PcuDialectValidationError::MissingOperationAttributes(0))
        );
        let duplicate_operation_metadata = [operation_attributes[0], operation_attributes[0]];
        assert_eq!(
            validate_dialect_program(
                &PcuDialectProgram {
                    operation_attributes: &duplicate_operation_metadata,
                    ..program
                },
                &support
            ),
            Err(PcuDialectValidationError::DuplicateOperationAttributes(0))
        );
        let unknown_attribute = [PcuDialectImmediate {
            name: "unknown",
            value: PcuDialectImmediateValue::U32(3),
        }];
        let unknown_operation_attributes = [PcuDialectOperationAttributes {
            operation_index: 0,
            attributes: &unknown_attribute,
        }];
        assert_eq!(
            validate_dialect_program(
                &PcuDialectProgram {
                    operation_attributes: &unknown_operation_attributes,
                    ..program
                },
                &support
            ),
            Err(PcuDialectValidationError::UnknownImmediate)
        );
        let duplicate_input_specs = [input_specs[0], input_specs[0]];
        let duplicate_support = PcuDialectProgramSupport {
            input_ports: &duplicate_input_specs,
            ..support
        };
        assert_eq!(
            validate_dialect_program(&program, &duplicate_support),
            Err(PcuDialectValidationError::DuplicatePortName)
        );
        let duplicate_immediate_specs = [immediate_specs[0], immediate_specs[0]];
        let duplicate_op_immediate_specs = [PcuDialectOperationImmediateSpec {
            attributes: &duplicate_immediate_specs,
            ..op_immediate_specs[0]
        }];
        let duplicate_immediate_support = PcuDialectProgramSupport {
            operation_immediates: &duplicate_op_immediate_specs,
            ..support
        };
        assert_eq!(
            validate_dialect_program(&program, &duplicate_immediate_support),
            Err(PcuDialectValidationError::DuplicateImmediateName)
        );
    }

    #[test]
    fn program_verifier_rejects_duplicate_value_ids_and_invalid_immediate_encodings() {
        let dialect = PcuDialectId("org.example.constants");
        let operation = PcuDialectOperation {
            name: "constant",
            inputs: no_inputs(),
            input_count: 0,
            result: Some(PcuDialectOperand {
                id: PcuDialectValueId(4),
                value_type: PcuValueType::u32(),
            }),
            effects: PcuDialectEffects::PURE,
        };
        let operations = [operation];
        let fragment = PcuDialectFragment {
            dialect,
            version: PcuDialectVersion { major: 1, minor: 0 },
            operations: &operations,
        };
        let input_ports = [PcuDialectPort {
            name: "seed",
            value: PcuDialectOperand {
                id: PcuDialectValueId(4),
                value_type: PcuValueType::u32(),
            },
        }];
        let output_ports = [PcuDialectPort {
            name: "value",
            value: operation.result.unwrap(),
        }];
        let op_attributes = [PcuDialectOperationAttributes {
            operation_index: 0,
            attributes: &[],
        }];
        let program = PcuDialectProgram {
            fragment,
            input_ports: &input_ports,
            output_ports: &output_ports,
            operation_attributes: &op_attributes,
        };
        let operation_specs = [PcuDialectOperationSpec {
            name: "constant",
            input_types: &[],
            result_type: Some(PcuValueType::u32()),
            effects: PcuDialectEffects::PURE,
        }];
        let dialect_support = PcuDialectSupport {
            dialect,
            major: 1,
            min_minor: 0,
            max_minor: 0,
            operations: &operation_specs,
            effects: PcuDialectEffects::PURE,
        };
        let input_specs = [PcuDialectPortSpec {
            name: "seed",
            value_type: PcuValueType::u32(),
        }];
        let output_specs = [PcuDialectPortSpec {
            name: "value",
            value_type: PcuValueType::u32(),
        }];
        let support = PcuDialectProgramSupport {
            dialect_support,
            input_ports: &input_specs,
            output_ports: &output_specs,
            operation_immediates: &[],
        };
        assert_eq!(
            validate_dialect_program(&program, &support),
            Err(PcuDialectValidationError::DuplicateResult(
                PcuDialectValueId(4)
            ))
        );

        let invalid = [PcuDialectImmediate {
            name: "tiny",
            value: PcuDialectImmediateValue::I4(8),
        }];
        assert_eq!(
            validate_immediates(
                &invalid,
                &[PcuDialectImmediateSpec {
                    name: "tiny",
                    scalar_type: PcuScalarType::I4,
                }]
            ),
            Err(PcuDialectValidationError::InvalidImmediateEncoding)
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)] // Exercises pipeline wiring, fan-out, output forwarding, and atomic preflight.
    fn program_composition_wires_named_ports_and_preflights_all_output_storage() {
        let dialect = PcuDialectId("org.example.pipeline");
        let first_op = PcuDialectOperation {
            name: "copy",
            inputs: [
                PcuDialectOperand {
                    id: PcuDialectValueId(1),
                    value_type: PcuValueType::u32(),
                },
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
            ],
            input_count: 1,
            result: Some(PcuDialectOperand {
                id: PcuDialectValueId(2),
                value_type: PcuValueType::u32(),
            }),
            effects: PcuDialectEffects::PURE,
        };
        let first_ops = [first_op];
        let first_inputs = [PcuDialectPort {
            name: "input",
            value: PcuDialectOperand {
                id: PcuDialectValueId(1),
                value_type: PcuValueType::u32(),
            },
        }];
        let first_outputs = [
            PcuDialectPort {
                name: "wire",
                value: first_op.result.unwrap(),
            },
            PcuDialectPort {
                name: "extra",
                value: first_inputs[0].value,
            },
        ];
        let empty_attributes: [PcuDialectImmediate<'_>; 0] = [];
        let first_attributes = [PcuDialectOperationAttributes {
            operation_index: 0,
            attributes: &empty_attributes,
        }];
        let first = PcuDialectProgram {
            fragment: PcuDialectFragment {
                dialect,
                version: PcuDialectVersion { major: 1, minor: 0 },
                operations: &first_ops,
            },
            input_ports: &first_inputs,
            output_ports: &first_outputs,
            operation_attributes: &first_attributes,
        };

        let second_op = PcuDialectOperation {
            name: "combine",
            inputs: [
                PcuDialectOperand {
                    id: PcuDialectValueId(10),
                    value_type: PcuValueType::u32(),
                },
                PcuDialectOperand {
                    id: PcuDialectValueId(11),
                    value_type: PcuValueType::u32(),
                },
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
            ],
            input_count: 2,
            result: Some(PcuDialectOperand {
                id: PcuDialectValueId(12),
                value_type: PcuValueType::u32(),
            }),
            effects: PcuDialectEffects::PURE,
        };
        let second_ops = [second_op];
        let second_inputs = [
            PcuDialectPort {
                name: "left",
                value: second_op.inputs[0],
            },
            PcuDialectPort {
                name: "right",
                value: second_op.inputs[1],
            },
        ];
        let second_outputs = [
            PcuDialectPort {
                name: "result",
                value: second_op.result.unwrap(),
            },
            PcuDialectPort {
                name: "forwarded",
                value: second_op.inputs[0],
            },
        ];
        let immediate = [PcuDialectImmediate {
            name: "mode",
            value: PcuDialectImmediateValue::U32(3),
        }];
        let second_attributes = [PcuDialectOperationAttributes {
            operation_index: 0,
            attributes: &immediate,
        }];
        let second = PcuDialectProgram {
            fragment: PcuDialectFragment {
                dialect,
                version: first.fragment.version,
                operations: &second_ops,
            },
            input_ports: &second_inputs,
            output_ports: &second_outputs,
            operation_attributes: &second_attributes,
        };

        let first_input_types = [PcuValueType::u32()];
        let first_operation_specs = [PcuDialectOperationSpec {
            name: "copy",
            input_types: &first_input_types,
            result_type: Some(PcuValueType::u32()),
            effects: PcuDialectEffects::PURE,
        }];
        let first_dialect_support = PcuDialectSupport {
            dialect,
            major: 1,
            min_minor: 0,
            max_minor: 0,
            operations: &first_operation_specs,
            effects: PcuDialectEffects::PURE,
        };
        let first_input_schema = [PcuDialectPortSpec {
            name: "input",
            value_type: PcuValueType::u32(),
        }];
        let first_output_schema = [
            PcuDialectPortSpec {
                name: "wire",
                value_type: PcuValueType::u32(),
            },
            PcuDialectPortSpec {
                name: "extra",
                value_type: PcuValueType::u32(),
            },
        ];
        let first_support = PcuDialectProgramSupport {
            dialect_support: first_dialect_support,
            input_ports: &first_input_schema,
            output_ports: &first_output_schema,
            operation_immediates: &[],
        };

        let second_input_types = [PcuValueType::u32(), PcuValueType::u32()];
        let second_operation_specs = [PcuDialectOperationSpec {
            name: "combine",
            input_types: &second_input_types,
            result_type: Some(PcuValueType::u32()),
            effects: PcuDialectEffects::PURE,
        }];
        let second_dialect_support = PcuDialectSupport {
            dialect,
            major: 1,
            min_minor: 0,
            max_minor: 0,
            operations: &second_operation_specs,
            effects: PcuDialectEffects::PURE,
        };
        let second_input_schema = [
            PcuDialectPortSpec {
                name: "left",
                value_type: PcuValueType::u32(),
            },
            PcuDialectPortSpec {
                name: "right",
                value_type: PcuValueType::u32(),
            },
        ];
        let second_output_schema = [
            PcuDialectPortSpec {
                name: "result",
                value_type: PcuValueType::u32(),
            },
            PcuDialectPortSpec {
                name: "forwarded",
                value_type: PcuValueType::u32(),
            },
        ];
        let immediate_schema = [PcuDialectImmediateSpec {
            name: "mode",
            scalar_type: PcuScalarType::U32,
        }];
        let second_immediate_schemas = [PcuDialectOperationImmediateSpec {
            operation_name: "combine",
            attributes: &immediate_schema,
        }];
        let second_support = PcuDialectProgramSupport {
            dialect_support: second_dialect_support,
            input_ports: &second_input_schema,
            output_ports: &second_output_schema,
            operation_immediates: &second_immediate_schemas,
        };

        let composed_input_schema = first_input_schema;
        let composed_output_schema = second_output_schema;
        let composed_second_input_types = second_input_types;
        let composed_operation_specs = [
            first_operation_specs[0],
            PcuDialectOperationSpec {
                name: "combine",
                input_types: &composed_second_input_types,
                result_type: Some(PcuValueType::u32()),
                effects: PcuDialectEffects::PURE,
            },
        ];
        let composed_dialect_support = PcuDialectSupport {
            dialect,
            major: 1,
            min_minor: 0,
            max_minor: 0,
            operations: &composed_operation_specs,
            effects: PcuDialectEffects::PURE,
        };
        let composed_support = PcuDialectProgramSupport {
            dialect_support: composed_dialect_support,
            input_ports: &composed_input_schema,
            output_ports: &composed_output_schema,
            operation_immediates: &second_immediate_schemas,
        };
        let supports = PcuDialectProgramComposeSupports {
            first: &first_support,
            second: &second_support,
            composed: &composed_support,
        };
        let bindings = [
            PcuDialectProgramBinding {
                first_output: "wire",
                second_input: "left",
            },
            PcuDialectProgramBinding {
                first_output: "wire",
                second_input: "right",
            },
        ];
        let mut operations = [first_op; 2];
        let mut inputs = [first_inputs[0]; 1];
        let mut outputs = [first_outputs[0]; 2];
        let mut attributes = [first_attributes[0]; 2];
        let composed = compose_dialect_programs(
            &first,
            &second,
            &bindings,
            supports,
            PcuDialectProgramComposeStorage {
                operations: &mut operations,
                input_ports: &mut inputs,
                output_ports: &mut outputs,
                operation_attributes: &mut attributes,
            },
        )
        .unwrap();
        assert_eq!(
            validate_dialect_program(&composed, &composed_support),
            Ok(())
        );
        assert_eq!(
            composed.fragment.operations[1].inputs[0].id,
            PcuDialectValueId(2)
        );
        assert_eq!(
            composed.fragment.operations[1].inputs[1].id,
            PcuDialectValueId(2)
        );
        assert_eq!(
            composed.fragment.operations[1].result.unwrap().id,
            PcuDialectValueId(15)
        );
        assert_eq!(composed.input_ports, first_inputs);
        assert_eq!(composed.output_ports[0].value.id, PcuDialectValueId(15));
        assert_eq!(composed.output_ports[1].value.id, PcuDialectValueId(2));
        assert_eq!(composed.operation_attributes[1].operation_index, 1);
        let mut too_small_operations = [first_op];
        let original_operations = too_small_operations;
        let mut unchanged_inputs = [first_inputs[0]; 1];
        let original_inputs = unchanged_inputs;
        let mut unchanged_outputs = [first_outputs[0]; 2];
        let original_outputs = unchanged_outputs;
        let mut unchanged_attributes = [first_attributes[0]; 2];
        let original_attributes = unchanged_attributes;
        let capacity_error = {
            let result = compose_dialect_programs(
                &first,
                &second,
                &bindings,
                supports,
                PcuDialectProgramComposeStorage {
                    operations: &mut too_small_operations,
                    input_ports: &mut unchanged_inputs,
                    output_ports: &mut unchanged_outputs,
                    operation_attributes: &mut unchanged_attributes,
                },
            );
            match result {
                Err(error) => error,
                Ok(_) => panic!("insufficient operation storage must fail during preflight"),
            }
        };
        assert_eq!(
            capacity_error,
            PcuDialectProgramComposeError::OperationStorageTooSmall
        );
        assert_eq!(too_small_operations, original_operations);
        assert_eq!(unchanged_inputs, original_inputs);
        assert_eq!(unchanged_outputs, original_outputs);
        assert_eq!(unchanged_attributes, original_attributes);
    }
}
