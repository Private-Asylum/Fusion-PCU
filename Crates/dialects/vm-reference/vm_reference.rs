//! A small executable consumer of public [`fusion_pcu`] dialect programs and fragments.
//!
//! This `no_std` reference VM implements a typed, versioned `u32` stream dialect: `input.u32`
//! reads the next caller input, `add.u32` performs wrapping addition, and `emit.u32` appends to
//! caller output. The program path binds named `u32` ports and supports `literal.u32` with a typed
//! `value` immediate plus wrapping addition. Both paths use caller-provided storage and validate
//! before execution.

#![no_std]

use fusion_pcu::{
    PcuDialectEffects,
    PcuDialectFragment,
    PcuDialectId,
    PcuDialectImmediateSpec,
    PcuDialectImmediateValue,
    PcuDialectOperationImmediateSpec,
    PcuDialectOperationSpec,
    PcuDialectPortSpec,
    PcuDialectProgram,
    PcuDialectProgramSupport,
    PcuDialectSupport,
    PcuDialectValidationError,
    PcuScalarType,
    PcuValueType,
    validate_dialect_fragment,
    validate_dialect_program,
};

/// Stable identity and version implemented by this VM.
pub const DIALECT: PcuDialectId<'static> = PcuDialectId("org.fusion.vm-reference");
/// Major version accepted by this VM.
pub const DIALECT_MAJOR: u16 = 1;
/// Only this minor version is currently implemented.
pub const DIALECT_MINOR: u16 = 0;

const INPUT: &str = "input.u32";
const ADD: &str = "add.u32";
const EMIT: &str = "emit.u32";
const LITERAL: &str = "literal.u32";
const LITERAL_VALUE: &str = "value";
const U32: PcuValueType = PcuValueType::u32();
const INPUT_TYPES: &[PcuValueType] = &[];
const BINARY_TYPES: &[PcuValueType] = &[U32, U32];
const UNARY_TYPES: &[PcuValueType] = &[U32];
const SPECS: &[PcuDialectOperationSpec<'static>] = &[
    PcuDialectOperationSpec {
        name: INPUT,
        input_types: INPUT_TYPES,
        result_type: Some(U32),
        effects: PcuDialectEffects::READ_MEMORY,
    },
    PcuDialectOperationSpec {
        name: ADD,
        input_types: BINARY_TYPES,
        result_type: Some(U32),
        effects: PcuDialectEffects::PURE,
    },
    PcuDialectOperationSpec {
        name: EMIT,
        input_types: UNARY_TYPES,
        result_type: None,
        effects: PcuDialectEffects::WRITE_MEMORY,
    },
    PcuDialectOperationSpec {
        name: LITERAL,
        input_types: INPUT_TYPES,
        result_type: Some(U32),
        effects: PcuDialectEffects::PURE,
    },
];

/// Exact consumer support declaration accepted by [`execute`].
pub const SUPPORT: PcuDialectSupport<'static> = PcuDialectSupport {
    dialect: DIALECT,
    major: DIALECT_MAJOR,
    min_minor: DIALECT_MINOR,
    max_minor: DIALECT_MINOR,
    operations: SPECS,
    effects: PcuDialectEffects::READ_MEMORY.union(PcuDialectEffects::WRITE_MEMORY),
};

const PROGRAM_INPUT_PORTS: &[PcuDialectPortSpec<'static>] = &[PcuDialectPortSpec {
    name: "lhs",
    value_type: U32,
}];
const PROGRAM_OUTPUT_PORTS: &[PcuDialectPortSpec<'static>] = &[PcuDialectPortSpec {
    name: "sum",
    value_type: U32,
}];
const LITERAL_IMMEDIATES: &[PcuDialectImmediateSpec<'static>] = &[PcuDialectImmediateSpec {
    name: LITERAL_VALUE,
    scalar_type: PcuScalarType::U32,
}];
const PROGRAM_IMMEDIATE_SCHEMAS: &[PcuDialectOperationImmediateSpec<'static>] =
    &[PcuDialectOperationImmediateSpec {
        operation_name: LITERAL,
        attributes: LITERAL_IMMEDIATES,
    }];

/// Program contract for a named `lhs: u32` input and `sum: u32` output.
pub const PROGRAM_SUPPORT: PcuDialectProgramSupport<'static> = PcuDialectProgramSupport {
    dialect_support: SUPPORT,
    input_ports: PROGRAM_INPUT_PORTS,
    output_ports: PROGRAM_OUTPUT_PORTS,
    operation_immediates: PROGRAM_IMMEDIATE_SCHEMAS,
};

/// Named caller input used by [`execute_program`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmProgramInput<'a> {
    /// Port name from the program contract.
    pub name: &'a str,
    /// Value bound to the named input port.
    pub value: u32,
}

/// Named caller output slot used by [`execute_program`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmProgramOutput<'a> {
    /// Port name from the program contract.
    pub name: &'a str,
    /// Value written by the program.
    pub value: u32,
}

/// Caller-owned buffers for program execution.
#[derive(Debug)]
pub struct VmProgramBuffers<'a> {
    /// Values indexed by local SSA ID, including IDs seeded by input ports.
    pub values: &'a mut [u32],
    /// Host bindings for the named input ports.
    pub inputs: &'a [VmProgramInput<'a>],
    /// Host slots for named output ports.
    pub outputs: &'a mut [VmProgramOutput<'a>],
}

/// Caller-owned execution buffers.
#[derive(Debug)]
pub struct VmBuffers<'a> {
    /// Values indexed by fragment-local SSA value ID. Must have an entry for each result ID.
    pub values: &'a mut [u32],
    /// Ordered values consumed by `input.u32` operations.
    pub inputs: &'a [u32],
    /// Output storage receiving values in `emit.u32` order.
    pub outputs: &'a mut [u32],
}

/// Successful execution counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmReport {
    /// Number of caller inputs consumed.
    pub inputs_consumed: usize,
    /// Number of outputs written.
    pub outputs_written: usize,
    /// Number of operations executed.
    pub operations_executed: usize,
}

/// Execution rejection. Rejections happen before caller buffers are modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmError {
    /// Fragment identity, version, signatures, effects, or SSA flow were rejected.
    InvalidFragment(PcuDialectValidationError),
    /// Ports or typed immediate attributes were rejected.
    InvalidProgram(PcuDialectValidationError),
    /// Host input bindings are missing, duplicated, or unknown.
    InvalidInputBindings,
    /// Host output slots are missing, duplicated, or unknown.
    InvalidOutputBindings,
    /// Fragment execution needs immediate metadata supplied by a program.
    ImmediateRequiresProgram,
    /// The program is valid but outside this evaluator's bounded semantic slice.
    UnsupportedProgramOperation,
    /// An SSA result ID does not fit the caller's value storage.
    ValueStorageTooSmall { required: usize, available: usize },
    /// There are fewer caller inputs than input operations.
    InputStorageTooSmall { required: usize, available: usize },
    /// There are fewer caller output slots than emit operations.
    OutputStorageTooSmall { required: usize, available: usize },
}

/// Validates and executes a supported fragment using only caller-provided buffers.
///
/// Addition wraps modulo 2^32. The input and output slices form ordered streams; there are no
/// immediate operands or named port bindings in the current core protocol.
///
/// # Errors
///
/// Returns [`VmError`] if the dialect contract or any required buffer capacity is invalid.
///
/// # Panics
///
/// Panics only if the core validator accepts an operation whose signature contradicts the
/// static support table; this is an internal protocol invariant violation.
pub fn execute(
    fragment: &PcuDialectFragment<'_>,
    buffers: &mut VmBuffers<'_>,
) -> Result<VmReport, VmError> {
    validate_dialect_fragment(fragment, &SUPPORT).map_err(VmError::InvalidFragment)?;

    let mut input_count = 0;
    let mut output_count = 0;
    let mut required_values = 0;
    for operation in fragment.operations {
        match operation.name {
            INPUT => input_count += 1,
            EMIT => output_count += 1,
            ADD => {}
            LITERAL => return Err(VmError::ImmediateRequiresProgram),
            _ => unreachable!("validation admits only the static support table"),
        }
        if let Some(result) = operation.result {
            required_values = required_values.max(usize::from(result.id.0) + 1);
        }
    }
    if buffers.values.len() < required_values {
        return Err(VmError::ValueStorageTooSmall {
            required: required_values,
            available: buffers.values.len(),
        });
    }
    if buffers.inputs.len() < input_count {
        return Err(VmError::InputStorageTooSmall {
            required: input_count,
            available: buffers.inputs.len(),
        });
    }
    if buffers.outputs.len() < output_count {
        return Err(VmError::OutputStorageTooSmall {
            required: output_count,
            available: buffers.outputs.len(),
        });
    }

    let (mut input_index, mut output_index) = (0, 0);
    for operation in fragment.operations {
        match operation.name {
            INPUT => {
                let result = operation
                    .result
                    .expect("signature validation requires result");
                buffers.values[usize::from(result.id.0)] = buffers.inputs[input_index];
                input_index += 1;
            }
            ADD => {
                let lhs = buffers.values[usize::from(operation.inputs[0].id.0)];
                let rhs = buffers.values[usize::from(operation.inputs[1].id.0)];
                let result = operation
                    .result
                    .expect("signature validation requires result");
                buffers.values[usize::from(result.id.0)] = lhs.wrapping_add(rhs);
            }
            EMIT => {
                buffers.outputs[output_index] =
                    buffers.values[usize::from(operation.inputs[0].id.0)];
                output_index += 1;
            }
            _ => unreachable!("validation admits only the static support table"),
        }
    }
    Ok(VmReport {
        inputs_consumed: input_index,
        outputs_written: output_index,
        operations_executed: fragment.operations.len(),
    })
}

/// Validates and executes a named-port program with literal and wrapping-add semantics.
///
/// Host binding slices must contain exactly the names declared by [`PROGRAM_SUPPORT`]. The
/// `literal.u32` operation reads its typed `value: u32` immediate. All contract and capacity
/// checks happen before caller buffers are changed.
///
/// # Errors
///
/// Returns [`VmError`] when the program, host bindings, operation set, or value capacity is
/// invalid.
///
/// # Panics
///
/// Panics only if a program accepted by [`validate_dialect_program`] lacks its declared literal
/// value or a validated operation result; that would be an internal contract contradiction.
pub fn execute_program(
    program: &PcuDialectProgram<'_>,
    buffers: &mut VmProgramBuffers<'_>,
) -> Result<VmReport, VmError> {
    validate_dialect_program(program, &PROGRAM_SUPPORT).map_err(VmError::InvalidProgram)?;
    if buffers.inputs.len() != program.input_ports.len()
        || program.input_ports.iter().any(|port| {
            buffers
                .inputs
                .iter()
                .filter(|binding| binding.name == port.name)
                .count()
                != 1
        })
    {
        return Err(VmError::InvalidInputBindings);
    }
    if buffers.outputs.len() != program.output_ports.len()
        || program.output_ports.iter().any(|port| {
            buffers
                .outputs
                .iter()
                .filter(|binding| binding.name == port.name)
                .count()
                != 1
        })
    {
        return Err(VmError::InvalidOutputBindings);
    }

    let mut required_values = 0;
    for port in program.input_ports {
        required_values = required_values.max(usize::from(port.value.id.0) + 1);
    }
    for operation in program.fragment.operations {
        if !matches!(operation.name, LITERAL | ADD) {
            return Err(VmError::UnsupportedProgramOperation);
        }
        if let Some(result) = operation.result {
            required_values = required_values.max(usize::from(result.id.0) + 1);
        }
    }
    if buffers.values.len() < required_values {
        return Err(VmError::ValueStorageTooSmall {
            required: required_values,
            available: buffers.values.len(),
        });
    }

    for port in program.input_ports {
        let binding = buffers
            .inputs
            .iter()
            .find(|binding| binding.name == port.name)
            .expect("input binding preflight requires every port");
        buffers.values[usize::from(port.value.id.0)] = binding.value;
    }
    for (index, operation) in program.fragment.operations.iter().enumerate() {
        match operation.name {
            LITERAL => {
                let attributes = program
                    .operation_attributes
                    .iter()
                    .find(|entry| usize::from(entry.operation_index) == index)
                    .expect("program validation requires operation attributes");
                let immediate = attributes
                    .attributes
                    .iter()
                    .find(|attribute| attribute.name == LITERAL_VALUE)
                    .expect("program validation requires literal value");
                let PcuDialectImmediateValue::U32(value) = immediate.value else {
                    unreachable!("program validation checked literal immediate type");
                };
                let result = operation
                    .result
                    .expect("signature validation requires result");
                buffers.values[usize::from(result.id.0)] = value;
            }
            ADD => {
                let lhs = buffers.values[usize::from(operation.inputs[0].id.0)];
                let rhs = buffers.values[usize::from(operation.inputs[1].id.0)];
                let result = operation
                    .result
                    .expect("signature validation requires result");
                buffers.values[usize::from(result.id.0)] = lhs.wrapping_add(rhs);
            }
            _ => unreachable!("program preflight admits only literal and add"),
        }
    }
    for port in program.output_ports {
        let value = buffers.values[usize::from(port.value.id.0)];
        let output = buffers
            .outputs
            .iter_mut()
            .find(|binding| binding.name == port.name)
            .expect("output binding preflight requires every port");
        output.value = value;
    }
    Ok(VmReport {
        inputs_consumed: program.input_ports.len(),
        outputs_written: program.output_ports.len(),
        operations_executed: program.fragment.operations.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusion_pcu::{
        PcuDialectImmediate,
        PcuDialectOperationAttributes,
        PcuDialectOperand,
        PcuDialectOperation,
        PcuDialectPort,
        PcuDialectValueId,
        compose_dialect_fragments,
    };

    const UNUSED: [PcuDialectOperand; 4] = [PcuDialectOperand::UNUSED; 4];

    fn fragment<'a>(operations: &'a [PcuDialectOperation<'a>]) -> PcuDialectFragment<'a> {
        PcuDialectFragment {
            dialect: DIALECT,
            version: fusion_pcu::PcuDialectVersion {
                major: DIALECT_MAJOR,
                minor: DIALECT_MINOR,
            },
            operations,
        }
    }

    fn producer() -> [PcuDialectOperation<'static>; 3] {
        [
            PcuDialectOperation {
                name: INPUT,
                inputs: UNUSED,
                input_count: 0,
                result: Some(PcuDialectOperand {
                    id: PcuDialectValueId(0),
                    value_type: U32,
                }),
                effects: PcuDialectEffects::READ_MEMORY,
            },
            PcuDialectOperation {
                name: INPUT,
                inputs: UNUSED,
                input_count: 0,
                result: Some(PcuDialectOperand {
                    id: PcuDialectValueId(1),
                    value_type: U32,
                }),
                effects: PcuDialectEffects::READ_MEMORY,
            },
            PcuDialectOperation {
                name: ADD,
                inputs: [
                    PcuDialectOperand {
                        id: PcuDialectValueId(0),
                        value_type: U32,
                    },
                    PcuDialectOperand {
                        id: PcuDialectValueId(1),
                        value_type: U32,
                    },
                    PcuDialectOperand::UNUSED,
                    PcuDialectOperand::UNUSED,
                ],
                input_count: 2,
                result: Some(PcuDialectOperand {
                    id: PcuDialectValueId(2),
                    value_type: U32,
                }),
                effects: PcuDialectEffects::PURE,
            },
        ]
    }

    fn emit(id: u16) -> PcuDialectOperation<'static> {
        PcuDialectOperation {
            name: EMIT,
            inputs: [
                PcuDialectOperand {
                    id: PcuDialectValueId(id),
                    value_type: U32,
                },
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
                PcuDialectOperand::UNUSED,
            ],
            input_count: 1,
            result: None,
            effects: PcuDialectEffects::WRITE_MEMORY,
        }
    }

    #[test]
    fn executes_stream_and_exposes_output_effect() {
        let operations = producer();
        let full = [operations[0], operations[1], operations[2], emit(2)];
        let fragment = fragment(&full);
        let mut values = [0; 3];
        let mut outputs = [0; 1];
        let mut buffers = VmBuffers {
            values: &mut values,
            inputs: &[19, 23],
            outputs: &mut outputs,
        };
        assert_eq!(
            execute(&fragment, &mut buffers),
            Ok(VmReport {
                inputs_consumed: 2,
                outputs_written: 1,
                operations_executed: 4,
            })
        );
        assert_eq!(outputs, [42]);
    }

    #[test]
    fn rejects_invalid_fragment_and_capacity_before_mutation() {
        let valid = producer();
        let mut invalid_emit = emit(0);
        invalid_emit.effects = PcuDialectEffects::PURE;
        let invalid_ops = [valid[0], invalid_emit];
        let invalid = fragment(&invalid_ops);
        let mut values = [77; 1];
        let mut outputs = [88; 1];
        let mut buffers = VmBuffers {
            values: &mut values,
            inputs: &[1],
            outputs: &mut outputs,
        };
        assert!(matches!(
            execute(&invalid, &mut buffers),
            Err(VmError::InvalidFragment(_))
        ));
        assert_eq!(values, [77]);
        assert_eq!(outputs, [88]);

        let insufficient = fragment(&valid);
        let mut values = [77; 2];
        let mut outputs = [];
        let mut buffers = VmBuffers {
            values: &mut values,
            inputs: &[1, 2],
            outputs: &mut outputs,
        };
        assert_eq!(
            execute(&insufficient, &mut buffers),
            Err(VmError::ValueStorageTooSmall {
                required: 3,
                available: 2,
            })
        );
        assert_eq!(values, [77; 2]);
    }

    #[test]
    fn composes_self_contained_fragments_and_preserves_semantics() {
        let first_ops = producer();
        let first = fragment(&first_ops);
        let second_ops = [producer()[0], producer()[1], producer()[2], emit(2)];
        let second = fragment(&second_ops);
        let mut composed_ops = [first_ops[0]; 8];
        let count = compose_dialect_fragments(&first, &second, &mut composed_ops).unwrap();
        let composed = fragment(&composed_ops[..count]);
        assert_eq!(composed_ops[3].result.unwrap().id, PcuDialectValueId(3));
        assert_eq!(composed_ops[6].inputs[0].id, PcuDialectValueId(5));

        let mut values = [0; 6];
        let mut outputs = [0; 1];
        let mut buffers = VmBuffers {
            values: &mut values,
            inputs: &[1, 2, 20, 22],
            outputs: &mut outputs,
        };
        assert_eq!(execute(&composed, &mut buffers).unwrap().outputs_written, 1);
        assert_eq!(outputs, [42]);
    }

    #[test]
    fn executes_validated_named_ports_and_typed_literal_immediate() {
        let operations = [
            PcuDialectOperation {
                name: LITERAL,
                inputs: UNUSED,
                input_count: 0,
                result: Some(PcuDialectOperand {
                    id: PcuDialectValueId(1),
                    value_type: U32,
                }),
                effects: PcuDialectEffects::PURE,
            },
            PcuDialectOperation {
                name: ADD,
                inputs: [
                    PcuDialectOperand {
                        id: PcuDialectValueId(0),
                        value_type: U32,
                    },
                    PcuDialectOperand {
                        id: PcuDialectValueId(1),
                        value_type: U32,
                    },
                    PcuDialectOperand::UNUSED,
                    PcuDialectOperand::UNUSED,
                ],
                input_count: 2,
                result: Some(PcuDialectOperand {
                    id: PcuDialectValueId(2),
                    value_type: U32,
                }),
                effects: PcuDialectEffects::PURE,
            },
        ];
        let immediate = PcuDialectImmediate {
            name: LITERAL_VALUE,
            value: PcuDialectImmediateValue::U32(10),
        };
        let literal_attributes = [immediate];
        let per_operation_attributes = [
            PcuDialectOperationAttributes {
                operation_index: 0,
                attributes: &literal_attributes,
            },
            PcuDialectOperationAttributes {
                operation_index: 1,
                attributes: &[],
            },
        ];
        let inputs = [PcuDialectPort {
            name: "lhs",
            value: PcuDialectOperand {
                id: PcuDialectValueId(0),
                value_type: U32,
            },
        }];
        let outputs = [PcuDialectPort {
            name: "sum",
            value: PcuDialectOperand {
                id: PcuDialectValueId(2),
                value_type: U32,
            },
        }];
        let program = PcuDialectProgram {
            fragment: fragment(&operations),
            input_ports: &inputs,
            output_ports: &outputs,
            operation_attributes: &per_operation_attributes,
        };
        let mut values = [0; 3];
        let mut caller_outputs = [VmProgramOutput {
            name: "sum",
            value: 999,
        }];
        let mut buffers = VmProgramBuffers {
            values: &mut values,
            inputs: &[VmProgramInput {
                name: "lhs",
                value: 32,
            }],
            outputs: &mut caller_outputs,
        };
        assert_eq!(validate_dialect_program(&program, &PROGRAM_SUPPORT), Ok(()));
        assert_eq!(
            execute_program(&program, &mut buffers),
            Ok(VmReport {
                inputs_consumed: 1,
                outputs_written: 1,
                operations_executed: 2,
            })
        );
        assert_eq!(buffers.outputs[0].value, 42);
    }

    #[test]
    fn rejects_bad_immediate_before_changing_values_or_outputs() {
        let operation = PcuDialectOperation {
            name: LITERAL,
            inputs: UNUSED,
            input_count: 0,
            result: Some(PcuDialectOperand {
                id: PcuDialectValueId(1),
                value_type: U32,
            }),
            effects: PcuDialectEffects::PURE,
        };
        let operations = [operation];
        let immediate = PcuDialectImmediate {
            name: LITERAL_VALUE,
            value: PcuDialectImmediateValue::F32(1.0_f32.to_bits()),
        };
        let attributes = [immediate];
        let op_attributes = [PcuDialectOperationAttributes {
            operation_index: 0,
            attributes: &attributes,
        }];
        let inputs = [PcuDialectPort {
            name: "lhs",
            value: PcuDialectOperand {
                id: PcuDialectValueId(0),
                value_type: U32,
            },
        }];
        let outputs = [PcuDialectPort {
            name: "sum",
            value: PcuDialectOperand {
                id: PcuDialectValueId(1),
                value_type: U32,
            },
        }];
        let program = PcuDialectProgram {
            fragment: fragment(&operations),
            input_ports: &inputs,
            output_ports: &outputs,
            operation_attributes: &op_attributes,
        };
        let mut values = [55; 2];
        let mut caller_outputs = [VmProgramOutput {
            name: "sum",
            value: 77,
        }];
        let mut buffers = VmProgramBuffers {
            values: &mut values,
            inputs: &[VmProgramInput {
                name: "lhs",
                value: 3,
            }],
            outputs: &mut caller_outputs,
        };
        assert!(matches!(
            execute_program(&program, &mut buffers),
            Err(VmError::InvalidProgram(_))
        ));
        assert_eq!(buffers.values, [55; 2]);
        assert_eq!(buffers.outputs[0].value, 77);
    }
}
