//! An external-crate consumer of the open dialect protocol.
//!
//! This deliberately small stream interpreter owns its operation meanings. PCU validates the
//! names, types, effects, and SSA flow without learning any of these operations itself.

use fusion_pcu::{
    PcuDialectComposeError,
    PcuDialectBuilder,
    PcuDialectEffects,
    PcuDialectFragment,
    PcuDialectId,
    PcuDialectOperand,
    PcuDialectOperation,
    PcuDialectOperationSpec,
    PcuDialectOperationAttributes,
    PcuDialectOperationImmediateSpec,
    PcuDialectImmediate,
    PcuDialectImmediateSpec,
    PcuDialectImmediateValue,
    PcuDialectPort,
    PcuDialectPortSpec,
    PcuDialectProgram,
    PcuDialectProgramBinding,
    PcuDialectProgramComposeStorage,
    PcuDialectProgramComposeSupports,
    PcuDialectProgramSupport,
    PcuDialectSupport,
    PcuDialectValidationError,
    PcuDialectValueId,
    PcuDialectVersion,
    PcuValueType,
    PcuScalarType,
    compose_dialect_fragments,
    compose_dialect_programs,
    validate_dialect_fragment,
    validate_dialect_program,
};

const DIALECT: PcuDialectId<'static> = PcuDialectId("org.fusion.reference.stream");
const VERSION: PcuDialectVersion = PcuDialectVersion { major: 1, minor: 0 };
const U32: PcuValueType = PcuValueType::u32();
const UNUSED: PcuDialectOperand = PcuDialectOperand::UNUSED;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamError {
    Invalid(PcuDialectValidationError),
    ValueStorageExhausted,
    OutputStorageExhausted,
    MissingValue(PcuDialectValueId),
}

const fn support<'a>(specs: &'a [PcuDialectOperationSpec<'a>]) -> PcuDialectSupport<'a> {
    PcuDialectSupport {
        dialect: DIALECT,
        major: VERSION.major,
        min_minor: VERSION.minor,
        max_minor: VERSION.minor,
        operations: specs,
        effects: PcuDialectEffects::WRITE_MEMORY,
    }
}

const fn specs() -> [PcuDialectOperationSpec<'static>; 4] {
    [
        PcuDialectOperationSpec {
            name: "seed.u32",
            input_types: &[],
            result_type: Some(U32),
            effects: PcuDialectEffects::PURE,
        },
        PcuDialectOperationSpec {
            name: "rotate_left_one.u32",
            input_types: &[U32],
            result_type: Some(U32),
            effects: PcuDialectEffects::PURE,
        },
        PcuDialectOperationSpec {
            name: "not.u32",
            input_types: &[U32],
            result_type: Some(U32),
            effects: PcuDialectEffects::PURE,
        },
        PcuDialectOperationSpec {
            name: "emit.u32",
            input_types: &[U32],
            result_type: None,
            effects: PcuDialectEffects::WRITE_MEMORY,
        },
    ]
}

const fn value(id: u16) -> PcuDialectOperand {
    PcuDialectOperand {
        id: PcuDialectValueId(id),
        value_type: U32,
    }
}

const fn produce(name: &'static str, id: u16) -> PcuDialectOperation<'static> {
    PcuDialectOperation {
        name,
        inputs: [UNUSED; 4],
        input_count: 0,
        result: Some(value(id)),
        effects: PcuDialectEffects::PURE,
    }
}

const fn unary(name: &'static str, input: u16, result: u16) -> PcuDialectOperation<'static> {
    PcuDialectOperation {
        name,
        inputs: [value(input), UNUSED, UNUSED, UNUSED],
        input_count: 1,
        result: Some(value(result)),
        effects: PcuDialectEffects::PURE,
    }
}

const fn emit(input: u16) -> PcuDialectOperation<'static> {
    PcuDialectOperation {
        name: "emit.u32",
        inputs: [value(input), UNUSED, UNUSED, UNUSED],
        input_count: 1,
        result: None,
        effects: PcuDialectEffects::WRITE_MEMORY,
    }
}

fn read_value(values: &[Option<u32>], id: PcuDialectValueId) -> Result<u32, StreamError> {
    values
        .get(usize::from(id.0))
        .and_then(|value| *value)
        .ok_or(StreamError::MissingValue(id))
}

fn execute(
    fragment: &PcuDialectFragment<'_>,
    seed: u32,
    outputs: &mut [u32],
) -> Result<usize, StreamError> {
    let specs = specs();
    validate_dialect_fragment(fragment, &support(&specs)).map_err(StreamError::Invalid)?;

    // This consumer sets its own finite execution limit; the shared dialect protocol does not
    // pretend every backend has an unbounded register file.
    let mut values = [None; 16];
    let mut emitted = 0;
    for operation in fragment.operations {
        let result = match operation.name {
            "seed.u32" => Some(seed),
            "rotate_left_one.u32" => {
                Some(read_value(&values, operation.inputs[0].id)?.rotate_left(1))
            }
            "not.u32" => Some(!read_value(&values, operation.inputs[0].id)?),
            "emit.u32" => {
                let output = outputs
                    .get_mut(emitted)
                    .ok_or(StreamError::OutputStorageExhausted)?;
                *output = read_value(&values, operation.inputs[0].id)?;
                emitted += 1;
                None
            }
            _ => unreachable!("the exact support table admitted every operation"),
        };
        if let (Some(result_id), Some(result)) = (operation.result, result) {
            *values
                .get_mut(usize::from(result_id.id.0))
                .ok_or(StreamError::ValueStorageExhausted)? = Some(result);
        }
    }
    Ok(emitted)
}

#[test]
fn independent_stream_consumer_executes_composed_fragments() {
    let support_specs = specs();
    let mut rotate_storage = [produce("seed.u32", 0); 3];
    let mut builder = PcuDialectBuilder::new(
        DIALECT,
        VERSION,
        support(&support_specs),
        &mut rotate_storage,
        1,
    );
    let seed = builder
        .operation0::<u32>("seed.u32", PcuDialectEffects::PURE)
        .unwrap();
    let rotated = builder
        .operation1::<u32, u32>("rotate_left_one.u32", PcuDialectEffects::PURE, seed)
        .unwrap();
    builder
        .effect1("emit.u32", PcuDialectEffects::WRITE_MEMORY, rotated)
        .unwrap();
    let rotate = builder.finish().unwrap();
    let invert_operations = [produce("seed.u32", 0), unary("not.u32", 0, 1), emit(1)];
    let invert = PcuDialectFragment {
        operations: &invert_operations,
        ..rotate
    };
    let mut combined = [produce("seed.u32", 0); 6];
    assert_eq!(
        compose_dialect_fragments(&rotate, &invert, &mut combined),
        Ok(6)
    );
    assert_eq!(
        combined[3].result.map(|value| value.id),
        Some(PcuDialectValueId(2))
    );
    assert_eq!(
        combined[4].result.map(|value| value.id),
        Some(PcuDialectValueId(3))
    );
    let fragment = PcuDialectFragment {
        operations: &combined,
        ..rotate
    };
    let mut outputs = [0; 2];
    assert_eq!(execute(&fragment, 0x1234_5678, &mut outputs), Ok(2));
    assert_eq!(outputs, [0x2468_acf0, !0x1234_5678]);
}

#[test]
fn independent_stream_consumer_rejects_unsupported_semantics_and_capacity() {
    let operations = [produce("seed.u32", 0), emit(0)];
    let fragment = PcuDialectFragment {
        dialect: DIALECT,
        version: VERSION,
        operations: &operations,
    };
    assert_eq!(
        execute(&fragment, 7, &mut []),
        Err(StreamError::OutputStorageExhausted)
    );

    let unsupported = [produce("unregistered.u32", 0)];
    let invalid = PcuDialectFragment {
        operations: &unsupported,
        ..fragment
    };
    assert_eq!(
        execute(&invalid, 7, &mut [0]),
        Err(StreamError::Invalid(
            PcuDialectValidationError::UnsupportedOperation
        ))
    );
    let wrong_version = PcuDialectFragment {
        version: PcuDialectVersion { major: 2, minor: 0 },
        ..fragment
    };
    assert_eq!(
        execute(&wrong_version, 7, &mut [0]),
        Err(StreamError::Invalid(
            PcuDialectValidationError::UnsupportedVersion
        ))
    );
}

#[test]
fn composition_rejects_cross_dialect_capture() {
    let operations = [produce("seed.u32", 0)];
    let first = PcuDialectFragment {
        dialect: DIALECT,
        version: VERSION,
        operations: &operations,
    };
    let other = PcuDialectFragment {
        dialect: PcuDialectId("org.example.other-stream"),
        ..first
    };
    let mut output = [operations[0]; 2];
    assert_eq!(
        compose_dialect_fragments(&first, &other, &mut output),
        Err(PcuDialectComposeError::IdentityMismatch)
    );
}

#[test]
fn named_stream_ports_and_typed_mask_execute() {
    let operations = [unary("xor_mask.u32", 0, 1)];
    let fragment = PcuDialectFragment {
        dialect: DIALECT,
        version: VERSION,
        operations: &operations,
    };
    let inputs = [PcuDialectPort {
        name: "source",
        value: value(0),
    }];
    let outputs = [PcuDialectPort {
        name: "result",
        value: value(1),
    }];
    let mask = [PcuDialectImmediate {
        name: "mask",
        value: PcuDialectImmediateValue::U32(0x00ff_00ff),
    }];
    let attributes = [PcuDialectOperationAttributes {
        operation_index: 0,
        attributes: &mask,
    }];
    let program = PcuDialectProgram {
        fragment,
        input_ports: &inputs,
        output_ports: &outputs,
        operation_attributes: &attributes,
    };
    let specs = [PcuDialectOperationSpec {
        name: "xor_mask.u32",
        input_types: &[U32],
        result_type: Some(U32),
        effects: PcuDialectEffects::PURE,
    }];
    let input_specs = [PcuDialectPortSpec {
        name: "source",
        value_type: U32,
    }];
    let output_specs = [PcuDialectPortSpec {
        name: "result",
        value_type: U32,
    }];
    let mask_specs = [PcuDialectImmediateSpec {
        name: "mask",
        scalar_type: PcuScalarType::U32,
    }];
    let operation_immediates = [PcuDialectOperationImmediateSpec {
        operation_name: "xor_mask.u32",
        attributes: &mask_specs,
    }];
    let support = PcuDialectProgramSupport {
        dialect_support: support(&specs),
        input_ports: &input_specs,
        output_ports: &output_specs,
        operation_immediates: &operation_immediates,
    };
    validate_dialect_program(&program, &support).unwrap();

    // This consumer binds its named input, executes the admitted operation, and reads the named
    // output. Core does not know what xor_mask means.
    let source = 0x1234_5678_u32;
    let mut values = [0_u32; 2];
    values[usize::from(program.input_ports[0].value.id.0)] = source;
    let PcuDialectImmediateValue::U32(mask_value) =
        program.operation_attributes[0].attributes[0].value
    else {
        unreachable!("validated support fixes the immediate type")
    };
    let operation = &program.fragment.operations[0];
    values[usize::from(operation.result.unwrap().id.0)] =
        values[usize::from(operation.inputs[0].id.0)] ^ mask_value;
    let result = values[usize::from(program.output_ports[0].value.id.0)];
    assert_eq!(result, 0x12cb_5687);

    let wrong_mask = [PcuDialectImmediate {
        name: "mask",
        value: PcuDialectImmediateValue::Bool(true),
    }];
    let wrong_attributes = [PcuDialectOperationAttributes {
        operation_index: 0,
        attributes: &wrong_mask,
    }];
    let invalid = PcuDialectProgram {
        operation_attributes: &wrong_attributes,
        ..program
    };
    assert_eq!(
        validate_dialect_program(&invalid, &support),
        Err(PcuDialectValidationError::ImmediateTypeMismatch)
    );
}

#[test]
#[allow(clippy::too_many_lines)] // One executable cross-crate composition vector owns all borrowed fixtures.
fn external_stream_programs_compose_with_explicit_wiring() {
    let first_ops = [unary("xor_mask.u32", 0, 1)];
    let second_ops = [unary("xor_mask.u32", 0, 1)];
    let inputs = [PcuDialectPort {
        name: "source",
        value: value(0),
    }];
    let outputs = [PcuDialectPort {
        name: "result",
        value: value(1),
    }];
    let first_mask = [PcuDialectImmediate {
        name: "mask",
        value: PcuDialectImmediateValue::U32(0x00ff_00ff),
    }];
    let second_mask = [PcuDialectImmediate {
        name: "mask",
        value: PcuDialectImmediateValue::U32(0x0f0f_0f0f),
    }];
    let first_attrs = [PcuDialectOperationAttributes {
        operation_index: 0,
        attributes: &first_mask,
    }];
    let second_attrs = [PcuDialectOperationAttributes {
        operation_index: 0,
        attributes: &second_mask,
    }];
    let first = PcuDialectProgram {
        fragment: PcuDialectFragment {
            dialect: DIALECT,
            version: VERSION,
            operations: &first_ops,
        },
        input_ports: &inputs,
        output_ports: &outputs,
        operation_attributes: &first_attrs,
    };
    let second = PcuDialectProgram {
        fragment: PcuDialectFragment {
            operations: &second_ops,
            ..first.fragment
        },
        operation_attributes: &second_attrs,
        ..first
    };
    let operation_specs = [PcuDialectOperationSpec {
        name: "xor_mask.u32",
        input_types: &[U32],
        result_type: Some(U32),
        effects: PcuDialectEffects::PURE,
    }];
    let port_inputs = [PcuDialectPortSpec {
        name: "source",
        value_type: U32,
    }];
    let port_outputs = [PcuDialectPortSpec {
        name: "result",
        value_type: U32,
    }];
    let immediate_specs = [PcuDialectImmediateSpec {
        name: "mask",
        scalar_type: PcuScalarType::U32,
    }];
    let operation_immediates = [PcuDialectOperationImmediateSpec {
        operation_name: "xor_mask.u32",
        attributes: &immediate_specs,
    }];
    let program_support = PcuDialectProgramSupport {
        dialect_support: support(&operation_specs),
        input_ports: &port_inputs,
        output_ports: &port_outputs,
        operation_immediates: &operation_immediates,
    };
    let bindings = [PcuDialectProgramBinding {
        first_output: "result",
        second_input: "source",
    }];
    let mut composed_ops = [first_ops[0]; 2];
    let mut composed_inputs = [inputs[0]];
    let mut composed_outputs = [outputs[0]];
    let mut composed_attrs = [first_attrs[0]; 2];
    let composed = compose_dialect_programs(
        &first,
        &second,
        &bindings,
        PcuDialectProgramComposeSupports {
            first: &program_support,
            second: &program_support,
            composed: &program_support,
        },
        PcuDialectProgramComposeStorage {
            operations: &mut composed_ops,
            input_ports: &mut composed_inputs,
            output_ports: &mut composed_outputs,
            operation_attributes: &mut composed_attrs,
        },
    )
    .unwrap();
    validate_dialect_program(&composed, &program_support).unwrap();
    assert_eq!(composed.fragment.operations[1].inputs[0].id, value(1).id);
    assert_eq!(composed.operation_attributes[1].operation_index, 1);
    let source = 0x1234_5678_u32;
    let mut values = [0_u32; 4];
    values[usize::from(composed.input_ports[0].value.id.0)] = source;
    for (operation, metadata) in composed
        .fragment
        .operations
        .iter()
        .zip(composed.operation_attributes)
    {
        let PcuDialectImmediateValue::U32(mask) = metadata.attributes[0].value else {
            unreachable!("validated support fixes the mask type")
        };
        values[usize::from(operation.result.unwrap().id.0)] =
            values[usize::from(operation.inputs[0].id.0)] ^ mask;
    }
    assert_eq!(
        values[usize::from(composed.output_ports[0].value.id.0)],
        0x1dc4_5988
    );
}
