//! Independent small dyadic/integer encodings, whole effect closure and real retained ownership.
#[rustfmt::skip]
use super::{
    MetalTensorBinaryPlan,
    MetalSession,
    MetalTensorInput,
    MetalTensorOwner,
    MetalError,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalarType,
    PcuImplementationRequirements,
    PcuRangePolicy,
    PcuReproducibility,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuDispatchFloatBinaryOp,
    PcuMemoryPoolId,
    PcuExecutionFaultKind,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorOwnedSelectedProgram,
    TensorArithmeticRewritePolicy,
    TensorArithmeticCapability,
    TensorPointwiseGroupingPolicy,
};
const TYPES: [PcuScalarType; 20] = [
    PcuScalarType::U8,
    PcuScalarType::I8,
    PcuScalarType::U16,
    PcuScalarType::I16,
    PcuScalarType::U32,
    PcuScalarType::I32,
    PcuScalarType::U64,
    PcuScalarType::I64,
    PcuScalarType::U128,
    PcuScalarType::I128,
    PcuScalarType::U256,
    PcuScalarType::I256,
    PcuScalarType::U512,
    PcuScalarType::I512,
    PcuScalarType::F16,
    PcuScalarType::BF16,
    PcuScalarType::F8E4M3FN,
    PcuScalarType::F8E5M2,
    PcuScalarType::F32,
    PcuScalarType::F64,
];
fn requests() -> Vec<PcuImplementationRequirements> {
    let mut out = Vec::new();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for float_underflow in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ] {
                    out.push(PcuImplementationRequirements {
                        numerical_mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            reproducibility: PcuReproducibility::Unspecified,
                        },
                        float_underflow,
                        range_policy: PcuRangePolicy::Reject,
                    });
                }
            }
        }
    }
    out
}
fn selected(
    graph: Graph,
    output: fusion_pcu::dialect::tensor::ValueId,
) -> TensorOwnedSelectedProgram {
    graph
        .into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap()
}
fn program(
    scalar: PcuScalarType,
    op: PcuDispatchFloatBinaryOp,
    request: PcuImplementationRequirements,
    repeated: bool,
    unused: bool,
) -> TensorOwnedSelectedProgram {
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input([5, 13], scalar).unwrap();
    let right = if repeated {
        left
    } else {
        graph.input([5, 13], scalar).unwrap()
    };
    graph.set_numerical_options(request.numerical_options);
    let effect = match op {
        PcuDispatchFloatBinaryOp::Add => graph.add(left, right),
        PcuDispatchFloatBinaryOp::Sub => graph.sub(right, left),
        PcuDispatchFloatBinaryOp::Mul => graph.mul(left, right),
        PcuDispatchFloatBinaryOp::Div => graph.div(right, left),
    }
    .unwrap();
    if scalar.binary_float_format().is_some() {
        graph
            .set_value_float_underflow_policy(effect, request.float_underflow)
            .unwrap();
    }
    selected(graph, if unused { left } else { effect })
}
#[test]
fn detached_twenty_type_binary_schema_freezes_actual_operands_and_policies() {
    for scalar in TYPES {
        for request in requests() {
            for op in [
                PcuDispatchFloatBinaryOp::Add,
                PcuDispatchFloatBinaryOp::Sub,
                PcuDispatchFloatBinaryOp::Mul,
            ] {
                for repeated in [false, true] {
                    for unused in [false, true] {
                        let program = program(scalar, op, request, repeated, unused);
                        let plan =
                            MetalTensorBinaryPlan::assess_program(&program, request).unwrap();
                        assert_eq!(plan.input_values().len(), if repeated { 1 } else { 2 });
                        assert_eq!(plan.requirements(), request);
                        assert_eq!(plan.shape(), [5, 13]);
                        assert_eq!(
                            plan.operand_inputs(),
                            if repeated {
                                [0, 0]
                            } else if op == PcuDispatchFloatBinaryOp::Sub {
                                [1, 0]
                            } else {
                                [0, 1]
                            }
                        );
                        let clamp = PcuImplementationRequirements {
                            range_policy: PcuRangePolicy::Clamp,
                            ..request
                        };
                        assert!(MetalTensorBinaryPlan::assess_program(&program, clamp).is_err());
                        let mut portable = request;
                        portable.numerical_options.reproducibility = PcuReproducibility::PortableV1;
                        assert!(MetalTensorBinaryPlan::assess_program(&program, portable).is_err());
                        let mut wrong = request;
                        wrong.numerical_options.precision = if request.numerical_options.precision
                            == PcuPrecisionPolicy::Preserve
                        {
                            PcuPrecisionPolicy::BackendOptimized
                        } else {
                            PcuPrecisionPolicy::Preserve
                        };
                        assert!(MetalTensorBinaryPlan::assess_program(&program, wrong).is_err());
                    }
                }
            }
        }
    }
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input([65], PcuScalarType::F32).unwrap();
    let right = graph.input([65], PcuScalarType::F32).unwrap();
    let sum = graph.add(left, right).unwrap();
    let output = graph.relu(sum).unwrap();
    assert!(
        MetalTensorBinaryPlan::assess_program(
            &selected(graph, output),
            PcuImplementationRequirements::default()
        )
        .is_err()
    );
}
fn codes(scalar: PcuScalarType) -> [u64; 4] {
    match scalar {
        PcuScalarType::F16 => [0x3c00, 0x4000, 0x4200, 0x4600],
        PcuScalarType::BF16 => [0x3f80, 0x4000, 0x4040, 0x40c0],
        PcuScalarType::F8E4M3FN => [0x38, 0x40, 0x44, 0x4c],
        PcuScalarType::F8E5M2 => [0x3c, 0x40, 0x42, 0x46],
        PcuScalarType::F32 => [0x3f80_0000, 0x4000_0000, 0x4040_0000, 0x40c0_0000],
        PcuScalarType::F64 => [
            0x3ff0_0000_0000_0000,
            0x4000_0000_0000_0000,
            0x4008_0000_0000_0000,
            0x4018_0000_0000_0000,
        ],
        _ => [1, 2, 3, 6],
    }
}
fn payload(scalar: PcuScalarType, code: u64) -> Vec<u8> {
    let width = usize::from(scalar.bit_width()) / 8;
    let mut bytes = vec![0; width * 65];
    for value in bytes.chunks_exact_mut(width) {
        let n = width.min(8);
        value[..n].copy_from_slice(&code.to_le_bytes()[..n]);
    }
    bytes
}
fn host(scalar: PcuScalarType, bytes: &[u8]) -> MetalTensorInput<'_> {
    MetalTensorInput::HostBytes {
        scalar,
        elements: 65,
        bytes,
    }
}
fn resident(scalar: PcuScalarType, owner: &MetalTensorOwner) -> MetalTensorInput<'_> {
    MetalTensorInput::Resident {
        scalar,
        elements: 65,
        resource: owner.resource(),
    }
}
fn read(owner: &MetalTensorOwner, expected: &[u8]) {
    let mut actual = vec![91; expected.len() + 3];
    owner.read_bytes_into(&mut actual).unwrap();
    assert_eq!(&actual[..expected.len()], expected);
    assert_eq!(&actual[expected.len()..], [91; 3]);
}
#[test]
#[ignore = "Requires actual Metal20-type binary owned effect, mixed-input and fatal-publication proof."]
fn native_twenty_type_binary_owners_and_checked_unused_effects() {
    let session = MetalSession::open(0).unwrap();
    for scalar in TYPES {
        for request in requests() {
            for op in [
                PcuDispatchFloatBinaryOp::Add,
                PcuDispatchFloatBinaryOp::Sub,
                PcuDispatchFloatBinaryOp::Mul,
                PcuDispatchFloatBinaryOp::Div,
            ] {
                if op == PcuDispatchFloatBinaryOp::Div && scalar.binary_float_format().is_none() {
                    continue;
                }
                native_case(&session, scalar, request, op);
            }
        }
    }
}
fn native_case(
    session: &MetalSession,
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
    op: PcuDispatchFloatBinaryOp,
) {
    let [one, two, three, six] = codes(scalar);
    let a = payload(
        scalar,
        if op == PcuDispatchFloatBinaryOp::Mul {
            two
        } else {
            one
        },
    );
    let b = payload(
        scalar,
        if op == PcuDispatchFloatBinaryOp::Mul {
            three
        } else if op == PcuDispatchFloatBinaryOp::Div {
            six
        } else {
            two
        },
    );
    let result = payload(
        scalar,
        match op {
            PcuDispatchFloatBinaryOp::Add => three,
            PcuDispatchFloatBinaryOp::Sub => one,
            PcuDispatchFloatBinaryOp::Mul | PcuDispatchFloatBinaryOp::Div => six,
        },
    );
    for unused in [false, true] {
        let prepared = session
            .prepare_tensor_binary_program(
                MetalTensorBinaryPlan::assess_program(
                    &program(scalar, op, request, false, unused),
                    request,
                )
                .unwrap(),
                PcuMemoryPoolId(141),
            )
            .unwrap();
        let first = prepared
            .execute(&[host(scalar, &a), host(scalar, &b)])
            .unwrap();
        read(&first, if unused { &a } else { &result });
        let leaf = |bytes: &[u8]| {
            let mut graph = Graph::try_new().unwrap();
            let input = graph.input([5, 13], scalar).unwrap();
            session
                .prepare_tensor_program(
                    super::super::MetalTensorPlan::assess_program(&selected(graph, input), request)
                        .unwrap(),
                    PcuMemoryPoolId(141),
                )
                .unwrap()
                .execute(host(scalar, bytes))
                .unwrap()
        };
        let left = leaf(&a);
        let right = leaf(&b);
        let mixed = prepared
            .execute(&[host(scalar, &a), resident(scalar, &right)])
            .unwrap();
        read(&mixed, if unused { &a } else { &result });
        let both = prepared
            .execute(&[resident(scalar, &left), resident(scalar, &right)])
            .unwrap();
        read(&both, if unused { &a } else { &result });
        exceptional(&prepared, scalar, op, &a, &b);
        read(
            &prepared
                .execute(&[host(scalar, &a), host(scalar, &b)])
                .unwrap(),
            if unused { &a } else { &result },
        );
        read(&left, &a);
        read(&right, &b);
        drop(mixed);
        drop(both);
        drop(prepared);
        read(&first, if unused { &a } else { &result });
    }
    repeated_case(session, scalar, request, op, &a);
}
fn repeated_case(
    session: &MetalSession,
    scalar: PcuScalarType,
    request: PcuImplementationRequirements,
    op: PcuDispatchFloatBinaryOp,
    a: &[u8],
) {
    let [one, two, _, _] = codes(scalar);
    let repeated = session
        .prepare_tensor_binary_program(
            MetalTensorBinaryPlan::assess_program(
                &program(scalar, op, request, true, false),
                request,
            )
            .unwrap(),
            PcuMemoryPoolId(141),
        )
        .unwrap();
    let actual = repeated.execute(&[host(scalar, a)]).unwrap();
    let expected = payload(
        scalar,
        match op {
            PcuDispatchFloatBinaryOp::Add => two,
            PcuDispatchFloatBinaryOp::Sub => 0,
            PcuDispatchFloatBinaryOp::Mul => match scalar.binary_float_format() {
                Some(_) => match scalar {
                    PcuScalarType::F16 => 0x4400,
                    PcuScalarType::BF16 => 0x4080,
                    PcuScalarType::F8E4M3FN => 0x48,
                    PcuScalarType::F8E5M2 => 0x44,
                    PcuScalarType::F32 => 0x4080_0000,
                    PcuScalarType::F64 => 0x4010_0000_0000_0000,
                    _ => unreachable!(),
                },
                None => 4,
            },
            PcuDispatchFloatBinaryOp::Div => one,
        },
    );
    read(&actual, &expected);
}

fn exceptional(
    prepared: &super::MetalPreparedTensorBinaryProgram,
    scalar: PcuScalarType,
    op: PcuDispatchFloatBinaryOp,
    a: &[u8],
    b: &[u8],
) {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    let width = usize::from(scalar.bit_width()) / 8;
    let signed = matches!(
        scalar,
        PcuScalarType::I8
            | PcuScalarType::I16
            | PcuScalarType::I32
            | PcuScalarType::I64
            | PcuScalarType::I128
            | PcuScalarType::I256
            | PcuScalarType::I512
    );
    let kind = if scalar.binary_float_format().is_some() {
        let nan = match scalar {
            PcuScalarType::F16 => 0x7e01_u64,
            PcuScalarType::BF16 => 0x7fc1,
            PcuScalarType::F8E4M3FN | PcuScalarType::F8E5M2 => 0x7f,
            PcuScalarType::F32 => 0x7fc1_2345,
            PcuScalarType::F64 => 0x7ff8_1234_5678_9abc,
            _ => unreachable!(),
        };
        for lane in [2, 6] {
            a[lane * width..(lane + 1) * width].copy_from_slice(&nan.to_le_bytes()[..width]);
        }
        PcuExecutionFaultKind::InvalidFloatingOperand
    } else {
        for lane in [2, 6] {
            if op == PcuDispatchFloatBinaryOp::Sub {
                b[lane * width..(lane + 1) * width].fill(0);
                if signed {
                    b[(lane + 1) * width - 1] = 0x80;
                }
            } else {
                a[lane * width..(lane + 1) * width].fill(0xff);
                if signed {
                    a[(lane + 1) * width - 1] = 0x7f;
                }
            }
        }
        if op == PcuDispatchFloatBinaryOp::Sub {
            PcuExecutionFaultKind::ArithmeticUnderflow
        } else {
            PcuExecutionFaultKind::ArithmeticOverflow
        }
    };
    assert!(
        matches!(prepared.execute(&[host(scalar,&a),host(scalar,&b)]),Err(MetalError::Arithmetic(fault)) if fault.kind==kind&&fault.invocation_id==2&&!fault.recovered)
    );
    if op == PcuDispatchFloatBinaryOp::Div {
        for lane in [2, 6] {
            a[lane * width..(lane + 1) * width].fill(0);
        }
        assert!(
            matches!(prepared.execute(&[host(scalar,&a),host(scalar,&b)]),Err(MetalError::Arithmetic(fault)) if fault.kind==PcuExecutionFaultKind::DivideByZero&&fault.invocation_id==2)
        );
    }
}

#[cfg(feature = "source-tensor")]
#[path = "../../../benches/tensor_binary/source/source.rs"]
mod source;
#[cfg(feature = "source-tensor")]
fn capture<T: fusion_pcu::PcuScalar>() {
    for request in requests() {
        for operation in [
            PcuDispatchFloatBinaryOp::Add,
            PcuDispatchFloatBinaryOp::Sub,
            PcuDispatchFloatBinaryOp::Mul,
            PcuDispatchFloatBinaryOp::Div,
        ] {
            if operation == PcuDispatchFloatBinaryOp::Div && T::TYPE.binary_float_format().is_none()
            {
                continue;
            }
            let built = pcu_facade::global::__pcu_capture_tensor_program::<T, 2, _>(
                [pcu_facade::global::PcuSourceShape::Slice { length: 65 }; 2],
                request.float_underflow,
                request.numerical_mode,
                request.numerical_options,
                |capture, inputs| match operation {
                    PcuDispatchFloatBinaryOp::Add => {
                        source::add::__pcu_capture_entry::<T>(capture, inputs)
                    }
                    PcuDispatchFloatBinaryOp::Sub => {
                        source::sub::__pcu_capture_entry::<T>(capture, inputs)
                    }
                    PcuDispatchFloatBinaryOp::Mul => {
                        source::mul::__pcu_capture_entry::<T>(capture, inputs)
                    }
                    PcuDispatchFloatBinaryOp::Div => {
                        source::div::__pcu_capture_entry::<T>(capture, inputs)
                    }
                },
            )
            .unwrap();
            let plan = MetalTensorBinaryPlan::assess_program(built.program(), request).unwrap();
            assert_eq!(plan.requirements(), request);
            assert_eq!(plan.scalar_type(), T::TYPE);
        }
    }
}
#[cfg(feature = "source-tensor")]
#[test]
fn genuine_twenty_type_binary_source_capture_preserves_tuple() {
    macro_rules! captures{($($ty:ty),+)=>{$(capture::<$ty>();)+};}
    captures!(
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
        u128,
        i128,
        fusion_pcu::PcuU256,
        fusion_pcu::PcuI256,
        fusion_pcu::PcuU512,
        fusion_pcu::PcuI512,
        fusion_pcu::PcuF16Bits,
        fusion_pcu::PcuBf16Bits,
        fusion_pcu::PcuF8E4M3FnBits,
        fusion_pcu::PcuF8E5M2Bits,
        f32,
        f64
    );
}
#[test]
#[ignore = "Requires two actual Metal sessions and no-work binary preflight/API census."]
fn native_binary_input_preflight_rejects_foreign_type_and_spans_before_work() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    let scalar = PcuScalarType::F32;
    let request = PcuImplementationRequirements::default();
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input([5, 13], scalar).unwrap();
    let bytes = payload(scalar, codes(scalar)[0]);
    let owner = foreign
        .prepare_tensor_program(
            super::super::MetalTensorPlan::assess_program(&selected(graph, input), request)
                .unwrap(),
            PcuMemoryPoolId(141),
        )
        .unwrap()
        .execute(host(scalar, &bytes))
        .unwrap();
    let prepared = session
        .prepare_tensor_binary_program(
            MetalTensorBinaryPlan::assess_program(
                &program(scalar, PcuDispatchFloatBinaryOp::Add, request, false, false),
                request,
            )
            .unwrap(),
            PcuMemoryPoolId(141),
        )
        .unwrap();
    #[cfg(feature = "api-census")]
    crate::reset_api_call_census();
    assert!(matches!(
        prepared.execute(&[host(scalar, &bytes), resident(scalar, &owner)]),
        Err(MetalError::ForeignSession)
    ));
    assert!(matches!(
        prepared.execute(&[host(scalar, &bytes), host(scalar, &[])]),
        Err(MetalError::InvalidExtent)
    ));
    assert!(matches!(
        prepared.execute(&[host(scalar, &bytes), host(PcuScalarType::U32, &bytes)]),
        Err(MetalError::Unsupported)
    ));
    assert!(matches!(
        prepared.execute(&[
            host(scalar, &bytes),
            MetalTensorInput::HostBytes {
                scalar,
                elements: 64,
                bytes: &bytes
            }
        ]),
        Err(MetalError::InvalidExtent)
    ));
    #[cfg(feature = "api-census")]
    assert_eq!(
        crate::api_call_census(),
        crate::MetalApiCallCensus::default()
    );
    read(&owner, &bytes);
    read(
        &prepared
            .execute(&[host(scalar, &bytes), host(scalar, &bytes)])
            .unwrap(),
        &payload(scalar, codes(scalar)[1]),
    );
}
