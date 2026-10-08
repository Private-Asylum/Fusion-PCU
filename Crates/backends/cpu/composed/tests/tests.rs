//! Genuine source fixtures for detached private publication and store/load sequencing.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuExecutionFaultKind,
    PcuHostKernelBackend,
    PcuKernelId,
    PcuReproducibility,
    PcuValueType,
    PcuValueTypeCaps,
};
use super::*;
use pcu_facade::pcu;
#[path = "integer/integer.rs"]
mod integer;
#[path = "offers/offers.rs"]
mod offers;
#[path = "single/single.rs"]
mod single;

#[pcu(invocations = 4, crate_path = ::pcu_facade)]
fn composed<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = (left[id] + right[id]) * right[id];
}
#[pcu(invocations = 4, crate_path = ::pcu_facade)]
fn constant(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + 1.0) * 2.0;
}
#[pcu(invocations = 4, crate_path = ::pcu_facade)]
fn unread<T: PcuCheckedFloat>(ignored: &[T], input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}

fn width<T: PcuCheckedFloat>(one: T, two: T, sentinel: T, id: u32) {
    let backend = PcuCpuCheckedComposedMap::<T>::new();
    let mut call = composed_prepare::<T, _>(&backend).unwrap();
    let mut output = [sentinel; 6];
    call(&[one; 4], &[one; 4], &mut output).unwrap();
    for value in &output[..4] {
        assert_eq!(value.encode_le().as_ref(), two.encode_le().as_ref());
    }
    for value in &output[4..] {
        assert_eq!(value.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
    let bindings = composed_bindings::<T>();
    composed_ir::<T>(&bindings).unwrap().with_ir(|kernel| {
        let mut plan = backend.prepare_host_kernel(kernel).unwrap();
        for (input, swap_position) in [(one, 2), (two, 1)] {
            let input = [input; 4];
            let right = [one; 4];
            let mut output = [sentinel; 6];
            let mut arguments = [
                PcuHostArgument::read(bindings[0].reference(), &input),
                PcuHostArgument::read(bindings[1].reference(), &right),
                PcuHostArgument::read_write(bindings[2].reference(), &mut output),
            ];
            arguments.swap(0, swap_position);
            plan.call(&mut arguments).unwrap();
            let expected = input[0]
                .pcu_checked_add(one)
                .unwrap()
                .pcu_checked_mul(one)
                .unwrap();
            for actual in &output[..4] {
                assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
            }
            for actual in &output[4..] {
                assert_eq!(actual.encode_le().as_ref(), sentinel.encode_le().as_ref());
            }
        }
        assert_eq!(plan.local_id(), id);
        assert_eq!(plan.workspace_bytes(), 4 * T::HOST_SIZE);
        assert_eq!(plan.argument_count(), 3);
        assert_eq!(plan.requirements(), kernel.numerical_requirements);
        let mut portable = *kernel;
        portable
            .numerical_requirements
            .numerical_options
            .reproducibility = PcuReproducibility::PortableV1;
        assert!(matches!(
            backend.prepare_host_kernel(&portable),
            Err(PcuCpuComposedMapError::UnsupportedProfile)
        ));
    });
    let mut ignored = unread_prepare::<T, _>(&backend).unwrap();
    ignored(&[], &[one; 4], &mut output).unwrap();
    let mut unified = composed_prepare::<T, _>(&crate::PcuCpuHostBackend::scalar()).unwrap();
    unified(&[one; 4], &[one; 4], &mut output).unwrap();
    for value in &output[..4] {
        assert_eq!(value.encode_le().as_ref(), two.encode_le().as_ref());
    }
}
#[test]
fn six_format_real_prepared_source_and_exact_cold_ids() {
    width(
        fusion_pcu::PcuF16Bits::from_bits(0x3c00),
        fusion_pcu::PcuF16Bits::from_bits(0x4000),
        fusion_pcu::PcuF16Bits::from_bits(0x4200),
        17408,
    );
    width(
        fusion_pcu::PcuBf16Bits::from_bits(0x3f80),
        fusion_pcu::PcuBf16Bits::from_bits(0x4000),
        fusion_pcu::PcuBf16Bits::from_bits(0x4040),
        17409,
    );
    width(
        fusion_pcu::PcuF8E4M3FnBits::from_bits(0x38),
        fusion_pcu::PcuF8E4M3FnBits::from_bits(0x40),
        fusion_pcu::PcuF8E4M3FnBits::from_bits(0x44),
        17410,
    );
    width(
        fusion_pcu::PcuF8E5M2Bits::from_bits(0x3c),
        fusion_pcu::PcuF8E5M2Bits::from_bits(0x40),
        fusion_pcu::PcuF8E5M2Bits::from_bits(0x42),
        17411,
    );
    width(1_f32, 2_f32, 3_f32, 17412);
    width(1_f64, 2_f64, 3_f64, 17413);
}
#[test]
fn recovered_payload_continues_but_later_fatal_rolls_back_all_outputs() {
    let policy = PcuFloatUnderflowPolicy::RejectSubnormalResult;
    let range = PcuRangePolicy::Clamp;
    let mut call = plan(
        &[
            load(0, 0, false),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                result: PcuDispatchValueId(1),
                op: PcuDispatchFloatUnaryOp::Relu,
                value_type: PcuValueType::f32(),
                value: PcuDispatchValueId(0),
                underflow_policy: policy,
                range_policy: range,
            }),
            store(2, 1),
            load(2, 1, false),
            binary(3, 0, 2, PcuDispatchFloatBinaryOp::Div, range, policy),
            binary(4, 3, 0, PcuDispatchFloatBinaryOp::Add, range, policy),
            store(3, 4),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ],
        true,
        false,
        range,
        policy,
    );
    let tiny = f32::from_bits(1);
    let mut first = [7_f32; 6];
    let mut second = [8_f32; 6];
    let recovered = dual_call(
        &mut call,
        &[tiny, 1., 2., 3.],
        &[1.; 4],
        &mut first,
        &mut second,
    )
    .unwrap_err()
    .fault()
    .unwrap();
    assert!(recovered.recovered);
    assert_eq!(recovered.invocation_id, 0);
    assert_eq!(recovered.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
    assert_eq!(first[0].to_bits(), 1);
    assert_eq!(second[0].to_bits(), 2);
    assert_eq!(&first[4..], &[7.; 2]);
    assert_eq!(&second[4..], &[8.; 2]);
    let old_first = first.map(f32::to_bits);
    let old_second = second.map(f32::to_bits);
    let fatal = dual_call(
        &mut call,
        &[tiny, 1., 2., 3.],
        &[1., 1., 0., 1.],
        &mut first,
        &mut second,
    )
    .unwrap_err()
    .fault()
    .unwrap();
    assert!(!fatal.recovered);
    assert_eq!(fatal.invocation_id, 2);
    assert_eq!(fatal.kind, PcuExecutionFaultKind::DivideByZero);
    assert_eq!(first.map(f32::to_bits), old_first);
    assert_eq!(second.map(f32::to_bits), old_second);
    assert!(dual_call(&mut call, &[1.; 4], &[1.; 4], &mut first, &mut [8.; 3]).is_err());
    assert_eq!(first.map(f32::to_bits), old_first);
    dual_call(&mut call, &[1.; 4], &[1.; 4], &mut first, &mut second).unwrap();
    assert_eq!(&second[..4], &[2.; 4]);
}
#[test]
fn private_same_lane_store_is_visible_to_later_load_and_hazard_is_rejected() {
    let backend = PcuCpuCheckedComposedMap::<f32>::new();
    let body = [
        load(0, 0, false),
        load(1, 1, false),
        binary(
            2,
            0,
            1,
            PcuDispatchFloatBinaryOp::Add,
            PcuRangePolicy::Reject,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        store(0, 2),
        load(3, 0, false),
        binary(
            4,
            3,
            1,
            PcuDispatchFloatBinaryOp::Mul,
            PcuRangePolicy::Reject,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ),
        store(2, 4),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let mut call = plan(
        &body,
        false,
        true,
        PcuRangePolicy::Reject,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    );
    let mut state = [1_f32; 6];
    let mut output = [7_f32; 6];
    call.call(&mut [
        PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut state),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), &[2_f32; 4]),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
    ])
    .unwrap();
    assert_eq!(
        state.map(f32::to_bits),
        [3_f32, 3., 3., 3., 1., 1.].map(f32::to_bits)
    );
    assert_eq!(
        output.map(f32::to_bits),
        [6_f32, 6., 6., 6., 7., 7.].map(f32::to_bits)
    );
    let old_state = state.map(f32::to_bits);
    let old_output = output.map(f32::to_bits);
    let fault = call
        .call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut state),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &[2_f32, 2., f32::MAX, 2.]),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
        ])
        .unwrap_err()
        .fault()
        .unwrap();
    assert_eq!(fault.invocation_id, 2);
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(state.map(f32::to_bits), old_state);
    assert_eq!(output.map(f32::to_bits), old_output);
    let mut hazard = body;
    hazard[0] = load(0, 0, true);
    assert!(
        try_plan(
            &hazard,
            false,
            true,
            PcuRangePolicy::Reject,
            PcuFloatUnderflowPolicy::IeeeAfterRounding
        )
        .is_err()
    );
    let mut constant = constant_prepare(&backend).unwrap();
    constant(&[2.; 4], &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [6_f32, 6., 6., 6., 7., 7.].map(f32::to_bits)
    );
}

fn load(result: u16, binding: u32, zero: bool) -> PcuDispatchOp<'static> {
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: PcuDispatchValueId(result),
        binding: PcuBindingRef::new(0, binding),
        index: if zero {
            PcuDispatchIndex::BindingElementZero
        } else {
            PcuDispatchIndex::InvocationId
        },
    })
}
fn store(binding: u32, value: u16) -> PcuDispatchOp<'static> {
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: PcuBindingRef::new(0, binding),
        value: PcuDispatchValueId(value),
        index: PcuDispatchIndex::InvocationId,
    })
}
fn binary(
    result: u16,
    left: u16,
    right: u16,
    op: PcuDispatchFloatBinaryOp,
    range_policy: PcuRangePolicy,
    underflow_policy: PcuFloatUnderflowPolicy,
) -> PcuDispatchOp<'static> {
    PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
        result: PcuDispatchValueId(result),
        lhs: PcuDispatchValueId(left),
        rhs: PcuDispatchValueId(right),
        op,
        value_type: PcuValueType::f32(),
        range_policy,
        underflow_policy,
    })
}
fn try_plan(
    body: &[PcuDispatchOp<'_>],
    four: bool,
    mutable: bool,
    range_policy: PcuRangePolicy,
    float_underflow: PcuFloatUnderflowPolicy,
) -> Result<PcuCpuPreparedComposedMap, PcuCpuComposedMapError> {
    let binding = |index, access| {
        PcuBinding::value(
            None,
            0,
            index,
            PcuBindingStorageClass::Storage,
            access,
            PcuValueType::f32(),
        )
    };
    let bindings = [
        binding(
            0,
            if mutable {
                PcuBindingAccess::ReadWrite
            } else {
                PcuBindingAccess::ReadOnly
            },
        ),
        binding(1, PcuBindingAccess::ReadOnly),
        binding(2, PcuBindingAccess::ReadWrite),
        binding(3, PcuBindingAccess::ReadWrite),
    ];
    PcuCpuCheckedComposedMap::<f32>::new().prepare_host_kernel(&PcuDispatchKernelIr {
        numerical_requirements: PcuImplementationRequirements {
            range_policy,
            float_underflow,
            ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
        },
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "private_composed",
            logical_shape: [4, 1, 1],
        },
        bindings: &bindings[..if four { 4 } else { 3 }],
        ports: &[],
        parameters: &[],
        ops: body,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    })
}
fn plan(
    body: &[PcuDispatchOp<'_>],
    four: bool,
    mutable: bool,
    range: PcuRangePolicy,
    underflow: PcuFloatUnderflowPolicy,
) -> PcuCpuPreparedComposedMap {
    try_plan(body, four, mutable, range, underflow).unwrap()
}
fn dual_call(
    plan: &mut PcuCpuPreparedComposedMap,
    input: &[f32],
    divisor: &[f32],
    first: &mut [f32],
    second: &mut [f32],
) -> Result<(), PcuCpuComposedMapError> {
    plan.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), divisor),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), first),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 3), second),
    ])
}

#[test]
fn sparse_ssa_constants_and_changing_inputs_use_dense_initialized_registers() {
    let range = PcuRangePolicy::Reject;
    let underflow = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    let mut call = plan(
        &[
            load(255, 0, false),
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(213),
                value: fusion_pcu::PcuParameterValue::F32(2_f32.to_bits()),
            }),
            binary(
                117,
                255,
                213,
                PcuDispatchFloatBinaryOp::Mul,
                range,
                underflow,
            ),
            binary(3, 117, 255, PcuDispatchFloatBinaryOp::Add, range, underflow),
            store(2, 3),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ],
        false,
        false,
        range,
        underflow,
    );
    // Sparse, descending source IDs occupy just four densely assigned slots.
    assert!(matches!(
        call.program.steps()[0],
        Step::Load { result: 0, .. }
    ));
    assert!(matches!(
        call.program.steps()[1],
        Step::Binary {
            result: 2,
            left: 0,
            right: 1,
            ..
        }
    ));
    for input in [[1., 2., 3., 4.], [4., 3., 2., 1.]] {
        let mut output = [99_f32; 6];
        call.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), &[] as &[f32]),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
        ])
        .unwrap();
        assert_eq!(&output[..4], &input.map(|value| value * 3.));
        assert_eq!(&output[4..], &[99.; 2]);
    }
}

#[test]
fn malformed_ssa_is_rejected_before_dense_execution() {
    let range = PcuRangePolicy::Reject;
    let underflow = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    for invalid in [
        [
            load(0, 0, false),
            load(0, 1, false),
            binary(2, 0, 0, PcuDispatchFloatBinaryOp::Add, range, underflow),
            store(2, 2),
        ],
        [
            load(0, 0, false),
            binary(2, 0, 1, PcuDispatchFloatBinaryOp::Add, range, underflow),
            load(1, 1, false),
            store(2, 2),
        ],
    ] {
        let mut body = invalid.to_vec();
        body.push(PcuDispatchOp::Control(PcuDispatchControlOp::Return));
        assert!(try_plan(&body, false, false, range, underflow).is_err());
    }
}

#[path = "arguments/arguments.rs"]
mod arguments;

#[path = "lane_blocks/lane_blocks.rs"]
mod lane_blocks;
