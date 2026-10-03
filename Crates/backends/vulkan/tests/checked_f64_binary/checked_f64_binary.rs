//! Actual U32-only binary64 checked arithmetic against the independent integer core oracle.
#[path = "../../../spirv/tests/checked_binary/support/support.rs"]
mod graph;
#[path = "../checked_binary/source/source.rs"]
#[allow(dead_code)] // F32 source peers remain in their original independent fixture.
mod reject_source;
#[path = "../clamped_binary/source/source.rs"]
#[allow(dead_code)] // Canonical family is also exercised in paired native benchmarks.
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingRef,
    PcuClampedError,
    PcuClampedFloat,
    PcuDispatchFloatBinaryOp,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend,PcuVulkanError};
const OPS: [PcuDispatchFloatBinaryOp; 4] = [
    PcuDispatchFloatBinaryOp::Add,
    PcuDispatchFloatBinaryOp::Sub,
    PcuDispatchFloatBinaryOp::Mul,
    PcuDispatchFloatBinaryOp::Div,
];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
];
fn oracle(
    left: f64,
    right: f64,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f64, PcuClampedError<f64>> {
    match op {
        PcuDispatchFloatBinaryOp::Add => left.pcu_clamped_add_with_policy(right, policy),
        PcuDispatchFloatBinaryOp::Sub => left.pcu_clamped_sub_with_policy(right, policy),
        PcuDispatchFloatBinaryOp::Mul => left.pcu_clamped_mul_with_policy(right, policy),
        PcuDispatchFloatBinaryOp::Div => left.pcu_clamped_div_with_policy(right, policy),
    }
}
fn call(
    prepared: &mut impl PcuPreparedHostKernel<Error = PcuVulkanError>,
    left: &[f64],
    right: &[f64],
    output: &mut [f64],
) -> Result<(), PcuVulkanError> {
    prepared.call(&mut [
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), right),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), left),
    ])
}
const fn diagnostic(
    extent: u32,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> graph::Graph {
    let mut graph = graph::Graph::new(extent, op, policy);
    graph.scalar = PcuScalarType::F64;
    graph.range = range;
    graph
}
const fn random(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    f64::from_bits(*state)
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
#[allow(clippy::too_many_lines)] // Complete raw-bit policy/range oracle matrix uses one retained owner per profile.
fn actual_binary64_raw_bits_rounding_tininess_and_signed_zero() {
    const COUNT: usize = 8192;
    let backend = PcuVulkanBackend::new().unwrap();
    let edges = [
        0,
        0x8000_0000_0000_0000,
        1,
        0x8000_0000_0000_0001,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x0010_0000_0000_0001,
        0x3fef_ffff_ffff_ffff,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0001,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
    ];
    for op in OPS {
        for policy in POLICIES {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let mut prepared = diagnostic(u32::try_from(COUNT).unwrap(), op, policy, range)
                    .with(|kernel| backend.prepare_host_kernel(kernel))
                    .unwrap();
                let mut state = 0x71ef_350d_32a4_b691;
                let mut left = vec![0.0; COUNT];
                let mut right = vec![0.0; COUNT];
                let mut expected = vec![0; COUNT];
                let mut recovered = None;
                for lane in 0..COUNT {
                    left[lane] = if lane < edges.len() * edges.len() {
                        f64::from_bits(edges[lane / edges.len()])
                    } else {
                        random(&mut state)
                    };
                    right[lane] = if lane < edges.len() * edges.len() {
                        f64::from_bits(edges[lane % edges.len()])
                    } else {
                        random(&mut state)
                    };
                    let value = match oracle(left[lane], right[lane], op, policy) {
                        Ok(value) => value,
                        Err(PcuClampedError::Range(fault)) if range == PcuRangePolicy::Clamp => {
                            recovered.get_or_insert_with(|| PcuExecutionFault {
                                recovered: true,
                                kind: fault.kind(),
                                invocation_id: lane as u64,
                            });
                            fault.clamped_value()
                        }
                        Err(_) => {
                            left[lane] = 1.0;
                            right[lane] = 1.0;
                            oracle(left[lane], right[lane], op, policy).unwrap()
                        }
                    };
                    expected[lane] = value.to_bits();
                }
                let mut output = vec![17.0; COUNT + 3];
                let result = call(&mut prepared, &left, &right, &mut output);
                match recovered {
                    Some(expected) => assert!(
                        matches!(result,Err(PcuVulkanError::Fault(actual)) if actual==expected)
                    ),
                    None => result.unwrap(),
                }
                for lane in 0..COUNT {
                    assert_eq!(
                        output[lane].to_bits(),
                        expected[lane],
                        "{op:?}/{policy:?}/{range:?} lane{lane}: {:016x} {:016x}",
                        left[lane].to_bits(),
                        right[lane].to_bits()
                    );
                }
                assert_eq!(
                    output[COUNT..]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    vec![17.0_f64.to_bits(); 3]
                );
            }
        }
    }
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
#[allow(clippy::too_many_lines)] // Every original owner is reused through recovered/fatal/preflight/retry permutations.
#[allow(clippy::cognitive_complexity)] // Ordered fatal/recovered publication and retry assertions preserve one native owner lifecycle.
fn fatal_priority_complete_rollback_preflight_tails_and_retry() {
    let backend = PcuVulkanBackend::new().unwrap();
    for op in OPS {
        for policy in POLICIES {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let mut prepared = diagnostic(7, op, policy, range)
                    .with(|kernel| backend.prepare_host_kernel(kernel))
                    .unwrap();
                let mut left = [1.0; 7];
                let mut right = [1.0; 7];
                let mut output = [17.0; 9];
                left[1] = f64::MAX;
                left[5] = -f64::MAX;
                right[1] = match op {
                    PcuDispatchFloatBinaryOp::Add => f64::MAX,
                    PcuDispatchFloatBinaryOp::Sub => -f64::MAX,
                    PcuDispatchFloatBinaryOp::Mul => 2.0,
                    PcuDispatchFloatBinaryOp::Div => 0.5,
                };
                right[5] = match op {
                    PcuDispatchFloatBinaryOp::Add => -f64::MAX,
                    PcuDispatchFloatBinaryOp::Sub => f64::MAX,
                    _ => right[1],
                };
                let error = call(&mut prepared, &left, &right, &mut output).unwrap_err();
                assert!(
                    matches!(error,PcuVulkanError::Fault(fault) if fault.recovered==(range==PcuRangePolicy::Clamp)&&fault.invocation_id==1&&fault.kind==PcuExecutionFaultKind::ArithmeticOverflow)
                );
                if range == PcuRangePolicy::Clamp {
                    assert_eq!(output[1].to_bits(), f64::MAX.to_bits());
                    assert_eq!(output[5].to_bits(), (-f64::MAX).to_bits());
                } else {
                    assert_eq!(output.map(f64::to_bits), [17.0_f64; 9].map(f64::to_bits));
                }
                for lane in [0, 3, 6] {
                    let before = output.map(f64::to_bits);
                    let mut invalid = left;
                    invalid[lane] = f64::from_bits(0x7ff0_0000_0000_0001);
                    let error = call(&mut prepared, &invalid, &right, &mut output).unwrap_err();
                    let expected_lane = if range == PcuRangePolicy::Reject && lane > 1 {
                        1
                    } else {
                        lane
                    };
                    let expected_kind = if range == PcuRangePolicy::Reject && lane > 1 {
                        PcuExecutionFaultKind::ArithmeticOverflow
                    } else {
                        PcuExecutionFaultKind::InvalidFloatingOperand
                    };
                    assert!(
                        matches!(error,PcuVulkanError::Fault(fault) if !fault.recovered&&fault.invocation_id==expected_lane as u64&&fault.kind==expected_kind)
                    );
                    assert_eq!(output.map(f64::to_bits), before);
                }
                let before = output.map(f64::to_bits);
                assert!(matches!(
                    call(&mut prepared, &left[..6], &right, &mut output),
                    Err(PcuVulkanError::InvalidArguments)
                ));
                assert_eq!(output.map(f64::to_bits), before);
                assert!(
                    prepared
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[1.0_f32; 7]),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output)
                        ])
                        .is_err()
                );
                assert_eq!(output.map(f64::to_bits), before);
                call(&mut prepared, &[1.0; 7], &[1.0; 7], &mut output).unwrap();
                let expected = oracle(1.0, 1.0, op, policy).unwrap();
                assert_eq!(
                    output[..7].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    vec![expected.to_bits(); 7]
                );
                assert_eq!(
                    output[7..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    vec![17.0_f64.to_bits(); 2]
                );
                if op == PcuDispatchFloatBinaryOp::Div {
                    let before = output.map(f64::to_bits);
                    let mut zero = [1.0; 7];
                    zero[6] = -0.0;
                    assert!(
                        matches!(call(&mut prepared,&[1.0;7],&zero,&mut output),Err(PcuVulkanError::Fault(fault)) if fault.kind==PcuExecutionFaultKind::DivideByZero&&fault.invocation_id==6&&!fault.recovered)
                    );
                    assert_eq!(output.map(f64::to_bits), before);
                }
            }
        }
    }
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn genuine_f64_source_direct_grid_broadcast_recovered_and_fatal() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let backend = PcuVulkanBackend::new().unwrap();
    let left = [f64::MAX, 1.0, -f64::MAX];
    let right = [0.5; 3];
    let mut output = [17.0; 5];
    let mut prepared = source::div_prepare::<f64, 3, _>(&backend).unwrap();
    for route in 0..2 {
        let fault = if route == 0 {
            match prepared(&left, &right, &mut output).unwrap_err() {
                PcuVulkanError::Fault(f) => f,
                e => panic!("{e:?}"),
            }
        } else {
            match source::div::<f64, 3>(&left, &right, &mut output).unwrap_err() {
                global::PcuExecutionError::ArithmeticFault(f) => f,
                e => panic!("{e:?}"),
            }
        };
        assert!(fault.recovered);
        assert_eq!(fault.invocation_id, 0);
        assert_eq!(
            output.map(f64::to_bits),
            [f64::MAX, 2.0, -f64::MAX, 17.0, 17.0].map(f64::to_bits)
        );
    }
    let mut invalid = right;
    invalid[2] = 0.0;
    let before = output.map(f64::to_bits);
    assert!(
        matches!(source::div::<f64,3>(&left,&invalid,&mut output),Err(global::PcuExecutionError::ArithmeticFault(f)) if !f.recovered&&f.invocation_id==2)
    );
    assert_eq!(output.map(f64::to_bits), before);
    source::grid::<f64, 3>(&[f64::from_bits(1), -0.0, 1.0], &[2.0; 3], &mut output).unwrap();
    assert_eq!(
        output[..3].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [0.0, -0.0, 0.5].map(f64::to_bits)
    );
    assert!(
        matches!(source::scale::<f64,3>(&[f64::MIN_POSITIVE,1.0,-0.0],&0.5,&mut output),Err(global::PcuExecutionError::ArithmeticFault(f)) if f.recovered&&f.kind==PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(output[0].to_bits(), f64::MIN_POSITIVE.to_bits() / 2);
    reject_source::f64_source::add::<3>(&[1.0, -0.0, 2.0], &[2.0, -0.0, 3.0], &mut output).unwrap();
    assert_eq!(
        output[..3].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [3.0, -0.0, 5.0].map(f64::to_bits)
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn original_binding_broadcast_grid_swapped_repeated_ssa_and_short_preflight() {
    let backend = PcuVulkanBackend::new().unwrap();
    for op in OPS {
        for grid in [false, true] {
            for broadcast in [[false, false], [true, false], [false, true], [true, true]] {
                for operands in [[1, 2], [2, 1], [1, 1], [2, 2]] {
                    let mut graph = diagnostic(
                        7,
                        op,
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuRangePolicy::Reject,
                    );
                    graph.grid = grid;
                    graph.reverse_loads = grid;
                    graph.broadcast = broadcast;
                    graph.operands = operands;
                    let mut prepared = graph
                        .with(|kernel| backend.prepare_host_kernel(kernel))
                        .unwrap();
                    let left = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
                    let right = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
                    let mut output = [17.0; 9];
                    let left = &left[..if broadcast[0] { 1 } else { 7 }];
                    let right = &right[..if broadcast[1] { 1 } else { 7 }];
                    call(&mut prepared, left, right, &mut output).unwrap();
                    for (lane, value) in output[..7].iter().enumerate() {
                        let operand = |binding| {
                            let bank = usize::from(binding - 1);
                            let index = if broadcast[bank] { 0 } else { lane };
                            if bank == 0 { left[index] } else { right[index] }
                        };
                        assert_eq!(
                            value.to_bits(),
                            oracle(
                                operand(operands[0]),
                                operand(operands[1]),
                                op,
                                PcuFloatUnderflowPolicy::IeeeAfterRounding
                            )
                            .unwrap()
                            .to_bits()
                        );
                    }
                    assert_eq!(
                        output[7..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        vec![17.0_f64.to_bits(); 2]
                    );
                    let before = output.map(f64::to_bits);
                    assert!(matches!(
                        call(&mut prepared, &left[..left.len() - 1], right, &mut output),
                        Err(PcuVulkanError::InvalidArguments)
                    ));
                    assert_eq!(output.map(f64::to_bits), before);
                    call(&mut prepared, left, right, &mut output).unwrap();
                    assert_eq!(output.map(f64::to_bits), before);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn tiny_inexact_exact_subnormal_and_rounded_minimum_normal_faults() {
    let backend = PcuVulkanBackend::new().unwrap();
    let cases = [
        (PcuDispatchFloatBinaryOp::Mul, f64::from_bits(1), 0.5),
        (
            PcuDispatchFloatBinaryOp::Mul,
            f64::from_bits(0x8000_0000_0000_0001),
            0.5,
        ),
        (PcuDispatchFloatBinaryOp::Mul, f64::MIN_POSITIVE, 0.5),
        (
            PcuDispatchFloatBinaryOp::Mul,
            f64::MIN_POSITIVE,
            f64::from_bits(0x3fef_ffff_ffff_ffff),
        ),
        (
            PcuDispatchFloatBinaryOp::Div,
            f64::MIN_POSITIVE,
            f64::from_bits(0x3ff0_0000_0000_0001),
        ),
        (PcuDispatchFloatBinaryOp::Div, f64::from_bits(1), 2.0),
        (PcuDispatchFloatBinaryOp::Add, f64::from_bits(1), 0.0),
        (
            PcuDispatchFloatBinaryOp::Sub,
            f64::MIN_POSITIVE,
            f64::from_bits(0x000f_ffff_ffff_ffff),
        ),
    ];
    for (op, left, right) in cases {
        for policy in POLICIES {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let mut prepared = diagnostic(1, op, policy, range)
                    .with(|kernel| backend.prepare_host_kernel(kernel))
                    .unwrap();
                let mut output = [17.0; 3];
                let result = call(&mut prepared, &[left], &[right], &mut output);
                match oracle(left, right, op, policy) {
                    Ok(value) => {
                        result.unwrap();
                        assert_eq!(output[0].to_bits(), value.to_bits());
                    }
                    Err(PcuClampedError::Range(fault)) => {
                        assert!(
                            matches!(result,Err(PcuVulkanError::Fault(actual)) if actual.kind==fault.kind()&&actual.recovered==(range==PcuRangePolicy::Clamp)&&actual.invocation_id==0)
                        );
                        assert_eq!(
                            output[0].to_bits(),
                            if range == PcuRangePolicy::Clamp {
                                fault.clamped_value().to_bits()
                            } else {
                                17.0_f64.to_bits()
                            }
                        );
                    }
                    Err(PcuClampedError::Fatal(kind)) => {
                        panic!("finite tiny fixture unexpectedly fatal {kind:?}")
                    }
                }
                assert_eq!(
                    output[1..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    vec![17.0_f64.to_bits(); 2]
                );
            }
        }
    }
}
