//! Actual integer-synthesized F32 Clamp publication, fatal priority and independent core payloads.
#[path = "../../../spirv/tests/checked_binary/support/support.rs"]
mod graph;
#[path = "source/source.rs"]
#[allow(dead_code)] // All four direct ops run in paired benchmark; policies/layouts run here.
mod source;
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend,PcuVulkanError};
#[rustfmt::skip]
use pcu_facade::{global,PcuBindingRef,PcuClampedFloat,PcuClampedError,PcuDispatchFloatBinaryOp,PcuExecutionFault,PcuExecutionFaultKind,PcuFloatUnderflowPolicy,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuRangePolicy};
const OPS: [PcuDispatchFloatBinaryOp; 4] = [
    PcuDispatchFloatBinaryOp::Add,
    PcuDispatchFloatBinaryOp::Sub,
    PcuDispatchFloatBinaryOp::Mul,
    PcuDispatchFloatBinaryOp::Div,
];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
];
fn call(
    prepared: &mut impl PcuPreparedHostKernel<Error = PcuVulkanError>,
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
) -> Result<(), PcuVulkanError> {
    prepared.call(&mut [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), left),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), right),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
    ])
}
fn oracle(
    left: f32,
    right: f32,
    op: PcuDispatchFloatBinaryOp,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f32, PcuClampedError<f32>> {
    match op {
        PcuDispatchFloatBinaryOp::Add => left.pcu_clamped_add_with_policy(right, policy),
        PcuDispatchFloatBinaryOp::Sub => left.pcu_clamped_sub_with_policy(right, policy),
        PcuDispatchFloatBinaryOp::Mul => left.pcu_clamped_mul_with_policy(right, policy),
        PcuDispatchFloatBinaryOp::Div => left.pcu_clamped_div_with_policy(right, policy),
    }
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
#[allow(clippy::too_many_lines)] // One ownership/retry matrix compares earlier and later faults on the same prepared owner.
#[allow(clippy::cognitive_complexity)] // Ordered fatal/recovered publication and retry assertions preserve one native owner lifecycle.
fn completed_clamp_fault_priority_payloads_preflight_tails_and_retry() {
    let backend = PcuVulkanBackend::new().unwrap();
    let sentinel = 17.0_f32;
    for op in OPS {
        for policy in POLICIES {
            let mut graph = graph::Graph::new(7, op, policy);
            graph.range = PcuRangePolicy::Clamp;
            let mut prepared = graph
                .with(|kernel| backend.prepare_host_kernel(kernel))
                .unwrap();
            let range_right = match op {
                PcuDispatchFloatBinaryOp::Add => f32::MAX,
                PcuDispatchFloatBinaryOp::Sub => -f32::MAX,
                PcuDispatchFloatBinaryOp::Mul => 2.0,
                PcuDispatchFloatBinaryOp::Div => 0.5,
            };
            let mut left = [1.0; 7];
            let mut right = [1.0; 7];
            left[1] = f32::MAX;
            right[1] = range_right;
            left[5] = f32::MAX;
            right[5] = range_right;
            let mut output = [sentinel; 9];
            assert!(
                matches!(call(&mut prepared,&left,&right,&mut output),Err(PcuVulkanError::Fault(fault)) if fault.recovered&&fault.invocation_id==1&&fault.kind==PcuExecutionFaultKind::ArithmeticOverflow)
            );
            assert_eq!(output[1].to_bits(), f32::MAX.to_bits());
            assert_eq!(output[5].to_bits(), f32::MAX.to_bits());
            assert_eq!(
                output[7..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                vec![sentinel.to_bits(); 2]
            );
            for lane in [0, 3, 6] {
                let mut invalid = left;
                invalid[lane] = f32::from_bits(0x7f80_0001);
                let before = output.map(f32::to_bits);
                assert!(
                    matches!(call(&mut prepared,&invalid,&right,&mut output),Err(PcuVulkanError::Fault(fault)) if !fault.recovered&&fault.invocation_id==lane as u64&&fault.kind==PcuExecutionFaultKind::InvalidFloatingOperand)
                );
                assert_eq!(output.map(f32::to_bits), before);
                assert!(
                    matches!(call(&mut prepared,&left,&right,&mut output),Err(PcuVulkanError::Fault(fault)) if fault.recovered)
                );
            }
            let before = output.map(f32::to_bits);
            assert!(call(&mut prepared, &left[..6], &right, &mut output).is_err());
            assert_eq!(output.map(f32::to_bits), before);
            if matches!(
                op,
                PcuDispatchFloatBinaryOp::Mul | PcuDispatchFloatBinaryOp::Div
            ) {
                let denominator = if op == PcuDispatchFloatBinaryOp::Mul {
                    0.5
                } else {
                    2.0
                };
                for tiny in [f32::from_bits(1), f32::from_bits(0x8000_0001)] {
                    let result = call(&mut prepared, &[tiny; 7], &[denominator; 7], &mut output);
                    if policy == PcuFloatUnderflowPolicy::AllowGradualUnderflow {
                        result.unwrap();
                    } else {
                        assert!(
                            matches!(result,Err(PcuVulkanError::Fault(fault)) if fault.recovered&&fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow)
                        );
                    }
                    assert_eq!(
                        output[..7].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        vec![tiny.to_bits() & 0x8000_0000; 7]
                    );
                }
            }
        }
    }
    let mut mixed = source::mul_prepare::<f32, 3, _>(&backend).unwrap();
    for reversed in [false, true] {
        let mut left = [f32::from_bits(1), f32::MAX, 1.0];
        let mut right = [0.5, 2.0, 1.0];
        if reversed {
            left.swap(0, 1);
            right.swap(0, 1);
        }
        let mut output = [sentinel; 5];
        assert!(
            matches!(mixed(&left,&right,&mut output),Err(PcuVulkanError::Fault(fault)) if fault.recovered&&fault.invocation_id==0&&fault.kind==if reversed{PcuExecutionFaultKind::ArithmeticOverflow}else{PcuExecutionFaultKind::ArithmeticUnderflow})
        );
        let before = output.map(f32::to_bits);
        left[2] = f32::NAN;
        assert!(
            matches!(mixed(&left,&right,&mut output),Err(PcuVulkanError::Fault(fault)) if !fault.recovered&&fault.invocation_id==2)
        );
        assert_eq!(output.map(f32::to_bits), before);
    }
    let mut scale = source::scale_prepare::<f32, 7, _>(&backend).unwrap();
    let mut output = [sentinel; 9];
    assert!(
        matches!(scale(&[f32::from_bits(1);7],&1.0,&mut output),Err(PcuVulkanError::Fault(fault)) if fault.recovered&&fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(
        output[..7].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![1; 7]
    );
    let mut mul = source::mul_prepare::<f32, 7, _>(&backend).unwrap();
    assert!(
        matches!(mul(&[f32::MIN_POSITIVE;7],&[f32::from_bits(0x3f7f_ffff);7],&mut output),Err(PcuVulkanError::Fault(fault)) if fault.recovered&&fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(
        output[..7].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![f32::MIN_POSITIVE.to_bits(); 7]
    );
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn random_finite_encoded_payloads_all_operations_and_underflow_policies() {
    const N: usize = 4096;
    let backend = PcuVulkanBackend::new().unwrap();
    let mut state = 0x716f_aa4d_u32;
    let mut finite = || {
        loop {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if (state >> 23) & 255 != 255 {
                return f32::from_bits(state);
            }
        }
    };
    let left: Vec<_> = (0..N).map(|_| finite()).collect();
    let right: Vec<_> = (0..N)
        .map(|_| {
            let value = finite();
            if value == 0.0 { 1.0 } else { value }
        })
        .collect();
    for op in OPS {
        for policy in POLICIES {
            let mut graph = graph::Graph::new(u32::try_from(N).unwrap(), op, policy);
            graph.range = PcuRangePolicy::Clamp;
            let mut prepared = graph
                .with(|kernel| backend.prepare_host_kernel(kernel))
                .unwrap();
            let mut expected = vec![0; N];
            let mut first = None;
            for index in 0..N {
                expected[index] = match oracle(left[index], right[index], op, policy) {
                    Ok(value) => value,
                    Err(PcuClampedError::Range(fault)) => {
                        first.get_or_insert_with(|| PcuExecutionFault {
                            recovered: true,
                            kind: fault.kind(),
                            invocation_id: u64::try_from(index).unwrap(),
                        });
                        fault.clamped_value()
                    }
                    Err(PcuClampedError::Fatal(kind)) => {
                        panic!("finite nonzero denominator unexpectedly fatal {kind:?}")
                    }
                }
                .to_bits();
            }
            let mut output = vec![17.0; N + 3];
            let result = call(&mut prepared, &left, &right, &mut output);
            match first {
                None => result.unwrap(),
                Some(expected) => {
                    assert!(matches!(result,Err(PcuVulkanError::Fault(fault)) if fault==expected));
                }
            }
            assert_eq!(
                output[..N].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                output[N..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                vec![17.0_f32.to_bits(); 3]
            );
        }
    }
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn genuine_ordinary_clamp_grid_broadcast_and_fatal_retry() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    let left = [f32::MAX, 1.0, f32::MAX, 1.0, f32::MAX, 1.0, f32::MAX];
    let right = [0.5; 7];
    let mut output = [17.0; 9];
    assert!(
        matches!(source::div::<f32,7>(&left,&right,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered&&fault.invocation_id==0)
    );
    assert_eq!(output[0].to_bits(), f32::MAX.to_bits());
    let before = output.map(f32::to_bits);
    let mut bad = right;
    bad[3] = 0.0;
    assert!(
        matches!(source::div::<f32,7>(&left,&bad,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if !fault.recovered&&fault.invocation_id==3&&fault.kind==PcuExecutionFaultKind::DivideByZero)
    );
    assert_eq!(output.map(f32::to_bits), before);
    assert!(
        matches!(source::div::<f32,7>(&left,&right,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert!(
        matches!(source::grid::<f32,7>(&left,&right,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert!(
        matches!(source::scale::<f32,7>(&[f32::from_bits(1);7],&1.0,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert_eq!(
        output[..7].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![1; 7]
    );
    assert_eq!(
        output[7..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![17.0_f32.to_bits(); 2]
    );
}
