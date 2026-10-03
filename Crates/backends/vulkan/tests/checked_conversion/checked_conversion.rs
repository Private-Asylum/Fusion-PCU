//! Actual mixed-width rounding, complete Clamp payloads and fatal rollback through U32-only shaders.
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../../../spirv/tests/checked_conversion/graph/graph.rs"]
mod graph;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "../../../cpu/tests/prepared_conversion/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use pcu_facade::{PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuDispatchCheckedFloatConversion as Conversion,PcuFloatUnderflowPolicy as Policy,PcuRangePolicy,PcuExecutionFaultKind as Kind,PcuExecutionFault};
use fusion_pcu_vulkan::{PcuVulkanBackend, PcuVulkanError};
fn fault(result: Result<(), PcuVulkanError>) -> Option<PcuExecutionFault> {
    result.err().map(|error| {
        let PcuVulkanError::Fault(fault) = error else {
            panic!("unexpected native error {error:?}")
        };
        fault
    })
}
fn prepare(
    backend: &PcuVulkanBackend,
    conversion: Conversion,
    extent: usize,
    range: PcuRangePolicy,
    policy: Policy,
    broadcast: bool,
    grid: bool,
) -> fusion_pcu_vulkan::PcuVulkanPreparedHost {
    graph::with(
        conversion,
        u32::try_from(extent).unwrap(),
        range,
        policy,
        broadcast,
        grid,
        |kernel| backend.prepare_host_kernel(kernel).unwrap(),
    )
}
fn representations() -> (Vec<f64>, Vec<f32>) {
    let mut narrow = Vec::new();
    let mut state = 0x1d3a_75c9_89ab_cd01_u64;
    for exponent in 0..2047_u64 {
        for sample in 0..64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let fraction = match sample {
                0 => 0,
                1 => 1,
                2 => 0x000f_ffff_ffff_ffff,
                3 => 0x000f_ffff_e000_0000,
                4 => 0x000f_ffff_f000_0000,
                5 => 0x000f_ffff_f800_0000,
                6 => 0x0000_0000_1000_0000,
                7 => 0x0000_0000_3000_0000,
                _ => state & 0x000f_ffff_ffff_ffff,
            };
            narrow.push(f64::from_bits(
                (state & 0x8000_0000_0000_0000) | (exponent << 52) | fraction,
            ));
        }
    }
    let mut widen = Vec::new();
    let mut state = 0xa359_8b13_u32;
    for exponent in 0..255_u32 {
        for sample in 0..256 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let fraction = match sample {
                0 => 0,
                1 => 1,
                2 => 0x007f_ffff,
                _ => state & 0x007f_ffff,
            };
            widen.push(f32::from_bits(
                (state & 0x8000_0000) | (exponent << 23) | fraction,
            ));
        }
    }
    (narrow, widen)
}
fn verify_widen(backend: &PcuVulkanBackend, widen: &[f32], policy: Policy) {
    let mut plan = prepare(
        backend,
        Conversion::F32ToF64,
        widen.len(),
        PcuRangePolicy::Reject,
        policy,
        false,
        false,
    );
    let mut output = vec![99.0_f64; widen.len() + 3];
    plan.call(&mut [
        PcuHostArgument::read_write(graph::OUTPUT, &mut output),
        PcuHostArgument::read(graph::INPUT, widen),
    ])
    .unwrap();
    for (value, input) in output.iter().zip(widen) {
        assert_eq!(value.to_bits(), oracle::widen(input.to_bits()).unwrap());
    }
    assert_eq!(
        output[widen.len()..]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [99.0_f64.to_bits(); 3]
    );
}
#[test]
#[ignore = "requires actual Vulkan device"]
fn generated_raw_bits_match_independent_packing_oracle() {
    let (backend, _) = device::selected();
    let (narrow, widen) = representations();
    for policy in [
        Policy::IeeeAfterRounding,
        Policy::AllowGradualUnderflow,
        Policy::RejectSubnormalResult,
    ] {
        let expected = narrow
            .iter()
            .map(|value| oracle::narrow(value.to_bits(), policy).unwrap())
            .collect::<Vec<_>>();
        let first = expected.iter().enumerate().find_map(|(lane, (_, notice))| {
            notice.map(|kind| PcuExecutionFault {
                kind,
                invocation_id: u64::try_from(lane).unwrap(),
                recovered: true,
            })
        });
        let mut plan = prepare(
            &backend,
            Conversion::F64ToF32,
            narrow.len(),
            PcuRangePolicy::Clamp,
            policy,
            false,
            true,
        );
        let mut output = vec![99.0_f32; narrow.len() + 3];
        let observed = fault(plan.call(&mut [
            PcuHostArgument::read_write(graph::OUTPUT, &mut output),
            PcuHostArgument::read(graph::INPUT, &narrow),
        ]));
        assert_eq!(observed, first);
        for (value, (bits, _)) in output.iter().zip(&expected) {
            assert_eq!(value.to_bits(), *bits);
        }
        assert_eq!(
            output[narrow.len()..]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [99.0_f32.to_bits(); 3]
        );
        let mut changed = narrow.clone();
        let last = changed.len() - 1;
        changed[last] = f64::from_bits(0x7ff0_0000_0000_0001);
        let saved = output.clone();
        let failed = fault(plan.call(&mut [
            PcuHostArgument::read(graph::INPUT, &changed),
            PcuHostArgument::read_write(graph::OUTPUT, &mut output),
        ]))
        .unwrap();
        assert_eq!(
            failed,
            PcuExecutionFault {
                kind: Kind::InvalidFloatingOperand,
                invocation_id: u64::try_from(narrow.len() - 1).unwrap(),
                recovered: false
            }
        );
        assert_eq!(
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            saved.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            fault(plan.call(&mut [
                PcuHostArgument::read(graph::INPUT, &narrow),
                PcuHostArgument::read_write(graph::OUTPUT, &mut output)
            ])),
            first
        );
        verify_widen(&backend, &widen, policy);
    }
    println!(
        "independent conversion encodings: {} narrow + {} widen per policy",
        narrow.len(),
        widen.len()
    );
}
#[test]
#[ignore = "requires actual Vulkan device"]
fn genuine_source_transactions_ties_signed_zero_and_retry() {
    let (backend, _) = device::selected();
    let mut call = source::narrow_prepare::<7, _>(&backend).unwrap();
    let input = [
        0.0_f64,
        -0.0,
        f64::from_bits(0x3ff0_0000_1000_0000),
        f64::from_bits(0x3ff0_0000_3000_0000),
        f64::from(f32::MAX),
        f64::from(f32::from_bits(1)),
        f64::from(f32::MIN_POSITIVE),
    ];
    let expected = [
        0.0_f32,
        -0.0,
        1.0,
        f32::from_bits(0x3f80_0002),
        f32::MAX,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
    ];
    let mut output = [99.0_f32; 9];
    call(&input, &mut output).unwrap();
    assert_eq!(
        output[..7].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.map(f32::to_bits)
    );
    let saved = output;
    let mut bad = input;
    bad[1] = f64::MAX;
    bad[6] = f64::NAN;
    assert_eq!(fault(call(&bad, &mut output)).unwrap().invocation_id, 1);
    assert_eq!(output.map(f32::to_bits), saved.map(f32::to_bits));
    assert!(call(&input[..6], &mut output).is_err());
    assert_eq!(output.map(f32::to_bits), saved.map(f32::to_bits));
    call(&input, &mut output).unwrap();
    let mut clamped = source::narrow_clamp_prepare::<7, _>(&backend).unwrap();
    let fatal = fault(clamped(&bad, &mut output)).unwrap();
    assert_eq!(
        (fatal.invocation_id, fatal.kind, fatal.recovered),
        (6, Kind::InvalidFloatingOperand, false)
    );
    assert_eq!(output.map(f32::to_bits), saved.map(f32::to_bits));
    bad[6] = 1.0;
    let recovered = fault(clamped(&bad, &mut output)).unwrap();
    assert_eq!(
        (recovered.invocation_id, recovered.kind, recovered.recovered),
        (1, Kind::ArithmeticOverflow, true)
    );
    assert_eq!(output[1].to_bits(), f32::MAX.to_bits());
    assert_eq!(output[6].to_bits(), 1.0_f32.to_bits());
    assert_eq!(
        output[7..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [99.0_f32.to_bits(); 2]
    );
    clamped(&input, &mut output).unwrap();
}

#[test]
fn independent_tininess_oracle_covers_normal_storage_after_tiny_precision_rounding() {
    use pcu_facade::{PcuClampedError, PcuClampedFloatConversion};
    for bits in [
        0x380f_ffff_dfff_ffff,
        0x380f_ffff_e000_0000,
        0x380f_ffff_e800_0000,
        0x380f_ffff_efff_ffff,
        0x380f_ffff_f000_0000,
        0x380f_ffff_f800_0000,
        0x3810_0000_0000_0000,
    ] {
        for sign in [0, 1_u64 << 63] {
            for policy in [
                Policy::IeeeAfterRounding,
                Policy::RejectSubnormalResult,
                Policy::AllowGradualUnderflow,
            ] {
                let (wanted, notice) = oracle::narrow(bits | sign, policy).unwrap();
                let actual = f64::from_bits(bits | sign).pcu_clamped_to_f32_with_policy(policy);
                match actual {
                    Ok(value) => {
                        assert_eq!(value.to_bits(), wanted);
                        assert!(notice.is_none());
                    }
                    Err(PcuClampedError::Range(error)) => {
                        assert_eq!(Some(error.kind()), notice);
                        assert_eq!(error.clamped_value().to_bits(), wanted);
                    }
                    Err(PcuClampedError::Fatal(kind)) => {
                        panic!("unexpected finite cast failure {kind:?}")
                    }
                }
            }
        }
    }
    // This value stores minnormal, yet its independently rounded unbounded precision
    // value remains tiny. IEEE after-rounding therefore still reports inexact underflow.
    assert_eq!(
        oracle::narrow(0x380f_ffff_e800_0000, Policy::IeeeAfterRounding).unwrap(),
        (0x0080_0000, Some(Kind::ArithmeticUnderflow))
    );
}
#[test]
#[ignore = "requires actual Vulkan device"]
fn near_normal_rounding_boundary_retains_tininess_fault_and_payload() {
    let (backend, _) = device::selected();
    for policy in [
        Policy::IeeeAfterRounding,
        Policy::RejectSubnormalResult,
        Policy::AllowGradualUnderflow,
    ] {
        let mut plan = prepare(
            &backend,
            Conversion::F64ToF32,
            1,
            PcuRangePolicy::Clamp,
            policy,
            false,
            false,
        );
        for bits in [
            0x380f_ffff_e000_0000,
            0x380f_ffff_e800_0000,
            0x380f_ffff_efff_ffff,
            0x380f_ffff_f000_0000,
            0x380f_ffff_f800_0000,
        ] {
            for sign in [0, 1_u64 << 63] {
                let (wanted, notice) = oracle::narrow(bits | sign, policy).unwrap();
                let mut output = [17.0_f32; 3];
                let observed = fault(plan.call(&mut [
                    PcuHostArgument::read(graph::INPUT, &[f64::from_bits(bits | sign)]),
                    PcuHostArgument::read_write(graph::OUTPUT, &mut output),
                ]));
                assert_eq!(
                    observed,
                    notice.map(|kind| PcuExecutionFault {
                        kind,
                        invocation_id: 0,
                        recovered: true
                    })
                );
                assert_eq!(
                    output.map(f32::to_bits),
                    [wanted, 17.0_f32.to_bits(), 17.0_f32.to_bits()]
                );
            }
        }
    }
}
