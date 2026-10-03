//! Independent exact encoding oracle through the real MLX C owner/sibling status boundary.
use super::*;
#[test]
fn status_scan_prioritizes_fatal_and_rejects_unknown_recovery_tags() {
    assert_eq!(
        select_fault(&[0x103, 4, 3]).unwrap(),
        Some(PcuExecutionFault {
            invocation_id: 1,
            kind: PcuExecutionFaultKind::InvalidFloatingOperand,
            recovered: false
        })
    );
    assert_eq!(
        select_fault(&[0, 0x103, 0x103]).unwrap(),
        Some(PcuExecutionFault {
            invocation_id: 1,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
            recovered: true
        })
    );
    for record in [1, 2, 5, 0x100, 0x101, 0x102, 0x104, u32::MAX] {
        assert!(select_fault(&[0x103, record]).is_err());
    }
}
#[cfg(target_os = "macos")]
#[path = "../../../../tests/checked_unary/oracle/oracle.rs"]
mod oracle;
fn diagnostics(map: &CheckedUnary, bytes: &[u8]) -> (Vec<u8>, Vec<u32>) {
    let api = &map.session.0.api;
    let mut input = Owner::empty(Rc::clone(api), api.array_free);
    let mut output = Owner::empty(Rc::clone(api), api.array_free);
    let mut records = Owner::empty(Rc::clone(api), api.array_free);
    // SAFETY: exact initialized byte span and live empty official owners; no caller storage
    // escapes. Tests retain all real owners until terminal MLX stream and both sibling waits.
    api.status(|| unsafe {
        (api.checked_upload)(
            &raw mut input.raw,
            bytes.as_ptr().cast(),
            map.input_carrier_count,
            map.upload_format,
        )
    })
    .unwrap();
    api.status(|| unsafe {
        (api.checked_apply)(
            &raw mut output.raw,
            &raw mut records.raw,
            map.owner.as_ref().unwrap().raw,
            input.raw,
        )
    })
    .unwrap();
    api.status(|| unsafe { (api.array_eval)(output.raw) })
        .unwrap();
    api.status(|| unsafe { (api.array_eval)(records.raw) })
        .unwrap();
    api.status(|| unsafe { (api.synchronize)(map.session.0.stream.raw) })
        .unwrap();
    api.status(|| unsafe { (api.array_wait)(output.raw) })
        .unwrap();
    api.status(|| unsafe { (api.array_wait)(records.raw) })
        .unwrap();
    validate(
        &map.session,
        output.raw,
        map.dtype,
        map.carrier_count,
        map.carrier_width,
    )
    .unwrap();
    validate(&map.session, records.raw, 3, map.count, 4).unwrap();
    available(&map.session, output.raw).unwrap();
    available(&map.session, records.raw).unwrap();
    // SAFETY: terminal dense exact dtype/count owners proved above; all accessors contain
    // native exceptions. Scratch results are diagnostics, never published as useful values.
    let values = api
        .guarded(|| unsafe {
            if map.dtype == 1 {
                (api.array_data_u8)(output.raw)
            } else if map.dtype == 2 {
                (api.array_data_u16)(output.raw).cast::<u8>()
            } else {
                (api.array_data_u32)(output.raw).cast::<u8>()
            }
        })
        .unwrap();
    let statuses = api
        .guarded(|| unsafe { (api.array_data_u32)(records.raw) })
        .unwrap();
    assert!(!values.is_null() && !statuses.is_null());
    // SAFETY: retained native owners hold exact initialized dense spans for both copies.
    unsafe {
        (
            std::slice::from_raw_parts(values, map.count * map.width).to_vec(),
            std::slice::from_raw_parts(statuses, map.count).to_vec(),
        )
    }
}
#[cfg(target_os = "macos")]
fn exhaustive<T: oracle::Low + fusion_pcu::PcuClampedFloat>() {
    use fusion_pcu::{PcuHostArgument, PcuBindingRef, PcuClampedError};
    let api = super::super::api::Api::load_default().unwrap();
    let session = api.open(0).unwrap();
    let count = usize::from(T::SIGN) * 2;
    let input: Vec<T> = (0..count)
        .map(|bits| T::from_bits(u16::try_from(bits).unwrap()))
        .collect();
    let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), &input);
    for op in [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let prepared =
                    CheckedUnary::prepare(&session, T::TYPE, op, policy, range, count, false)
                        .unwrap();
                let (values, statuses) = diagnostics(&prepared, bytes.bytes());
                for (index, value) in input.iter().copied().enumerate() {
                    let expected = oracle::evaluate::<T>(value.bits(), op, policy);
                    let core = match op {
                        PcuDispatchFloatUnaryOp::Neg => value.pcu_clamped_neg_with_policy(policy),
                        PcuDispatchFloatUnaryOp::Relu => value.pcu_clamped_relu_with_policy(policy),
                    };
                    let core = match core {
                        Ok(value) => Ok((value.bits(), false)),
                        Err(PcuClampedError::Range(fault)) => {
                            assert_eq!(fault.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
                            Ok((fault.clamped_value().bits(), true))
                        }
                        Err(PcuClampedError::Fatal(kind)) => Err(kind),
                    };
                    assert_eq!(
                        core,
                        expected,
                        "core {:?}/{op:?}/{policy:?}/{range:?}/{index:x}",
                        T::TYPE
                    );
                    let mut reference = [T::from_bits(17)];
                    let fault = oracle::native::<T, 1>(&[value], &mut reference, op, policy, range);
                    let (status, bits) = match fault {
                        Ok(()) => (0, reference[0].bits()),
                        Err(fault) => (
                            if fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand {
                                4
                            } else if fault.recovered {
                                0x103
                            } else {
                                3
                            },
                            if fault.recovered {
                                reference[0].bits()
                            } else {
                                0
                            },
                        ),
                    };
                    let actual = if prepared.width == 1 {
                        u16::from(values[index])
                    } else {
                        u16::from_le_bytes(values[index * 2..index * 2 + 2].try_into().unwrap())
                    };
                    assert_eq!(
                        (statuses[index], actual),
                        (status, bits),
                        "MLX {:?}/{op:?}/{policy:?}/{range:?}/{index:x}",
                        T::TYPE
                    );
                }
            }
        }
    }
}
#[cfg(target_os = "macos")]
macro_rules! proof {($name:ident,$ty:ty)=>{
    #[test]#[ignore="Requires real pinned MLX GPU custom kernel; every encoding/core/status is independently compared."]
    fn $name(){exhaustive::<$ty>();}
};}
#[cfg(target_os = "macos")]
proof!(f16_complete_mlx_unary_encodings, fusion_pcu::PcuF16Bits);
#[cfg(target_os = "macos")]
proof!(bf16_complete_mlx_unary_encodings, fusion_pcu::PcuBf16Bits);
#[cfg(target_os = "macos")]
proof!(
    e4m3fn_complete_mlx_unary_encodings,
    fusion_pcu::PcuF8E4M3FnBits
);
#[cfg(target_os = "macos")]
proof!(e5m2_complete_mlx_unary_encodings, fusion_pcu::PcuF8E5M2Bits);

#[path = "native_float/native_float.rs"]
mod native_float;
