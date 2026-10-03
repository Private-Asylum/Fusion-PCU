//! Prepared and ordinary source proof, whole-map rollback, original schemas and exact offers.
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PCU_CPU_DIV_REM_IMPLEMENTATION_REVISION,
    PcuCpuHostBackend,
    PcuCpuHostError,
    PcuCpuHostOffers,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    global::PcuBackendChoice,
    global::PcuExecutionPolicy,
    PcuExecutionFaultKind,
    PcuBindingRef,
    PcuCostBoundary,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuDeviceIdentity,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuExecutorId,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationCost,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuObjectKind,
    PcuObjectRef,
    PcuPreparedHostKernel,
    PcuProviderId,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[path = "source/source.rs"]
mod source;

#[test]
fn exhaustive_eight_bit_native_oracle_checks_every_pair() {
    macro_rules! width {
        ($module:ident, $ty:ty) => {{
            let mut call =
                source::$module::direct_prepare::<1, _>(&PcuCpuHostBackend::scalar()).unwrap();
            for left in 0..=u8::MAX {
                for right in 0..=u8::MAX {
                    let lhs = <$ty>::from_le_bytes([left]);
                    let rhs = <$ty>::from_le_bytes([right]);
                    let mut q = [17 as $ty; 2];
                    let mut r = q;
                    match (lhs.checked_div(rhs), lhs.checked_rem(rhs)) {
                        (Some(quotient), Some(remainder)) => {
                            call(&[lhs], &[rhs], &mut q, &mut r).unwrap();
                            assert_eq!((q, r), ([quotient, 17], [remainder, 17]));
                        }
                        _ => {
                            let fault = call(&[lhs], &[rhs], &mut q, &mut r)
                                .unwrap_err()
                                .fault()
                                .unwrap();
                            assert_eq!(fault.invocation_id, 0);
                            assert_eq!(
                                fault.kind,
                                if rhs == 0 {
                                    PcuExecutionFaultKind::DivideByZero
                                } else {
                                    PcuExecutionFaultKind::SignedDivisionOverflow
                                }
                            );
                            assert_eq!((q, r), ([17; 2], [17; 2]));
                        }
                    }
                }
            }
        }};
    }
    width!(i8_source, i8);
    width!(u8_source, u8);
}

#[test]
#[allow(clippy::cognitive_complexity)] // Complete format/policy/publication matrix preserves one owner lifecycle and exact case ordering.
fn all_widths_source_direct_grid_first_fault_and_retry() {
    global::configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Cpu,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
    macro_rules! width {
        ($module:ident, $ty:ty) => {{
            let mut direct = source::$module::direct_prepare::<17, _>(&PcuCpuHostBackend::scalar()).unwrap();
            let mut grid = source::$module::grid_prepare::<17, _>(&PcuCpuHostBackend::scalar()).unwrap();
            let mut state = 0x12ab_9876_eeff_0415_u64;
            for _ in 0..128 {
                let mut lhs = [0 as $ty; 17];
                let mut rhs = [0 as $ty; 17];
                for index in 0..17 {
                    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                    lhs[index] = <$ty>::from_le_bytes(state.to_le_bytes()[..core::mem::size_of::<$ty>()].try_into().unwrap());
                    state = state.rotate_left(19);
                    rhs[index] = <$ty>::from_le_bytes(state.to_le_bytes()[..core::mem::size_of::<$ty>()].try_into().unwrap());
                }
                let mut q = [<$ty>::MAX; 19]; let mut r = q;
                let mut expected_q = q; let mut expected_r = r;
                let mut fault = None;
                for index in 0..17 {
                    match (lhs[index].checked_div(rhs[index]), lhs[index].checked_rem(rhs[index])) {
                        (Some(quotient), Some(remainder)) => { expected_q[index] = quotient; expected_r[index] = remainder; }
                        _ => { fault = Some((index, if rhs[index] == 0 { PcuExecutionFaultKind::DivideByZero }
                            else { PcuExecutionFaultKind::SignedDivisionOverflow })); break; }
                    }
                }
                let result = direct(&lhs, &rhs, &mut q, &mut r);
                let mut gq = [<$ty>::MAX; 19]; let mut gr = gq;
                assert_eq!(grid(&lhs, &rhs, &mut gq, &mut gr), result);
                assert_eq!((gq, gr), (q, r));
                if let Some((index, kind)) = fault {
                    assert_eq!(result.unwrap_err().fault().map(|f| (f.invocation_id, f.kind)), Some((index as u64, kind)));
                    assert_eq!((q, r), ([<$ty>::MAX; 19], [<$ty>::MAX; 19]));
                } else { result.unwrap(); assert_eq!((q, r), (expected_q, expected_r)); }
            }
            let lhs = [7 as $ty; 17]; let mut rhs = [3 as $ty; 17];
            let mut q = [99 as $ty; 19]; let mut r = q;
            rhs[5] = 0; rhs[11] = 0;
            let fault = direct(&lhs, &rhs, &mut q, &mut r).unwrap_err().fault().unwrap();
            assert_eq!((fault.invocation_id, fault.kind), (5, PcuExecutionFaultKind::DivideByZero));
            assert_eq!((q, r), ([99 as $ty; 19], [99 as $ty; 19]));
            rhs.fill(3);
            direct(&lhs, &rhs, &mut q, &mut r).unwrap();
            assert_eq!(q[..17], [2 as $ty; 17]); assert_eq!(r[..17], [1 as $ty; 17]);
            assert_eq!((q[17..].to_vec(), r[17..].to_vec()), (vec![99 as $ty; 2], vec![99 as $ty; 2]));
            source::$module::direct::<17>(&lhs, &rhs, &mut q, &mut r).unwrap();
            source::$module::grid::<17>(&lhs, &rhs, &mut q, &mut r).unwrap();
            source::$module::strict::<17>(&lhs, &rhs, &mut q, &mut r).unwrap();
            let before = (q, r);
            rhs[5] = 0;
            assert!(matches!(source::$module::direct::<17>(&lhs, &rhs, &mut q, &mut r),
                Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.invocation_id == 5 && fault.kind == PcuExecutionFaultKind::DivideByZero));
            assert_eq!((q, r), before);
            rhs[5] = 3;
            source::$module::direct::<17>(&lhs, &rhs, &mut q, &mut r).unwrap();
        }};
    }
    width!(i8_source, i8);
    width!(u8_source, u8);
    width!(i16_source, i16);
    width!(u16_source, u16);
    width!(i32_source, i32);
    width!(u32_source, u32);
    width!(i64_source, i64);
    width!(u64_source, u64);
}

#[test]
#[allow(clippy::cognitive_complexity)] // Complete format/policy/publication matrix preserves one owner lifecycle and exact case ordering.
fn signed_min_minus_one_rejects_both_outputs_and_short_remainder_rolls_back() {
    macro_rules! signed {
        ($module:ident, $ty:ty) => {{
            let mut call =
                source::$module::direct_prepare::<3, _>(&PcuCpuHostBackend::scalar()).unwrap();
            let mut q = [17 as $ty; 4];
            let mut r = q;
            let fault = call(&[7, <$ty>::MIN, 9], &[3, -1, 2], &mut q, &mut r)
                .unwrap_err()
                .fault()
                .unwrap();
            assert_eq!(
                (fault.invocation_id, fault.kind),
                (1, PcuExecutionFaultKind::SignedDivisionOverflow)
            );
            assert_eq!((q, r), ([17; 4], [17; 4]));
            assert!(matches!(
                call(&[7, 8, 9], &[3, 2, 2], &mut q, &mut r[..2]),
                Err(PcuCpuHostError::Arguments(_))
            ));
            assert_eq!((q, r), ([17; 4], [17; 4]));
            call(&[-7, 7, -7], &[3, -3, -3], &mut q, &mut r).unwrap();
            assert_eq!(q, [-2, -2, 2, 17]);
            assert_eq!(r, [-1, 1, -1, 17]);
        }};
    }
    signed!(i8_source, i8);
    signed!(i16_source, i16);
    signed!(i32_source, i32);
    signed!(i64_source, i64);
}

#[test]
#[allow(clippy::cognitive_complexity)] // Complete format/policy/publication matrix preserves one owner lifecycle and exact case ordering.
fn exact_cold_offers_keep_numerical_permissions_independent() {
    let identity = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), identity, PcuExecutorId(0));
    macro_rules! width {
        ($module:ident, $id:expr) => {{
            let bindings = source::$module::direct_bindings();
            let builder = source::$module::direct_ir::<3>(&bindings).unwrap();
            let kernel = builder.ir();
            let mut request = PcuImplementationRequest {
                device: identity,
                executor: PcuExecutorId(0),
                operation: &kernel,
                requirements: PcuImplementationRequirements::default(),
                boundary: PcuCostBoundary::Host,
            };
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                request.requirements.numerical_mode = mode;
                let kernel = pcu_facade::PcuDispatchKernelIr {
                    numerical_requirements: request.requirements,
                    ..*request.operation
                };
                let mut request = pcu_facade::PcuImplementationRequest {
                    operation: &kernel,
                    ..request
                };
                let mut output = [None];
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(1));
                let offer = output[0].unwrap();
                assert_eq!(
                    (offer.implementation.local_id, offer.implementation.revision),
                    ($id, PCU_CPU_DIV_REM_IMPLEMENTATION_REVISION)
                );
                assert_eq!(offer.requirements, request.requirements);
                assert_eq!(offer.workspace_bytes, Some(0));
                assert_eq!(
                    offer.cost,
                    PcuImplementationCost::unknown(PcuCostBoundary::Host)
                );
                request.requirements.numerical_options.reproducibility =
                    PcuReproducibility::PortableV1;
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                request.requirements.numerical_options = PcuNumericalOptions::default();
                request.requirements.numerical_options.compound_arithmetic =
                    PcuCompoundArithmeticPolicy::BackendDefined;
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                request.requirements.numerical_options = PcuNumericalOptions::default();
                request.requirements.numerical_options.precision =
                    PcuPrecisionPolicy::BackendOptimized;
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                request.requirements.numerical_options = PcuNumericalOptions::default();
                request.requirements.range_policy = PcuRangePolicy::Clamp;
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
                request.requirements.range_policy = PcuRangePolicy::Reject;
                request.boundary = PcuCostBoundary::Resident;
                assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
            }
            let mut body = kernel.ops.to_vec();
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { flags, .. }) =
                &mut body[2]
            {
                *flags = pcu_facade::model::PcuIntegerDivFlags::DIV_OR_ZERO;
            }
            let changed = PcuDispatchKernelIr {
                ops: &body,
                ..kernel
            };
            request.operation = &changed;
            assert_eq!(offers.implementation_offers(&request, &mut [None]), Ok(0));
        }};
    }
    width!(i8_source, 128);
    width!(u8_source, 129);
    width!(i16_source, 130);
    width!(u16_source, 131);
    width!(i32_source, 132);
    width!(u32_source, 133);
    width!(i64_source, 134);
    width!(u64_source, 135);
}

#[test]
fn detached_ssa_mapping_argument_order_and_unused_input_schema() {
    let bindings = source::i32_source::direct_bindings();
    let builder = source::i32_source::direct_ir::<3>(&bindings).unwrap();
    let kernel = builder.ir();
    for operands in [[1, 2], [2, 1], [1, 1], [2, 2]] {
        let mut body = kernel.ops.to_vec();
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { lhs, rhs, .. }) = &mut body[2]
        {
            *lhs = pcu_facade::PcuDispatchValueId(operands[0]);
            *rhs = pcu_facade::PcuDispatchValueId(operands[1]);
        }
        let changed = PcuDispatchKernelIr {
            ops: &body,
            ..kernel
        };
        let mut prepared = PcuCpuHostBackend::scalar()
            .prepare_host_kernel(&changed)
            .unwrap();
        drop(body);
        assert_eq!(prepared.argument_count(), 4);
        let a = [7_i32, -7, 11];
        let b = [3_i32, 3, -2];
        let inputs = [&a, &b];
        let mut q = [99; 5];
        let mut r = q;
        prepared
            .call(&mut [
                PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut r),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &b),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut q),
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &a),
            ])
            .unwrap();
        for index in 0..3 {
            let lhs = inputs[usize::from(operands[0] - 1)][index];
            let rhs = inputs[usize::from(operands[1] - 1)][index];
            assert_eq!((q[index], r[index]), (lhs / rhs, lhs % rhs));
        }
        let before = (q, r);
        assert!(
            prepared
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &a),
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), &b[..2]),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut q),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut r),
                ])
                .is_err()
        );
        assert_eq!((q, r), before);
    }
}

#[test]
fn broader_profiles_and_total_division_reject_cold() {
    let bindings = source::i32_source::direct_bindings();
    let builder = source::i32_source::direct_ir::<3>(&bindings).unwrap();
    let kernel = builder.ir();
    // Only sealed integer division profiles admit; float and Bool carriers stay closed.
    for scalar in [
        PcuScalarType::F32,
        PcuScalarType::F64,
        PcuScalarType::F128,
        PcuScalarType::F256,
        PcuScalarType::BF16,
        PcuScalarType::Bool,
    ] {
        let mut body = kernel.ops.to_vec();
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { value_type, .. }) =
            &mut body[2]
        {
            *value_type = PcuValueType::Scalar(scalar);
        }
        let mut bindings = kernel.bindings.to_vec();
        for binding in &mut bindings {
            binding.binding_type = pcu_facade::PcuBindingType::Value(PcuValueType::Scalar(scalar));
        }
        let changed = PcuDispatchKernelIr {
            bindings: &bindings,
            ops: &body,
            type_caps: PcuValueTypeCaps::for_scalar(scalar),
            ..kernel
        };
        assert!(matches!(
            PcuCpuHostBackend::scalar().prepare_host_kernel(&changed),
            Err(PcuCpuHostError::UnsupportedProfile)
        ));
    }
    let mut flags_body = kernel.ops.to_vec();
    if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { flags, .. }) = &mut flags_body[2]
    {
        *flags = pcu_facade::model::PcuIntegerDivFlags::DIV_OR_ZERO;
    }
    assert!(
        PcuCpuHostBackend::scalar()
            .prepare_host_kernel(&PcuDispatchKernelIr {
                ops: &flags_body,
                ..kernel
            })
            .is_err()
    );
    let mut body = kernel.ops.to_vec();
    if let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { index, .. }) = &mut body[0] {
        *index = PcuDispatchIndex::BindingElementZero;
    }
    let projected = PcuCpuHostBackend::scalar()
        .prepare_host_kernel(&PcuDispatchKernelIr {
            ops: &body,
            ..kernel
        })
        .unwrap();
    let fusion_pcu_cpu::PcuCpuPreparedHost::DivRem(plan) = projected else {
        panic!("DivRem route");
    };
    assert_eq!((plan.local_id(), plan.implementation_revision()), (8196, 1));
}

#[test]
fn all_argument_permutations_and_complete_schema_reject_before_publication() {
    let bindings = source::i32_source::direct_bindings();
    let builder = source::i32_source::direct_ir::<3>(&bindings).unwrap();
    let mut prepared = PcuCpuHostBackend::scalar()
        .prepare_host_kernel(&builder.ir())
        .unwrap();
    let a = [7_i32, -7, 11];
    let b = [3_i32, 3, -2];
    for first in 0..4 {
        for second in 0..4 {
            for third in 0..4 {
                for fourth in 0..4 {
                    let order = [first, second, third, fourth];
                    if (0..4).any(|position| order[..position].contains(&order[position])) {
                        continue;
                    }
                    let mut q = [99; 5];
                    let mut r = q;
                    let mut arguments = [
                        Some(PcuHostArgument::read(PcuBindingRef::new(0, 0), &a)),
                        Some(PcuHostArgument::read(PcuBindingRef::new(0, 1), &b)),
                        Some(PcuHostArgument::read_write(
                            PcuBindingRef::new(0, 2),
                            &mut q,
                        )),
                        Some(PcuHostArgument::read_write(
                            PcuBindingRef::new(0, 3),
                            &mut r,
                        )),
                    ];
                    prepared
                        .call(&mut order.map(|position| arguments[position].take().unwrap()))
                        .unwrap();
                    assert_eq!(q, [2, -2, -5, 99, 99]);
                    assert_eq!(r, [1, -1, 1, 99, 99]);
                }
            }
        }
    }
    let mut q = [99; 5];
    let mut r = q;
    let first = PcuBindingRef::new(0, 0);
    let second = PcuBindingRef::new(0, 1);
    let quotient = PcuBindingRef::new(0, 2);
    let remainder = PcuBindingRef::new(0, 3);
    assert!(
        matches!(prepared.call(&mut [PcuHostArgument::read(first, &a),
        PcuHostArgument::read(first, &b), PcuHostArgument::read_write(quotient, &mut q),
        PcuHostArgument::read_write(remainder, &mut r)]),
        Err(PcuCpuHostError::Arguments(fusion_pcu_cpu::PcuCpuHostArgumentError::DuplicateBinding(binding))) if binding == first)
    );
    assert!(
        matches!(prepared.call(&mut [PcuHostArgument::read(first, &a),
        PcuHostArgument::read(second, &[1.0_f32; 3]), PcuHostArgument::read_write(quotient, &mut q),
        PcuHostArgument::read_write(remainder, &mut r)]),
        Err(PcuCpuHostError::Arguments(fusion_pcu_cpu::PcuCpuHostArgumentError::TypeMismatch { binding, .. })) if binding == second)
    );
    assert!(
        matches!(prepared.call(&mut [PcuHostArgument::read(first, &a),
        PcuHostArgument::read(second, &b), PcuHostArgument::read_write(quotient, &mut q),
        PcuHostArgument::read(remainder, &r)]),
        Err(PcuCpuHostError::Arguments(fusion_pcu_cpu::PcuCpuHostArgumentError::AccessMismatch { binding, .. })) if binding == remainder)
    );
    assert_eq!((q, r), ([99; 5], [99; 5]));
}
