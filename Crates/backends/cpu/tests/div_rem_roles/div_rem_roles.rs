//! Independent exact-bit source and detached execution proof for new joint operand roles.
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuHostBackend,
    PcuCpuPreparedHost,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuCheckedIntegerDivision,
    PcuCompoundArithmeticPolicy,
    PcuExecutionFaultKind,
    PcuHostKernelBackend,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuFloatUnderflowPolicy,
};
#[path = "../wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)]
// The independent byte domain/division is reused; text goldens have separate tests.
mod oracle;
#[path = "source/source.rs"]
mod source;
use oracle::Wide;

fn invoke<T: PcuCheckedIntegerDivision>(
    kind: usize,
    left: &[T],
    right: &[T],
    q: &mut [T],
    r: &mut [T],
) -> Result<(), global::PcuExecutionError> {
    match kind {
        0 => source::repeated::<T, 5>(q, r, left),
        1 => source::unused::<T, 5>(&[], left, r, q),
        2 => source::reordered::<T, 5>(right, r, left, q),
        3 => source::mixed::<T, 5>(q, left, r),
        4 => source::grid::<T, 5>(r, &[], q, left),
        _ => source::scalar::<T, 5>(&left[0], q, &right[0], r),
    }
}
const fn operands<T: Copy>(kind: usize, left: &[T], right: &[T], lane: usize) -> (T, T) {
    match kind {
        0 | 1 => (left[lane], left[lane]),
        2 => (left[lane], right[lane]),
        3 => (left[lane], left[0]),
        4 => (left[0], left[lane]),
        _ => (left[0], right[0]),
    }
}
fn verify_call<T: Wide + PcuCheckedIntegerDivision>(kind: usize, left: &[T], right: &[T]) {
    let sentinel = oracle::small::<T>(99);
    let mut q = [sentinel; 8];
    let mut r = q;
    let mut expected_q = q;
    let mut expected_r = r;
    let mut fault = None;
    for lane in 0..5 {
        let (a, b) = operands(kind, left, right, lane);
        match oracle::evaluate(a, b) {
            Ok((quotient, remainder)) => {
                expected_q[lane] = quotient;
                expected_r[lane] = remainder;
            }
            Err(error) => {
                fault = Some((lane as u64, error));
                break;
            }
        }
    }
    let result = invoke(kind, left, right, &mut q, &mut r);
    if let Some((lane, kind)) = fault {
        let actual = result.unwrap_err().arithmetic_fault().unwrap();
        assert_eq!(
            (actual.invocation_id, actual.kind, actual.recovered),
            (lane, kind, false)
        );
        assert_eq!((q, r), ([sentinel; 8], [sentinel; 8]));
    } else {
        result.unwrap();
        assert_eq!((q, r), (expected_q, expected_r));
    }
}
fn verify<T: Wide + PcuCheckedIntegerDivision>() {
    let three = oracle::small(3);
    let seven = oracle::small(7);
    let zero = oracle::small(0);
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for underflow in [
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                ] {
                    global::configure(global::PcuExecutionPolicy {
                        backend: global::PcuBackendChoice::Cpu,
                        numerical_mode: mode,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            ..Default::default()
                        },
                        float_underflow: underflow,
                        ..Default::default()
                    })
                    .unwrap();
                    for kind in 0..6 {
                        let mut left = [seven; 5];
                        left[0] = three;
                        left[2] = oracle::maximum::<T>();
                        let mut right = [three; 5];
                        for _ in 0..3 {
                            left[3] = if left[3] == seven {
                                oracle::maximum::<T>()
                            } else {
                                seven
                            };
                            verify_call(kind, &left, &right);
                        }
                        if kind == 2 || kind == 5 {
                            right[if kind == 5 { 0 } else { 2 }] = zero;
                        } else {
                            left[if kind == 3 { 0 } else { 2 }] = zero;
                        }
                        verify_call(kind, &left, &right);
                        right.fill(three);
                        left.fill(seven);
                        verify_call(kind, &left, &right);
                        let sentinel = oracle::small::<T>(99);
                        let mut q = [sentinel; 8];
                        let mut r = q;
                        assert!(invoke(kind, &left, &right, &mut q, &mut r[..4]).is_err());
                        assert_eq!((q, r), ([sentinel; 8], [sentinel; 8]));
                        invoke(kind, &left, &right, &mut q, &mut r).unwrap();
                        if T::SIGNED && (2..=5).contains(&kind) {
                            let minus_one = T::from_bytes([255; 64]);
                            if kind == 3 {
                                left[0] = minus_one;
                                left[2] = oracle::minimum();
                            } else if kind == 4 {
                                left[0] = oracle::minimum();
                                left[2] = minus_one;
                            } else {
                                left[if kind == 5 { 0 } else { 2 }] = oracle::minimum();
                                right[if kind == 5 { 0 } else { 2 }] = minus_one;
                            }
                            verify_call(kind, &left, &right);
                        }
                    }
                }
            }
        }
    }
}
macro_rules! case {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            verify::<$ty>();
        }
    };
}
case!(i8_roles, i8);
case!(u8_roles, u8);
case!(i16_roles, i16);
case!(u16_roles, u16);
case!(i32_roles, i32);
case!(u32_roles, u32);
case!(i64_roles, i64);
case!(u64_roles, u64);
case!(i128_roles, i128);
case!(u128_roles, u128);
case!(i256_roles, pcu_facade::PcuI256);
case!(u256_roles, pcu_facade::PcuU256);
case!(i512_roles, pcu_facade::PcuI512);
case!(u512_roles, pcu_facade::PcuU512);

#[test]
fn separate_cold_identity_for_new_roles() {
    let bindings = source::repeated_bindings::<i32>();
    let ir = source::repeated_ir::<i32, 5>(&bindings).unwrap();
    let prepared = PcuCpuHostBackend::scalar()
        .prepare_host_kernel(&ir.ir())
        .unwrap();
    let PcuCpuPreparedHost::DivRem(plan) = prepared else {
        panic!("joint route");
    };
    assert_eq!(
        (
            plan.local_id(),
            plan.implementation_revision(),
            plan.argument_count()
        ),
        (8196, 1, 3)
    );
    let policy = pcu_facade::PcuDispatchKernelIr {
        numerical_requirements: pcu_facade::PcuImplementationRequirements {
            range_policy: pcu_facade::PcuRangePolicy::Clamp,
            ..Default::default()
        },
        ..ir.ir()
    };
    assert!(
        PcuCpuHostBackend::scalar()
            .prepare_host_kernel(&policy)
            .is_err()
    );
    assert_ne!(
        PcuExecutionFaultKind::DivideByZero,
        PcuExecutionFaultKind::SignedDivisionOverflow
    );
}

#[test]
fn original_declarations_and_exact_spans_are_validated_before_dual_publication() {
    #[rustfmt::skip]
    use pcu_facade::{
        PcuBindingRef as Binding,
        PcuHostArgument as Argument,
        PcuPreparedHostKernel,
    };
    let bindings = source::unused_bindings::<i32>();
    let ir = source::unused_ir::<i32, 5>(&bindings).unwrap();
    let mut prepared = PcuCpuHostBackend::scalar()
        .prepare_host_kernel(&ir.ir())
        .unwrap();
    let left = [7_i32; 5];
    let mut q = [99_i32; 8];
    let mut r = q;
    assert!(
        prepared
            .call(&mut [
                Argument::read(Binding::new(0, 0), &[] as &[u8]),
                Argument::read(Binding::new(0, 1), &left),
                Argument::read_write(Binding::new(0, 2), &mut r),
                Argument::read_write(Binding::new(0, 3), &mut q),
            ])
            .is_err()
    );
    assert_eq!((q, r), ([99; 8], [99; 8]));
    prepared
        .call(&mut [
            Argument::read_write(Binding::new(0, 3), &mut q),
            Argument::read(Binding::new(0, 0), &[] as &[i32]),
            Argument::read_write(Binding::new(0, 2), &mut r),
            Argument::read(Binding::new(0, 1), &left),
        ])
        .unwrap();
    assert_eq!(
        (q, r),
        ([1, 1, 1, 1, 1, 99, 99, 99], [0, 0, 0, 0, 0, 99, 99, 99])
    );
    let before = (q, r);
    assert!(
        prepared
            .call(&mut [
                Argument::read(Binding::new(0, 0), &[] as &[i32]),
                Argument::read(Binding::new(0, 1), &left[..4]),
                Argument::read_write(Binding::new(0, 2), &mut r),
                Argument::read_write(Binding::new(0, 3), &mut q),
            ])
            .is_err()
    );
    assert_eq!((q, r), before);
    let bindings = source::scalar_bindings::<i32>();
    let ir = source::scalar_ir::<i32, 5>(&bindings).unwrap();
    let mut prepared = PcuCpuHostBackend::scalar()
        .prepare_host_kernel(&ir.ir())
        .unwrap();
    prepared
        .call(&mut [
            Argument::read(Binding::new(0, 0), &[7_i32]),
            Argument::read_write(Binding::new(0, 1), &mut q),
            Argument::read(Binding::new(0, 2), &[3_i32]),
            Argument::read_write(Binding::new(0, 3), &mut r),
        ])
        .unwrap();
    assert_eq!(
        (q, r),
        ([2, 2, 2, 2, 2, 99, 99, 99], [1, 1, 1, 1, 1, 99, 99, 99])
    );
}

fn exact_offer<T: PcuCheckedIntegerDivision>(id: u32) {
    #[rustfmt::skip]
    use pcu_facade::{
        PcuImplementationOffers,
        PcuImplementationRequest,
    };
    let device = pcu_facade::PcuDeviceIdentity::from_device_ref(pcu_facade::PcuObjectRef {
        provider: pcu_facade::PcuProviderId(3),
        generation: 7,
        kind: pcu_facade::PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = fusion_pcu_cpu::PcuCpuHostOffers::new(
        PcuCpuHostBackend::scalar(),
        device,
        pcu_facade::PcuExecutorId(0),
    );
    let bindings = source::repeated_bindings::<T>();
    let ir = source::repeated_ir::<T, 5>(&bindings).unwrap();
    let kernel = ir.ir();
    let mut request = PcuImplementationRequest {
        device,
        executor: pcu_facade::PcuExecutorId(0),
        operation: &kernel,
        requirements: kernel.numerical_requirements,
        boundary: pcu_facade::PcuCostBoundary::Host,
    };
    let mut output = [None];
    assert_eq!(offers.implementation_offers(&request, &mut output), Ok(1));
    let implementation = output[0].unwrap().implementation;
    assert_eq!((implementation.local_id, implementation.revision), (id, 1));
    assert_eq!(offers.implementation_offers(&request, &mut []), Ok(1));
    request.requirements.numerical_options.reproducibility =
        pcu_facade::PcuReproducibility::PortableV1;
    assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
    request.requirements = kernel.numerical_requirements;
    request.requirements.range_policy = pcu_facade::PcuRangePolicy::Clamp;
    assert_eq!(offers.implementation_offers(&request, &mut output), Ok(0));
}
#[test]
fn fourteen_disjoint_cold_role_ids_and_exact_tuple_offers() {
    exact_offer::<i8>(8192);
    exact_offer::<u8>(8193);
    exact_offer::<i16>(8194);
    exact_offer::<u16>(8195);
    exact_offer::<i32>(8196);
    exact_offer::<u32>(8197);
    exact_offer::<i64>(8198);
    exact_offer::<u64>(8199);
    exact_offer::<i128>(8200);
    exact_offer::<u128>(8201);
    exact_offer::<pcu_facade::PcuI256>(8202);
    exact_offer::<pcu_facade::PcuU256>(8203);
    exact_offer::<pcu_facade::PcuI512>(8204);
    exact_offer::<pcu_facade::PcuU512>(8205);
}
