//! Genuine fourteen-width joint source routes and independent byte-oracle publication checks.
extern crate pcu_facade as fusion_pcu;
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../../benches/checked_div_rem/ffi/ffi.rs"]
#[allow(dead_code)]
// Fixture uses exact native controls, not statistical timing or allocator sampling.
mod ffi;
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
#[path = "../../../cpu/tests/wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)]
// The independent byte domain/division is reused; text goldens have separate tests.
mod oracle;
#[path = "../../../cpu/tests/div_rem_roles/source/source.rs"]
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
                        backend: global::PcuBackendChoice::Vulkan,
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

#[test]
#[ignore = "requires physical Vulkan device; fourteen exact source-role profiles"]
fn fourteen_width_source_roles_faults_and_joint_publication() {
    macro_rules! widths {($($ty:ty),+)=>{$(verify::<$ty>();)+};}
    widths!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        pcu_facade::PcuI256,
        pcu_facade::PcuU256,
        pcu_facade::PcuI512,
        pcu_facade::PcuU512
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
#[test]
#[ignore = "requires physical Vulkan device; typed metadata and dual output rollback"]
fn original_declarations_and_exact_spans_are_validated_before_dual_publication() {
    #[rustfmt::skip]
    use pcu_facade::{
        PcuBindingRef as Binding,
        PcuHostArgument as Argument,
        PcuPreparedHostKernel,
    };
    let (backend, _) = device::selected();
    let bindings = source::unused_bindings::<i32>();
    let ir = source::unused_ir::<i32, 5>(&bindings).unwrap();
    let mut prepared = backend.prepare_host_kernel(&ir.ir()).unwrap();
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
    let mut prepared = backend.prepare_host_kernel(&ir.ir()).unwrap();
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

fn native_roles<T: Wide>() {
    let (_, identity) = device::selected();
    for kind in 0..6 {
        let mut native =
            ffi::NativeDivRem::new_roles(identity, 5, T::SIGNED, T::HOST_SIZE, kind).unwrap();
        let sentinel = oracle::small::<T>(99);
        let mut q = [sentinel; 8];
        let mut r = q;
        let mut statuses = [u32::MAX; 5];
        let mut left = [oracle::small::<T>(7); 5];
        left[0] = oracle::small(3);
        left[2] = oracle::maximum();
        let mut right = [oracle::small::<T>(3); 5];
        for phase in 0..4 {
            if phase == 1 {
                if kind == 2 || kind == 5 {
                    right[if kind == 5 { 0 } else { 2 }] = oracle::small(0);
                } else {
                    left[if kind == 3 { 0 } else { 2 }] = oracle::small(0);
                }
            }
            if phase == 2 {
                left.fill(oracle::small(7));
                right.fill(oracle::small(3));
            }
            if phase == 3 && T::SIGNED && kind >= 2 {
                if kind == 3 {
                    left[0] = T::from_bytes([255; 64]);
                    left[2] = oracle::minimum();
                } else if kind == 4 {
                    left[0] = oracle::minimum();
                    left[2] = T::from_bytes([255; 64]);
                } else {
                    left[if kind == 5 { 0 } else { 2 }] = oracle::minimum();
                    right[if kind == 5 { 0 } else { 2 }] = T::from_bytes([255; 64]);
                }
            }
            q.fill(sentinel);
            r.fill(sentinel);
            let mut first = None;
            native
                .diagnostics(
                    ffi::bytes(&left),
                    ffi::bytes(&right),
                    ffi::bytes_mut(&mut q),
                    ffi::bytes_mut(&mut r),
                    &mut statuses,
                )
                .unwrap();
            for lane in 0..5 {
                let (a, b) = operands(kind, &left, &right, lane);
                match oracle::evaluate(a, b) {
                    Ok(pair) => {
                        assert_eq!(statuses[lane], 0);
                        assert_eq!((q[lane], r[lane]), pair);
                    }
                    Err(fault) => {
                        assert_eq!(
                            statuses[lane],
                            if fault == PcuExecutionFaultKind::DivideByZero {
                                4
                            } else {
                                5
                            }
                        );
                        first.get_or_insert((lane as u64, fault));
                    }
                }
            }
            assert_eq!((&q[5..], &r[5..]), (&[sentinel; 3][..], &[sentinel; 3][..]));
            q.fill(sentinel);
            r.fill(sentinel);
            let fault = native
                .call(
                    ffi::bytes(&left),
                    ffi::bytes(&right),
                    ffi::bytes_mut(&mut q),
                    ffi::bytes_mut(&mut r),
                    None,
                )
                .unwrap();
            assert_eq!(fault.map(|f| (f.invocation_id, f.kind)), first);
            if first.is_some() {
                assert_eq!((q, r), ([sentinel; 8], [sentinel; 8]));
            }
        }
    }
}
#[test]
#[ignore = "requires physical Vulkan device and GLSL tools; independent compiler/byte oracle"]
fn fourteen_width_independent_native_role_status_and_rollback() {
    macro_rules! widths {($($ty:ty),+)=>{$(native_roles::<$ty>();)+};}
    widths!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        pcu_facade::PcuI256,
        pcu_facade::PcuU256,
        pcu_facade::PcuI512,
        pcu_facade::PcuU512
    );
}
#[test]
#[ignore = "requires physical Vulkan device; one physical input for repeated resources"]
fn cold_unique_input_storage_and_implementation_identity() {
    let (backend, _) = device::selected();
    let b = source::repeated_bindings::<i32>();
    let ir = source::repeated_ir::<i32, 5>(&b).unwrap();
    let plan = backend.prepare_host_kernel(&ir.ir()).unwrap();
    assert_eq!(plan.argument_count(), 3);
    assert_eq!(
        match plan.memory_realizations().unwrap() {
            fusion_pcu_vulkan::PcuVulkanPreparedMemoryRealizations::DivRem(m) => m.len(),
            _ => panic!("DivRem memory"),
        },
        4
    );
    let b = source::scalar_bindings::<i32>();
    let ir = source::scalar_ir::<i32, 5>(&b).unwrap();
    let plan = backend.prepare_host_kernel(&ir.ir()).unwrap();
    assert_eq!(plan.argument_count(), 4);
    assert_eq!(
        match plan.memory_realizations().unwrap() {
            fusion_pcu_vulkan::PcuVulkanPreparedMemoryRealizations::DivRem(m) => m.len(),
            _ => panic!("DivRem memory"),
        },
        5
    );
    {
        let range = pcu_facade::PcuRangePolicy::Clamp;
        let mut bad = ir.ir();
        bad.numerical_requirements.range_policy = range;
        assert!(backend.prepare_host_kernel(&bad).is_err());
    }
    let mut bad = ir.ir();
    bad.numerical_requirements.numerical_options.reproducibility =
        pcu_facade::PcuReproducibility::PortableV1;
    let portable = backend.prepare_host_kernel(&bad).unwrap();
    assert_eq!(portable.argument_count(), 4);
    let fusion_pcu_vulkan::PcuVulkanPreparedHost::DivRem(plan) = portable else {
        panic!("checked joint division plan");
    };
    assert_eq!(plan.profile().local_id(), Some(12548));
    assert!(plan.profile().portable);
}
