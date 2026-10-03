//! Exact integer input-role, fault, publication and unread-owner qualification.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/portable_integer_operands/native/native.rs"]
#[allow(dead_code)] // The benchmark additionally exercises full-host native calls.
mod native;
#[path = "../../benches/portable_integer_operands/oracle/oracle.rs"]
#[allow(dead_code)] // Benchmark finite-bank generator is not part of every fixture.
mod oracle;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[path = "../../benches/portable_integer_operands/source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuDispatchKernelIr,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuRangePolicy,
    PcuImplementationRequirements,
    PcuFloatUnderflowPolicy,
    PcuTensor,
};
use fusion_pcu_rocm::RocmOwnedDispatchBackend;
use oracle::Format;
fn equal<T: Format>(actual: &[T], want: &[T]) {
    assert_eq!(actual.len(), want.len());
    for (a, b) in actual.iter().zip(want) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn host<T: Format, const N: usize>(
    kind: u32,
    a: &[T],
    b: &[T],
    out: &mut [T],
) -> Result<(), PcuExecutionError> {
    match kind {
        0 => source::repeated_add::<T, N>(out, a),
        1 => source::unused_mul::<T, N>(&[], out, a),
        2 => source::reordered_sub::<T, N>(b, out, a),
        3 => source::indexed_zero_sub::<T, N>(out, a),
        4 => source::grid_zero_sub::<T, N>(&[], out, a),
        _ => unreachable!(),
    }
}
fn outcome(result: Result<(), PcuExecutionError>, lane: u64, code: u32, clamp: bool) {
    let error = result.unwrap_err();
    let fault = error
        .arithmetic_fault()
        .unwrap_or_else(|| panic!("{error:?}"));
    assert_eq!(fault.invocation_id, lane);
    assert_eq!(
        fault.kind,
        if code == 3 {
            PcuExecutionFaultKind::ArithmeticOverflow
        } else {
            PcuExecutionFaultKind::ArithmeticUnderflow
        }
    );
    assert_eq!(fault.recovered, clamp);
}
fn ir<T: Format>(
    kind: u32,
    requirements: PcuImplementationRequirements,
    mut f: impl FnMut(&PcuDispatchKernelIr<'_>),
) {
    macro_rules! build {
        ($bindings:ident,$builder:ident) => {{
            let bindings = source::$bindings::<T>();
            let builder = source::$builder::<T, 65>(
                &bindings,
                requirements.float_underflow,
                requirements.range_policy,
                requirements,
            )
            .unwrap();
            builder.with_ir(|ir| f(ir));
        }};
    }
    match kind {
        0 => build!(
            repeated_add_bindings,
            __repeated_add_ir_with_float_underflow_policy
        ),
        1 => build!(
            unused_mul_bindings,
            __unused_mul_ir_with_float_underflow_policy
        ),
        2 => build!(
            reordered_sub_bindings,
            __reordered_sub_ir_with_float_underflow_policy
        ),
        3 => build!(
            indexed_zero_sub_bindings,
            __indexed_zero_sub_ir_with_float_underflow_policy
        ),
        4 => build!(
            grid_zero_sub_bindings,
            __grid_zero_sub_ir_with_float_underflow_policy
        ),
        _ => unreachable!(),
    }
}
fn prepared<T: Format>(
    kernel: &mut impl PcuPreparedHostKernel<Error = fusion_pcu_rocm::RocmHostKernelError>,
    kind: u32,
    a: &[T],
    b: &[T],
    out: &mut [T],
) -> Result<(), PcuExecutionError> {
    let empty: &[T] = &[];
    let result = match kind {
        0 | 3 => kernel.call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), out),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), a),
        ]),
        1 | 4 => kernel.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), empty),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), out),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), a),
        ]),
        2 => kernel.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), b),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), out),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), a),
        ]),
        _ => unreachable!(),
    };
    result.map_err(|error| match error {
        fusion_pcu_rocm::RocmHostKernelError::CheckedExecutionFault(fault) => {
            PcuExecutionError::ArithmeticFault(fault)
        }
        other => panic!("{other:?}"),
    })
}
fn configure(requirements: PcuImplementationRequirements) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        numerical_mode: requirements.numerical_mode,
        numerical_options: requirements.numerical_options,
        float_underflow: requirements.float_underflow,
        range_policy: requirements.range_policy,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
}
#[allow(clippy::too_many_lines)] // Each exact host/native/fault/retry cohort shares one cold executable.
fn device<T: Format>(
    backend: &RocmOwnedDispatchBackend,
    requirements: PcuImplementationRequirements,
) {
    configure(requirements);
    let clamp = requirements.range_policy == PcuRangePolicy::Clamp;
    for kind in 0..5 {
        ir::<T>(kind, requirements, |ir| {
            assert_eq!(ir.numerical_requirements, requirements);
            let mut kernel = backend.prepare_host_kernel(ir).unwrap();
            let mut native = native::Native::new::<T, 65>(backend, ir, 1);
            for phase in 0..2 {
                let (a, b, want) = oracle::inputs::<T>(65, phase, kind);
                let mut out = vec![T::sentinel(); 67];
                host::<T, 65>(kind, &a, &b, &mut out).unwrap();
                oracle::verify(&want, &out);
                prepared(&mut kernel, kind, &a, &b, &mut out).unwrap();
                oracle::verify(&want, &out);
                if kind == 2 {
                    native.host(&b, &a, &mut out);
                } else {
                    native.host(&a, &b, &mut out);
                }
                oracle::verify(&want, &out);
            }
            let op = if kind == 0 {
                0
            } else if kind == 1 {
                2
            } else {
                1
            };
            let rows = oracle::rows::<T>(op)
                .into_iter()
                .filter(|(a, b, _, code)| {
                    *code != 0 && (kind >= 2 || a.encode_le().as_ref() == b.encode_le().as_ref())
                })
                .collect::<Vec<_>>();
            assert!(!rows.is_empty());
            for (a, b, want, code) in rows {
                let zero = T::small(0);
                let mut left = vec![zero; 65];
                let mut right = vec![zero; 65];
                let mut expected = vec![zero; 65];
                match kind {
                    0 | 1 => {
                        left[2] = a;
                        left[6] = a;
                    }
                    2 => {
                        left[2] = a;
                        left[6] = a;
                        right[2] = b;
                        right[6] = b;
                    }
                    3 => {
                        left.fill(b);
                        left[2] = a;
                        left[6] = a;
                    }
                    4 => {
                        left.fill(a);
                        left[2] = b;
                        left[6] = b;
                    }
                    _ => unreachable!(),
                }
                expected[2] = want;
                expected[6] = want;
                let mut out = vec![T::sentinel(); 67];
                let prior = out.clone();
                outcome(host::<T, 65>(kind, &left, &right, &mut out), 2, code, clamp);
                if clamp {
                    oracle::verify(&expected, &out);
                } else {
                    equal(&out, &prior);
                }
                outcome(
                    prepared(&mut kernel, kind, &left, &right, &mut out),
                    2,
                    code,
                    clamp,
                );
                if clamp {
                    oracle::verify(&expected, &out);
                } else {
                    equal(&out, &prior);
                }
                if kind == 2 {
                    native.upload(0, &right, &left);
                } else {
                    native.upload(0, &left, &right);
                }
                let word = (2u64 << 3) | u64::from(code) | if clamp { 1u64 << 63 } else { 0 };
                assert_eq!(native.submit(0), word);
                if clamp {
                    native.read(&mut out);
                    oracle::verify(&expected, &out);
                }
                let (left, right, want) = oracle::inputs::<T>(65, 1, kind);
                host::<T, 65>(kind, &left, &right, &mut out).unwrap();
                oracle::verify(&want, &out);
                prepared(&mut kernel, kind, &left, &right, &mut out).unwrap();
                oracle::verify(&want, &out);
            }
        });
    }
}
fn cold<T: Format>() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for uf in [
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    PcuFloatUnderflowPolicy::RejectSubnormalResult,
                ] {
                    for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        let requirements = PcuImplementationRequirements {
                            numerical_mode: mode,
                            numerical_options: PcuNumericalOptions {
                                compound_arithmetic: compound,
                                precision,
                                reproducibility: PcuReproducibility::PortableV1,
                            },
                            float_underflow: uf,
                            range_policy: range,
                        };
                        for kind in 0..5 {
                            ir::<T>(kind, requirements, |ir| {
                                let source =
                                    fusion_pcu_rocm::lower_dispatch_to_hip_source(ir).unwrap();
                                assert!(source.contains("PCU numerical requirements"));
                                if kind == 1 || kind == 4 {
                                    assert!(!source.contains("b_0_0"));
                                }
                                assert_eq!(ir.numerical_requirements, requirements);
                                let descriptor =
                                    fusion_pcu::describe_portable_v1_integer_map(ir).unwrap();
                                assert_eq!(descriptor.scalar, T::TYPE);
                                assert_eq!(descriptor.range_policy, range);
                                let mut ordinary = *ir;
                                ordinary
                                    .numerical_requirements
                                    .numerical_options
                                    .reproducibility = PcuReproducibility::Unspecified;
                                let ordinary =
                                    fusion_pcu_rocm::lower_dispatch_to_hip_source(&ordinary)
                                        .unwrap();
                                assert_eq!(
                                    source.split_once('\n').unwrap().1,
                                    ordinary.split_once('\n').unwrap().1
                                );
                            });
                        }
                    }
                }
            }
        }
    }
}
macro_rules! all{($f:ident$(,$arg:expr)*)=>{$f::<i8>($($arg),*);$f::<u8>($($arg),*);$f::<i16>($($arg),*);$f::<u16>($($arg),*);$f::<i32>($($arg),*);$f::<u32>($($arg),*);$f::<i64>($($arg),*);$f::<u64>($($arg),*);$f::<i128>($($arg),*);$f::<u128>($($arg),*);$f::<fusion_pcu::PcuI256>($($arg),*);$f::<fusion_pcu::PcuU256>($($arg),*);$f::<fusion_pcu::PcuI512>($($arg),*);$f::<fusion_pcu::PcuU512>($($arg),*);};}
#[test]
fn portable_fourteen_width_cold_requested_tuples_and_same_checked_body() {
    all!(cold);
}
#[test]
#[ignore = "actual GPU source/prepared/native bits and fault/publication qualification; serial window"]
fn portable_fourteen_width_repeated_reordered_indexed_operands() {
    let (_, backend, _) = selection::selected_device();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let requirements = PcuImplementationRequirements {
                numerical_mode: mode,
                range_policy: range,
                numerical_options: PcuNumericalOptions {
                    reproducibility: PcuReproducibility::PortableV1,
                    ..PcuNumericalOptions::default()
                },
                ..PcuImplementationRequirements::DEFAULT
            };
            all!(device, &backend, requirements);
        }
    }
}
fn owner<T: Format>() {
    let requirements = PcuImplementationRequirements::DEFAULT;
    configure(requirements);
    let bank = [T::small(2); 65];
    let want = [T::small(4); 65];
    let left = source::identity(&bank).unwrap();
    let mut output = source::identity(&[T::sentinel(); 67]).unwrap();
    source::unused_mul::<T, 65>(&[], &mut output, &left).unwrap();
    let mut values = vec![T::sentinel(); 67];
    output.read_into(&mut values).unwrap();
    oracle::verify(&want, &values);
    let (a, _, saturated, code) = oracle::rows::<T>(2)
        .into_iter()
        .find(|(a, b, _, code)| *code != 0 && a.encode_le().as_ref() == b.encode_le().as_ref())
        .unwrap();
    let mut bad = bank;
    bad[2] = a;
    bad[6] = a;
    let bad = source::identity(&bad).unwrap();
    outcome(
        source::unused_mul::<T, 65>(&[], &mut output, &bad),
        2,
        code,
        false,
    );
    let prior = values.clone();
    assert!(output.read_into(&mut values).is_err());
    equal(&values, &prior);
    let mut fresh = source::identity(&[T::sentinel(); 67]).unwrap();
    source::unused_mul::<T, 65>(&output, &mut fresh, &left).unwrap();
    fresh.read_into(&mut values).unwrap();
    oracle::verify(&want, &values);
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        device: None,
        ..Default::default()
    })
    .unwrap();
    let foreign = source::identity(&bank).unwrap();
    configure(requirements);
    source::unused_mul::<T, 65>(&foreign, &mut fresh, &left).unwrap();
    fresh.read_into(&mut values).unwrap();
    oracle::verify(&want, &values);
    let mut unchanged = [T::small(0); 65];
    foreign.read_into(&mut unchanged).unwrap();
    equal(&unchanged, &bank);
    left.read_into(&mut unchanged).unwrap();
    equal(&unchanged, &bank);
    // A genuine owned operation borrows the selected old-session input after cache clear.
    // Host-only creation can lawfully select another session; affinity must preserve this one.
    let mut recovered = source::owned_sub::<T>(&bad).unwrap();
    recovered.read_into(&mut values).unwrap();
    oracle::verify(&[T::small(0); 65], &values);
    let mut clamp = PcuImplementationRequirements::DEFAULT;
    clamp.range_policy = PcuRangePolicy::Clamp;
    configure(clamp);
    outcome(
        source::unused_mul::<T, 65>(&foreign, &mut recovered, &bad),
        2,
        code,
        true,
    );
    recovered.read_into(&mut values).unwrap();
    let mut expected = want;
    expected[2] = saturated;
    expected[6] = saturated;
    oracle::verify(&expected, &values);
    source::unused_mul::<T, 65>(&output, &mut recovered, &left).unwrap();
    recovered.read_into(&mut values).unwrap();
    oracle::verify(&want, &values);
    let _: &PcuTensor<T> = &fresh;
}
#[test]
#[ignore = "actual GPU macro skips unread discarded/foreign owners before carrier/affinity/lease conversion"]
fn portable_fourteen_width_unread_foreign_and_discarded_owner() {
    all!(owner);
}

fn broadcast<T: Format>(backend: &RocmOwnedDispatchBackend) {
    configure(PcuImplementationRequirements::DEFAULT);
    let left = [T::small(3)];
    let want = [T::small(6); 65];
    let mut output = vec![T::sentinel(); 67];
    source::zero_broadcast_add::<T, 65>(&mut output, &left).unwrap();
    oracle::verify(&want, &output);
    let bindings = source::zero_broadcast_add_bindings::<T>();
    let builder = source::zero_broadcast_add_ir::<T, 65>(&bindings).unwrap();
    builder.with_ir(|ir| {
        let descriptor = fusion_pcu::describe_portable_v1_integer_map(ir).unwrap();
        assert_eq!(descriptor.operands.input_element_counts(65), [1, 0]);
        let mut kernel = backend.prepare_host_kernel(ir).unwrap();
        kernel
            .call(&mut [
                PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut output),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &left),
            ])
            .unwrap();
        oracle::verify(&want, &output);
        let mut native = native::Native::new::<T, 65>(backend, ir, 1);
        native.host(&left, &[], &mut output);
        oracle::verify(&want, &output);
        let prior = output.clone();
        assert!(source::zero_broadcast_add::<T, 65>(&mut output, &[]).is_err());
        equal(&output, &prior);
    });
}
#[test]
#[ignore = "actual one-element portable broadcast source/prepared/native physical boundary"]
fn portable_fourteen_width_single_element_broadcast() {
    let (_, backend, _) = selection::selected_device();
    all!(broadcast, &backend);
}

fn negatives<T: Format + fusion_pcu::PcuCheckedIntegerDivision>() {
    let bindings = source::unsupported_transport_bindings::<T>();
    let builder = source::unsupported_transport_ir::<T, 65>(&bindings).unwrap();
    builder.with_ir(|ir| {
        assert!(fusion_pcu::describe_portable_v1_integer_map(ir).is_err());
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(ir).is_err());
    });
    let bindings = source::requested_div_rem_bindings::<T>();
    let builder = source::requested_div_rem_ir::<T, 65>(&bindings).unwrap();
    builder.with_ir(|ir| {
        assert!(fusion_pcu::describe_portable_v1_integer_map(ir).is_err());
        assert!(fusion_pcu::describe_portable_v1_integer_div_rem_map(ir).is_ok());
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(ir).is_ok());
        let mut clamp = *ir;
        clamp.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&clamp).is_err());
    });
    let mut requirements = PcuImplementationRequirements::DEFAULT;
    requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    ir::<T>(0, requirements, |ir| {
        let mut bad = *ir;
        bad.entry.logical_shape = [0, 1, 1];
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&bad).is_err());
        let mut bad = *ir;
        bad.numerical_requirements.range_policy = PcuRangePolicy::Clamp;
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&bad).is_err());
        let mut extra = ir.ops.to_vec();
        extra.insert(3, ir.ops[2]);
        let bad = PcuDispatchKernelIr { ops: &extra, ..*ir };
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&bad).is_err());
    });
}
#[test]
fn portable_fourteen_width_joint_eligibility_and_malformed_refusals() {
    all!(negatives);
}
#[test]
#[ignore = "actual ordinary Portable joint source plus transport/owned arithmetic refusal, no fallback"]
fn portable_source_unsupported_profiles_preserve_outputs() {
    configure(PcuImplementationRequirements::DEFAULT);
    let input = [3i32; 65];
    let mut quotient = [17i32; 67];
    let mut remainder = quotient;
    let result = source::unsupported_transport::<i32, 65>(&input, &mut quotient);
    assert!(
        matches!(result, Err(PcuExecutionError::UnsupportedNumericalOptions(options)) if options.reproducibility == PcuReproducibility::PortableV1)
    );
    assert_eq!(quotient, [17; 67]);
    assert_eq!(remainder, [17; 67]);
    source::requested_div_rem::<i32, 65>(&input, &input, &mut quotient, &mut remainder).unwrap();
    assert_eq!(&quotient[..65], &[1; 65]);
    assert_eq!(&remainder[..65], &[0; 65]);
    assert_eq!(&quotient[65..], &[17; 2]);
    assert_eq!(&remainder[65..], &[17; 2]);
    assert!(source::unsupported_owned::<i32>(&input).is_err());
    // The scoped identity has no new arithmetic node and keeps the input producer's policy.
    let preserved = source::scoped_identity::<i32>(&input).unwrap();
    let mut stack = [17i32; 67];
    preserved.read_into(&mut stack).unwrap();
    oracle::verify(&input, &stack);
}

#[path = "golden/golden.rs"]
mod golden;

#[path = "../../benches/checked_integer_operands/source/source.rs"]
#[allow(dead_code)] // Ordinary source is imported to prove global Portable policy inheritance.
mod ordinary;
fn inherited<T: Format>() {
    let mut requirements = PcuImplementationRequirements::DEFAULT;
    requirements.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    configure(requirements);
    let input = [T::small(3); 65];
    let expected = [T::small(6); 65];
    let mut output = vec![T::sentinel(); 67];
    ordinary::repeated_add::<T, 65>(&mut output, &input).unwrap();
    oracle::verify(&expected, &output);
}
#[test]
#[ignore = "actual global requested Portable inherited by an ordinary unflagged scalar source"]
fn portable_global_inheritance_preserves_exact_bits_and_tails() {
    all!(inherited);
}

fn owned_selected<T: Format>() {
    configure(PcuImplementationRequirements::DEFAULT);
    let mut input = [T::small(3); 65];
    let first = source::owned_repeated::<T>(&[], &input).unwrap();
    let mut stack = vec![T::sentinel(); 67];
    first.read_into(&mut stack).unwrap();
    oracle::verify(&[T::small(9); 65], &stack);
    for phase in 0..3 {
        input[2] = T::small(if phase % 2 == 0 { 4 } else { 3 });
        let next = source::owned_repeated::<T>(&[], &input).unwrap();
        next.read_into(&mut stack).unwrap();
        let mut expected = [T::small(9); 65];
        expected[2] = T::small(if phase % 2 == 0 { 16 } else { 9 });
        oracle::verify(&expected, &stack);
        first.read_into(&mut stack).unwrap();
        oracle::verify(&[T::small(9); 65], &stack);
    }
    let resident = source::identity(&input).unwrap();
    let next = source::owned_repeated::<T>(&[], &resident).unwrap();
    next.read_into(&mut stack).unwrap();
    let mut expected = [T::small(9); 65];
    expected[2] = T::small(16);
    oracle::verify(&expected, &stack);
    for result in [
        source::owned_repeated::<T>(&input, &[]),
        source::owned_discarded::<T>(&[], &input),
    ] {
        assert!(
            matches!(result, Err(PcuExecutionError::EmptyTensorInput)),
            "{result:?}"
        );
    }
    first.read_into(&mut stack).unwrap();
    oracle::verify(&[T::small(9); 65], &stack);
}
#[test]
#[ignore = "actual normal owned selected-input empty declaration cold/warm source and checked-effect negatives"]
fn normal_owned_selected_inputs_preserve_empty_unread_and_effect_law() {
    all!(owned_selected);
}
