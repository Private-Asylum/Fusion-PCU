//! Checked helper composition retains overwritten range faults and host transactions.
use super::*;

macro_rules! family {
    ($module:ident, $ty:ident) => {
        mod $module {
            use super::*;

            #[pcu]
            #[allow(clippy::missing_const_for_fn)] // PCU scalar companions have a non-const lowering contract.
            fn helper(mut value: $ty, seed: $ty) -> $ty {
                let original: $ty = value;
                value += seed;
                value *= original;
                value - original
            }

            #[pcu]
            #[allow(unused_assignments, clippy::missing_const_for_fn)] // Discarding an arithmetic value must retain its fault.
            fn discarded_overflow(mut value: $ty, seed: $ty) -> $ty {
                let saved = value;
                value += seed;
                value = saved;
                value
            }

            #[pcu]
            #[allow(unused_assignments, clippy::missing_const_for_fn)] // Minimum-endpoint classification remains observable.
            fn discarded_underflow(mut value: $ty, seed: $ty) -> $ty {
                let saved = value;
                value -= seed;
                value = saved;
                value
            }

            #[pcu(invocations = N)]
            fn direct<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                output[id] = helper(input[id], *seed);
            }

            #[pcu(invocations = 3)]
            fn grid<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
                let mut id = pcu::context::global_invocation_id();
                let stride = pcu::context::invocation_count();
                while id < N {
                    output[id] = helper(input[id], *seed);
                    id += stride;
                }
            }

            #[pcu(invocations = N)]
            fn overflow<const N: usize>(input: &[$ty], seed: &$ty, stage: &mut [$ty], output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                stage[id] = input[id];
                output[id] = discarded_overflow(input[id], *seed);
            }

            #[pcu(invocations = N)]
            fn underflow<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                output[id] = discarded_underflow(input[id], *seed);
            }

            #[pcu(invocations = N, flag(clamp_range))]
            fn clamped<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                output[id] = discarded_overflow(input[id], *seed);
            }

            pub fn verify(recovered: bool) {
                const N: usize = 47;
                let seed = <$ty>::try_from(1_u8).unwrap();
                let sentinel = <$ty>::try_from(17_u8).unwrap();
                let mut output = [sentinel; N + 5];
                for phase in [0_u8, 17, 83] {
                    let values: [u8; N] = core::array::from_fn(|index| {
                        1 + (u8::try_from(index).unwrap() + phase) % 7
                    });
                    let input = values.map(|value| <$ty>::try_from(value).unwrap());
                    let expected = values.map(|value| <$ty>::try_from(value * value).unwrap());
                    direct::<N>(&input, &seed, &mut output).unwrap();
                    assert_eq!(output[..N], expected);
                    grid::<N>(&input, &seed, &mut output).unwrap();
                    assert_eq!(output[..N], expected);
                    assert_eq!(output[N..], [sentinel; 5]);
                    let before = output;
                    assert!(direct::<N>(&input, &seed, &mut output[..N - 1]).is_err());
                    assert_eq!(output, before);
                    direct::<N>(&input, &seed, &mut output).unwrap();
                }

                let mut stage = [sentinel; N + 3];
                let before_stage = stage;
                let before_output = output;
                let error = overflow::<N>(&[<$ty>::MAX; N], &seed, &mut stage, &mut output).unwrap_err();
                assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
                    if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow
                        && fault.invocation_id == 0 && fault.recovered == recovered));
                if recovered {
                    assert_eq!(stage[..N], [<$ty>::MAX; N]);
                    assert_eq!(output[..N], [<$ty>::MAX; N]);
                } else {
                    assert_eq!(stage, before_stage);
                    assert_eq!(output, before_output);
                }
                overflow::<N>(&[seed; N], &seed, &mut stage, &mut output).unwrap();
                assert_eq!(stage[..N], [seed; N]);
                assert_eq!(output[..N], [seed; N]);
                assert_eq!(stage[N..], [sentinel; 3]);
                assert_eq!(output[N..], [sentinel; 5]);

                let before_output = output;
                let error = underflow::<N>(&[<$ty>::MIN; N], &seed, &mut output).unwrap_err();
                assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
                    if fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow
                        && fault.invocation_id == 0 && fault.recovered == recovered));
                if recovered {
                    assert_eq!(output[..N], [<$ty>::MIN; N]);
                } else {
                    assert_eq!(output, before_output);
                }
                let error = clamped::<N>(&[<$ty>::MAX; N], &seed, &mut output).unwrap_err();
                assert!(matches!(error, PcuExecutionError::ArithmeticFault(fault)
                    if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.recovered));
                assert_eq!(output[..N], [<$ty>::MAX; N]);
                assert_eq!(output[N..], [sentinel; 5]);
            }
        }
    };
}
family!(unsigned8, u8);
family!(unsigned16, u16);
family!(unsigned32, u32);
family!(unsigned64, u64);
family!(unsigned128, u128);
family!(signed8, i8);
family!(signed16, i16);
family!(signed32, i32);
family!(signed64, i64);
family!(signed128, i128);

fn verify_backend(backend: PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            configure(PcuExecutionPolicy {
                backend,
                numerical_mode,
                range_policy,
                ..Default::default()
            })
            .unwrap();
            clear_thread_cache().unwrap();
            unsigned8::verify(range_policy == PcuRangePolicy::Clamp);
            unsigned16::verify(range_policy == PcuRangePolicy::Clamp);
            unsigned32::verify(range_policy == PcuRangePolicy::Clamp);
            unsigned64::verify(range_policy == PcuRangePolicy::Clamp);
            unsigned128::verify(range_policy == PcuRangePolicy::Clamp);
            signed8::verify(range_policy == PcuRangePolicy::Clamp);
            signed16::verify(range_policy == PcuRangePolicy::Clamp);
            signed32::verify(range_policy == PcuRangePolicy::Clamp);
            signed64::verify(range_policy == PcuRangePolicy::Clamp);
            signed128::verify(range_policy == PcuRangePolicy::Clamp);
        }
    }
    clear_thread_cache().unwrap();
    use_defaults().unwrap();
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ten_integer_helper_profiles_and_atomic_publication() {
    verify_backend(PcuBackendChoice::Cpu);
}

macro_rules! gate {
    ($feature:literal, $name:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native concrete integer helper SSA, fault and publication parity"]
        fn $name() {
            verify_backend(PcuBackendChoice::$backend);
        }
    };
}
gate!(
    "rocm",
    rocm_ten_integer_helper_profiles_and_atomic_publication,
    Rocm
);
gate!(
    "cuda",
    cuda_ten_integer_helper_profiles_and_atomic_publication,
    Cuda
);
gate!(
    "vulkan",
    vulkan_ten_integer_helper_profiles_and_atomic_publication,
    Vulkan
);
gate!(
    "metal",
    metal_ten_integer_helper_profiles_and_atomic_publication,
    Metal
);
gate!(
    "mlx",
    mlx_ten_integer_helper_profiles_and_atomic_publication,
    Mlx
);
