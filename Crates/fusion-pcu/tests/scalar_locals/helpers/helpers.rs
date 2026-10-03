//! Ordinary helper calls preserve local SSA and all caller-output transactions.
use super::*;

#[path = "integer/integer.rs"]
mod integer;
#[rustfmt::skip]
use fusion_pcu::{
    PcuNumericalMode,
    PcuRangePolicy,
};

macro_rules! family {
    ($module:ident, $ty:ident, $test:ident) => {
        mod $module {
            use super::*;

            #[pcu]
            #[allow(clippy::let_with_type_underscore)] // `_` annotations must retain saved SSA.
            fn transform(mut value: $ty, seed: $ty) -> $ty {
                let original: _ = value;
                value += seed;
                let transform = value * original;
                transform + original
            }

            #[pcu]
            #[allow(unused_assignments)] // Overwritten checked division must still report its fault.
            fn dead_division(mut value: $ty, denominator: $ty) -> $ty {
                let saved = value;
                value /= denominator;
                value = saved;
                value
            }

            #[pcu]
            #[allow(unused_assignments)] // Discarded checked overflow retains its observation.
            fn dead_overflow(value: $ty) -> $ty {
                let mut value: $ty = value;
                let saved = value;
                value += value;
                value = saved;
                value
            }

            #[pcu(invocations = N)]
            fn overflow<const N: usize>(input: &[$ty], output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                output[id] = dead_overflow(input[id]);
            }

            #[pcu(invocations = N, flag(clamp_range))]
            fn clamped<const N: usize>(input: &[$ty], output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                output[id] = dead_overflow(input[id]);
            }

            #[pcu(invocations = N)]
            fn direct<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
                let id = pcu::context::global_invocation_id();
                output[id] = transform(input[id], *seed);
            }

            #[pcu(invocations = 3)]
            fn grid<const N: usize>(input: &[$ty], seed: &$ty, output: &mut [$ty]) {
                let mut id = pcu::context::global_invocation_id();
                let stride = pcu::context::invocation_count();
                while id < N {
                    output[id] = transform(input[id], *seed);
                    id += stride;
                }
            }

            #[pcu(invocations = N)]
            fn fatal<const N: usize>(
                input: &[$ty], denominator: &[$ty], stage: &mut [$ty], output: &mut [$ty],
            ) {
                let id = pcu::context::global_invocation_id();
                stage[id] = input[id] + input[id];
                output[id] = dead_division(input[id], denominator[id]);
            }

            pub fn verify(backend: PcuBackendChoice) {
                const N: usize = 47;
                let _guard = POLICY_LOCK.lock().unwrap();
                for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                        configure(PcuExecutionPolicy {
                            backend, numerical_mode, range_policy,
                            ..Default::default()
                        }).unwrap();
                        clear_thread_cache().unwrap();
                        let sentinel = -17.0;
                        let mut output = [sentinel; N + 5];
                        for phase in [0_u16, 17, 83] {
                            let integers: [u16; N] = core::array::from_fn(|index| {
                                1 + u16::try_from(index).unwrap() + phase
                            });
                            let input = integers.map($ty::from);
                            let expected = integers.map(|value| $ty::from((value + 2) * value + value));
                            direct::<N>(&input, &2.0, &mut output).unwrap();
                            for (actual, expected) in output[..N].iter().zip(expected) {
                                assert_eq!(actual.to_bits(), expected.to_bits());
                            }
                            grid::<N>(&input, &2.0, &mut output).unwrap();
                            for (actual, expected) in output[..N].iter().zip(expected) {
                                assert_eq!(actual.to_bits(), expected.to_bits());
                            }
                            assert!(output[N..].iter().all(|value| value.to_bits() == sentinel.to_bits()));
                            let before = output.map($ty::to_bits);
                            assert!(direct::<N>(&input, &2.0, &mut output[..N - 1]).is_err());
                            assert_eq!(output.map($ty::to_bits), before);
                            direct::<N>(&input, &2.0, &mut output).unwrap();
                        }

                        let mut stage = [sentinel; N + 3];
                        let before_stage = stage.map($ty::to_bits);
                        let before_output = output.map($ty::to_bits);
                        let mut denominator = [1.0; N];
                        denominator[2] = 0.0;
                        let fault = fatal::<N>(&[2.0; N], &denominator, &mut stage, &mut output).unwrap_err();
                        assert!(matches!(fault, PcuExecutionError::ArithmeticFault(fault)
                            if fault.kind == PcuExecutionFaultKind::DivideByZero
                                && fault.invocation_id == 2 && !fault.recovered));
                        assert_eq!(stage.map($ty::to_bits), before_stage);
                        assert_eq!(output.map($ty::to_bits), before_output);
                        denominator[2] = 1.0;
                        fatal::<N>(&[2.0; N], &denominator, &mut stage, &mut output).unwrap();
                        assert!(stage[..N].iter().all(|value| value.to_bits() == <$ty>::from(4_u8).to_bits()));
                        assert!(output[..N].iter().all(|value| value.to_bits() == <$ty>::from(2_u8).to_bits()));

                        let before = output.map($ty::to_bits);
                        let fault = overflow::<N>(&[<$ty>::MAX; N], &mut output).unwrap_err();
                        assert!(matches!(fault, PcuExecutionError::ArithmeticFault(fault)
                            if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow
                                && fault.invocation_id == 0
                                && fault.recovered == (range_policy == PcuRangePolicy::Clamp)));
                        if range_policy == PcuRangePolicy::Reject {
                            assert_eq!(output.map($ty::to_bits), before);
                        } else {
                            assert!(output[..N].iter().all(|value| value.to_bits() == <$ty>::MAX.to_bits()));
                        }
                        let fault = clamped::<N>(&[<$ty>::MAX; N], &mut output).unwrap_err();
                        assert!(matches!(fault, PcuExecutionError::ArithmeticFault(fault)
                            if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow && fault.recovered));
                        assert!(output[..N].iter().all(|value| value.to_bits() == <$ty>::MAX.to_bits()));
                        assert!(output[N..].iter().all(|value| value.to_bits() == sentinel.to_bits()));
                        let before = output.map($ty::to_bits);
                        let mut input = [<$ty>::MAX; N];
                        input[1] = <$ty>::NAN;
                        let fault = clamped::<N>(&input, &mut output).unwrap_err();
                        assert!(matches!(fault, PcuExecutionError::ArithmeticFault(fault)
                            if fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand
                                && fault.invocation_id == 1 && !fault.recovered));
                        assert_eq!(output.map($ty::to_bits), before);
                    }
                }
                clear_thread_cache().unwrap();
                use_defaults().unwrap();
            }
            #[cfg(feature = "cpu")]
            #[test]
            fn $test() {
                verify(PcuBackendChoice::Cpu);
            }

        }
    };
}
family!(
    single,
    f32,
    cpu_f32_helper_locals_saved_values_and_overwritten_faults
);
family!(
    double,
    f64,
    cpu_f64_helper_locals_saved_values_and_overwritten_faults
);

macro_rules! provider_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native per-function helper locals, policies and atomic publication qualification"]
        fn $test() {
            single::verify(PcuBackendChoice::$backend);
            double::verify(PcuBackendChoice::$backend);
        }
    };
}
provider_gate!(
    "rocm",
    rocm_typed_helpers_compound_locals_and_atomic_publication,
    Rocm
);
provider_gate!(
    "cuda",
    cuda_typed_helpers_compound_locals_and_atomic_publication,
    Cuda
);
provider_gate!(
    "vulkan",
    vulkan_typed_helpers_compound_locals_and_atomic_publication,
    Vulkan
);
provider_gate!(
    "metal",
    metal_typed_helpers_compound_locals_and_atomic_publication,
    Metal
);
provider_gate!(
    "mlx",
    mlx_typed_helpers_compound_locals_and_atomic_publication,
    Mlx
);
