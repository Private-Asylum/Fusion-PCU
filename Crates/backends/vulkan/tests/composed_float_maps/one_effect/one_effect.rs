//! An overwritten scalar result still leaves its checked operation observable.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    validate_composed_float_map,
    validate_one_effect_float_map,
    PcuSpirvComposedFloatProfile,
    PcuSpirvError,
};

#[pcu(crate_path=::pcu_facade,invocations=N)]
fn overwritten_division<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    divisor: &[T],
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    let mut value = input[id];
    value /= divisor[id];
    value = input[id];
    output[id] = value;
}

#[pcu(crate_path=::pcu_facade,invocations=N)]
fn overwritten_add<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    let mut value = input[id];
    value += value;
    value = input[id];
    output[id] = value;
}

struct Cold;
fn assert_policy(
    ir: &PcuDispatchKernelIr<'_>,
    requirements: pcu_facade::PcuImplementationRequirements,
) {
    assert_eq!(ir.numerical_requirements, requirements);
    let mut arithmetic = 0;
    for op in ir.ops {
        if let pcu_facade::PcuDispatchOp::Data(
            pcu_facade::PcuDispatchDataOp::CheckedFloatBinary {
                range_policy,
                underflow_policy,
                ..
            },
        ) = op
        {
            assert_eq!(*range_policy, requirements.range_policy);
            assert_eq!(*underflow_policy, requirements.float_underflow);
            arithmetic += 1;
        }
    }
    assert_eq!(arithmetic, 1);
}
struct Captured(PcuSpirvComposedFloatProfile);
impl PcuHostKernelBackend for Cold {
    type Prepared = Captured;
    type Error = PcuSpirvError;
    fn prepare_host_kernel(&self, ir: &PcuDispatchKernelIr<'_>) -> Result<Captured, Self::Error> {
        assert!(validate_composed_float_map(ir).is_err());
        validate_one_effect_float_map(ir).map(|profile| {
            assert!(profile.is_one_effect());
            Captured(profile)
        })
    }
}
impl PcuPreparedHostKernel for Captured {
    type Error = PcuSpirvError;
    fn call(&mut self, _: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        assert!((2..=3).contains(&self.0.resources().len()));
        Ok(())
    }
}

#[test]
fn checked_effect_survives_overwrite_in_genuine_source_cold_capture() {
    macro_rules! width {
        ($ty:ty) => {
            assert!(overwritten_add_prepare::<$ty, 7, _>(&Cold).is_ok());
            assert!(overwritten_division_prepare::<$ty, 7, _>(&Cold).is_ok());
        };
    }
    width!(pcu_facade::PcuF16Bits);
    width!(pcu_facade::PcuBf16Bits);
    width!(pcu_facade::PcuF8E4M3FnBits);
    width!(pcu_facade::PcuF8E5M2Bits);
    width!(f32);
    width!(f64);
}

#[test]
fn policy_aware_genuine_source_freezes_complete_header_and_instruction_policy() {
    use pcu_facade::{PcuImplementationRequirements, PcuNumericalMode, PcuRangePolicy};
    for compound in [
        pcu_facade::PcuCompoundArithmeticPolicy::Checked,
        pcu_facade::PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            pcu_facade::PcuPrecisionPolicy::Preserve,
            pcu_facade::PcuPrecisionPolicy::BackendOptimized,
        ] {
            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    for uf in super::UF {
                        let requirements = PcuImplementationRequirements {
                            numerical_mode: mode,
                            range_policy: range,
                            float_underflow: uf,
                            ..Default::default()
                        };
                        let mut requirements = requirements;
                        requirements.numerical_options.compound_arithmetic = compound;
                        requirements.numerical_options.precision = precision;
                        macro_rules! width {
                            ($ty:ty) => {{
                                let bindings = overwritten_add_bindings::<$ty>();
                                __overwritten_add_ir_with_float_underflow_policy::<$ty, 7>(
                                    &bindings,
                                    uf,
                                    range,
                                    requirements,
                                )
                                .unwrap()
                                .with_ir(|ir| assert_policy(ir, requirements));
                                let bindings = overwritten_division_bindings::<$ty>();
                                __overwritten_division_ir_with_float_underflow_policy::<$ty, 7>(
                                    &bindings,
                                    uf,
                                    range,
                                    requirements,
                                )
                                .unwrap()
                                .with_ir(|ir| assert_policy(ir, requirements));
                            }};
                        }
                        width!(pcu_facade::PcuF16Bits);
                        width!(pcu_facade::PcuBf16Bits);
                        width!(pcu_facade::PcuF8E4M3FnBits);
                        width!(pcu_facade::PcuF8E5M2Bits);
                        width!(f32);
                        width!(f64);
                    }
                }
            }
        }
    }
}

#[path = "native/native.rs"]
mod native;
