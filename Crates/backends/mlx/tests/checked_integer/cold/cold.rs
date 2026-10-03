//! Detached arithmetic admission and native actual-unique-input schemas.
#[rustfmt::skip]
use pcu_facade::{
    PcuScalar,
    PcuRangePolicy,
    PcuDispatchIntegerBinaryOp,
    PcuReproducibility,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuNumericalMode,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
use fusion_pcu_mlx::MlxCheckedIntegerPlan;
use super::graph;
fn type_profile<T: PcuScalar>() {
    for operation in [
        PcuDispatchIntegerBinaryOp::Add,
        PcuDispatchIntegerBinaryOp::Sub,
        PcuDispatchIntegerBinaryOp::Mul,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            for grid in [false, true] {
                for broadcast in [[false, false], [true, false], [false, true], [true, true]] {
                    graph::fixture_profile::<T, _>(
                        65,
                        operation,
                        range,
                        grid,
                        broadcast,
                        |kernel| {
                            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                                for compound in [
                                    PcuCompoundArithmeticPolicy::Checked,
                                    PcuCompoundArithmeticPolicy::BackendDefined,
                                ] {
                                    for precision in [
                                        PcuPrecisionPolicy::Preserve,
                                        PcuPrecisionPolicy::BackendOptimized,
                                    ] {
                                        let mut kernel = *kernel;
                                        kernel.numerical_requirements.numerical_mode = mode;
                                        kernel
                                            .numerical_requirements
                                            .numerical_options
                                            .compound_arithmetic = compound;
                                        kernel.numerical_requirements.numerical_options.precision =
                                            precision;
                                        assert_eq!(
                                            MlxCheckedIntegerPlan::assess(&kernel)
                                                .unwrap()
                                                .scalar_type(),
                                            T::TYPE
                                        );
                                        kernel
                                            .numerical_requirements
                                            .numerical_options
                                            .reproducibility = PcuReproducibility::PortableV1;
                                        assert!(MlxCheckedIntegerPlan::assess(&kernel).is_err());
                                    }
                                }
                            }
                        },
                    );
                }
            }
        }
    }
}
#[test]
fn fourteen_width_exact_header_profiles_without_runtime() {
    type_profile::<u8>();
    type_profile::<i8>();
    type_profile::<u16>();
    type_profile::<i16>();
    type_profile::<u32>();
    type_profile::<i32>();
    type_profile::<u64>();
    type_profile::<i64>();
    type_profile::<u128>();
    type_profile::<i128>();
    type_profile::<PcuU256>();
    type_profile::<PcuI256>();
    type_profile::<PcuU512>();
    type_profile::<PcuI512>();
}
