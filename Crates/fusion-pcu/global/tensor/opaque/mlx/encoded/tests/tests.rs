//! Cold storage-family admission must distinguish a leaf from arithmetic.
use super::assess;
use crate::global::PcuExecutionPolicy;
use crate::global::__pcu_capture_tensor_program;
use crate::global::PcuSourceShape;
use crate::PcuScalar;

#[crate::pcu(crate_path = crate)]
fn retain<T: PcuScalar>(input: &[T]) -> Result<crate::PcuTensor<T>, crate::PcuExecutionError> {
    pcu::identity(input)
}

#[crate::pcu(crate_path = crate, flag(non_strict), flag(non_deterministic), flag(native_compound), flag(backend_precision))]
fn product(
    left: &[[f32; 2]; 2],
    right: &[[f32; 2]; 2],
) -> Result<crate::PcuTensor<f32>, crate::PcuExecutionError> {
    pcu::matmul(left, right)
}

fn leaf<T: PcuScalar>() {
    let options = PcuExecutionPolicy::default();
    let built = __pcu_capture_tensor_program::<T, 1, _>(
        [PcuSourceShape::Slice { length: 7 }],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        retain::__pcu_capture_entry::<T>,
    )
    .unwrap();
    assert!(assess(&built, options).is_ok());
}

#[test]
fn all_byte_aligned_leaf_types_use_exact_encoded_admission() {
    leaf::<u8>();
    leaf::<i8>();
    leaf::<u16>();
    leaf::<i16>();
    leaf::<u32>();
    leaf::<i32>();
    leaf::<u64>();
    leaf::<i64>();
    leaf::<u128>();
    leaf::<i128>();
    leaf::<crate::PcuU256>();
    leaf::<crate::PcuI256>();
    leaf::<crate::PcuU512>();
    leaf::<crate::PcuI512>();
    leaf::<crate::PcuF16Bits>();
    leaf::<crate::PcuBf16Bits>();
    leaf::<crate::PcuF8E4M3FnBits>();
    leaf::<crate::PcuF8E5M2Bits>();
    leaf::<f32>();
    leaf::<f64>();
    leaf::<crate::PcuF128Bits>();
    leaf::<crate::PcuF256Bits>();
}

#[test]
fn native_matmul_is_not_misclassified_as_encoded_transport() {
    let options = PcuExecutionPolicy::default();
    let built = __pcu_capture_tensor_program::<f32, 2, _>(
        [PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 2,
        }; 2],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        product::__pcu_capture_entry,
    )
    .unwrap();
    assert!(assess(&built, options).is_err());
    assert!(fusion_pcu_mlx::MlxMatmulPlan::assess_program(built.program()).is_ok());
}

#[crate::pcu(crate_path = crate, flag(allow_gradual_underflow))]
fn reversed<T: crate::PcuCheckedFloat>(
    _unused: &[T],
    left: &[T],
    right: &[T],
) -> Result<crate::PcuTensor<T>, crate::PcuExecutionError> {
    pcu::sub(right, left)
}

#[crate::pcu(crate_path = crate)]
fn repeated<T: crate::PcuCheckedFloat>(
    _unused: &[T],
    input: &[T],
) -> Result<crate::PcuTensor<T>, crate::PcuExecutionError> {
    pcu::mul(input, input)
}

#[crate::pcu(crate_path = crate)]
fn unused_effect(
    input: &[f64],
    right: &[f64],
) -> Result<crate::PcuTensor<f64>, crate::PcuExecutionError> {
    let _checked = pcu::div(input, right);
    pcu::identity(input)
}

fn binary<T: crate::PcuCheckedFloat>() {
    let options = PcuExecutionPolicy {
        float_underflow: crate::PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ..Default::default()
    };
    let built = __pcu_capture_tensor_program::<T, 3, _>(
        [
            PcuSourceShape::Slice { length: 99 },
            PcuSourceShape::Slice { length: 7 },
            PcuSourceShape::Slice { length: 7 },
        ],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        reversed::__pcu_capture_entry::<T>,
    )
    .unwrap();
    let requirements = assess(&built, options).unwrap();
    assert_eq!(
        requirements.float_underflow,
        crate::PcuFloatUnderflowPolicy::AllowGradualUnderflow
    );
    assert_eq!(built.input_indices, [1, 2]);
    let plan =
        fusion_pcu_mlx::MlxCheckedTensorBinaryPlan::assess_program(&built.program, requirements)
            .unwrap();
    assert_eq!(plan.operand_inputs(), [1, 0]);
    assert_eq!(plan.input_values().len(), 2);
    assert_eq!(plan.element_count(), 7);
    assert_eq!(plan.scalar_type(), T::TYPE);

    let repeated = __pcu_capture_tensor_program::<T, 2, _>(
        [
            PcuSourceShape::Slice { length: 99 },
            PcuSourceShape::Slice { length: 7 },
        ],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        repeated::__pcu_capture_entry::<T>,
    )
    .unwrap();
    assert_eq!(repeated.input_indices, [1]);
    let requirements = assess(&repeated, options).unwrap();
    let plan =
        fusion_pcu_mlx::MlxCheckedTensorBinaryPlan::assess_program(&repeated.program, requirements)
            .unwrap();
    assert_eq!(plan.operand_inputs(), [0, 0]);
    assert_eq!(plan.input_values().len(), 1);
}

#[test]
fn six_format_binary_keeps_actual_unique_inputs_and_local_underflow() {
    binary::<crate::PcuF16Bits>();
    binary::<crate::PcuBf16Bits>();
    binary::<crate::PcuF8E4M3FnBits>();
    binary::<crate::PcuF8E5M2Bits>();
    binary::<f32>();
    binary::<f64>();
}

#[test]
fn unused_binary_fault_remains_observable_before_identity_publication() {
    let mut options = PcuExecutionPolicy::default();
    let built = __pcu_capture_tensor_program::<f64, 2, _>(
        [PcuSourceShape::Slice { length: 7 }; 2],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        unused_effect::__pcu_capture_entry,
    )
    .unwrap();
    let requirements = assess(&built, options).unwrap();
    let plan =
        fusion_pcu_mlx::MlxCheckedTensorBinaryPlan::assess_program(&built.program, requirements)
            .unwrap();
    assert_eq!(plan.output(), plan.input_values()[0]);
    assert_ne!(plan.output(), plan.effect());
    options.range_policy = crate::PcuRangePolicy::Clamp;
    assert!(assess(&built, options).is_err());
    options.range_policy = crate::PcuRangePolicy::Reject;
    options.numerical_options.reproducibility = crate::PcuReproducibility::PortableV1;
    // The effect owns its captured tuple, not a subsequently changed caller
    // default. Recapture to actually request Portable arithmetic.
    assert!(assess(&built, options).is_ok());
    let portable = __pcu_capture_tensor_program::<f64, 2, _>(
        [PcuSourceShape::Slice { length: 7 }; 2],
        options.float_underflow,
        options.numerical_mode,
        options.numerical_options,
        unused_effect::__pcu_capture_entry,
    )
    .unwrap();
    assert!(assess(&portable, options).is_err());
}

#[crate::pcu(crate_path = crate)]
fn integer_repeated<T: PcuScalar>(
    unused: &[T],
    input: &[T],
) -> Result<crate::PcuTensor<T>, crate::PcuExecutionError> {
    pcu::mul(input, input)
}

fn integer_metadata<T: crate::PcuCheckedInteger>() {
    for mode in [
        crate::PcuNumericalMode::Boundary,
        crate::PcuNumericalMode::Strict,
    ] {
        for underflow in [
            crate::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            crate::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            crate::PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            let options = PcuExecutionPolicy {
                numerical_mode: mode,
                float_underflow: underflow,
                ..Default::default()
            };
            let built = __pcu_capture_tensor_program::<T, 2, _>(
                [
                    PcuSourceShape::Slice { length: 1 },
                    PcuSourceShape::Slice { length: 7 },
                ],
                underflow,
                mode,
                options.numerical_options,
                integer_repeated::__pcu_capture_entry::<T>,
            )
            .unwrap();
            assert_eq!(built.input_indices, [1]);
            let requirements = super::requirements(&built, options).unwrap();
            assert_eq!(requirements.float_underflow, underflow);
            assert_eq!(requirements.numerical_mode, mode);
            let plan = fusion_pcu_mlx::MlxCheckedTensorIntegerPlan::assess_program(
                &built.program,
                requirements,
            )
            .unwrap();
            assert_eq!(plan.input_values().len(), 1);
            assert_eq!(plan.operand_inputs(), [0, 0]);
            assert_eq!(plan.scalar_type(), T::TYPE);
            assert_eq!(plan.element_count(), 7);
            let effect = built.program.graph().node(plan.effect()).unwrap();
            assert_eq!(effect.float_underflow_policy, None);
            assert_eq!(effect.numerical_mode, None);
            // Integer source now reaches its exact static factory; Clamp and
            // Portable remain separate unqualified owned-result contracts.
            assert!(matches!(
                super::program::Plan::assess(&built.program, requirements).unwrap(),
                super::program::Plan::Integer
            ));
            assert_eq!(assess(&built, options).unwrap(), requirements);
            let clamp = PcuExecutionPolicy {
                range_policy: crate::PcuRangePolicy::Clamp,
                ..options
            };
            assert!(assess(&built, clamp).is_err());
        }
    }
}

#[test]
fn fourteen_integer_types_have_no_floating_tininess_metadata() {
    integer_metadata::<u8>();
    integer_metadata::<i8>();
    integer_metadata::<u16>();
    integer_metadata::<i16>();
    integer_metadata::<u32>();
    integer_metadata::<i32>();
    integer_metadata::<u64>();
    integer_metadata::<i64>();
    integer_metadata::<u128>();
    integer_metadata::<i128>();
    integer_metadata::<crate::PcuU256>();
    integer_metadata::<crate::PcuI256>();
    integer_metadata::<crate::PcuU512>();
    integer_metadata::<crate::PcuI512>();
}
