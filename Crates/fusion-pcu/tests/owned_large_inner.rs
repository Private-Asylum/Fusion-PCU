//! Ordinary selected `MatMul` beyond inner sixteen, with exact late-step faults.
//! These are correctness witnesses; ignored native gates are not acceptance.
#![cfg(all(feature = "tensor", feature = "std"))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuScalar,
    PcuTensor,
    dialect::tensor::{
        TensorArithmeticStep,
        TensorError,
    },
};

#[pcu]
fn retain<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(flag(strict))]
fn product<T: PcuScalar, const K: usize>(
    left: &[[T; K]; 2],
    right: &[[T; 2]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(strict))]
fn discarded_product<T: PcuScalar, const K: usize>(
    left: &[[T; K]; 2],
    right: &[[T; 2]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _effect = pcu::matmul(left, right)?;
    pcu::identity(left)
}

fn read<T: PcuScalar>(owner: &PcuTensor<T>, expected: [T; 4], sentinel: T) {
    assert_eq!(owner.shape(), [2, 2]);
    let mut stack = [sentinel; 6];
    owner.read_into(&mut stack).unwrap();
    for (actual, want) in stack[..4].iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), want.encode_le().as_ref());
    }
    for actual in &stack[4..] {
        assert_eq!(actual.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}

fn fault<T: PcuScalar, const K: usize>(
    result: Result<PcuTensor<T>, PcuExecutionError>,
    expected_step: TensorArithmeticStep,
) {
    let error = result.unwrap_err();
    let common = error
        .arithmetic_fault()
        .unwrap_or_else(|| panic!("{error:?}"));
    assert_eq!(common.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(common.invocation_id, 2);
    assert!(!common.recovered);
    let PcuExecutionError::TensorBuild(TensorError::CompoundArithmeticFault {
        element_index,
        reduction_index,
        step,
        ..
    }) = error
    else {
        panic!("lost original MatMul coordinates: {error:?}");
    };
    assert_eq!(
        (element_index, reduction_index, step),
        (2, K - 1, expected_step)
    );
}

#[allow(clippy::too_many_lines)] // One cohort retains pre-fault owners through all four borrow roles and retries.
fn cohort<T: PcuScalar, const K: usize>(value: impl Fn(f64) -> T + Copy, maximum: T) {
    let sentinel = value(19.0);
    let left = [[value(1.0); K], [value(2.0); K]];
    let right = [[value(0.5), value(-0.25)]; K];
    // Independent exact dyadic sums: no backend or PCU arithmetic oracle reuse.
    let k = f64::from(u16::try_from(K).unwrap());
    let expected = [k / 2.0, -k / 4.0, k, -k / 2.0].map(value);
    let older = product(&left, &right).unwrap();
    for generation in [1.0, 2.0, 4.0] {
        let changing = [[value(generation); K], [value(2.0 * generation); K]];
        let resident_left = retain(&changing).unwrap();
        let resident_right = retain(&right).unwrap();
        for role in 0..4 {
            let output = match role {
                0 => product(&changing, &right),
                1 => product::<T, K>(&resident_left, &right),
                2 => product::<T, K>(&changing, &resident_right),
                _ => product::<T, K>(&resident_left, &resident_right),
            }
            .unwrap();
            let want = [k / 2.0, -k / 4.0, k, -k / 2.0].map(|v| value(v * generation));
            read(&output, want, sentinel);
            read(&older, expected, sentinel);
        }
    }
    for expected_step in [TensorArithmeticStep::Multiply, TensorArithmeticStep::Add] {
        let mut failing = left;
        failing[1][K - 1] = maximum;
        let mut factor = [[value(1.0); 2]; K];
        if expected_step == TensorArithmeticStep::Multiply {
            factor[K - 1][0] = value(2.0);
        } else {
            failing[1][K - 2] = maximum;
        }
        let resident_left = retain(&failing).unwrap();
        let resident_right = retain(&factor).unwrap();
        for role in 0..4 {
            let result = match role {
                0 => product(&failing, &factor),
                1 => product::<T, K>(&resident_left, &factor),
                2 => product::<T, K>(&failing, &resident_right),
                _ => product::<T, K>(&resident_left, &resident_right),
            };
            fault::<T, K>(result, expected_step);
            let discarded = match role {
                0 => discarded_product(&failing, &factor),
                1 => discarded_product::<T, K>(&resident_left, &factor),
                2 => discarded_product::<T, K>(&failing, &resident_right),
                _ => discarded_product::<T, K>(&resident_left, &resident_right),
            };
            fault::<T, K>(discarded, expected_step);
            read(&product(&left, &right).unwrap(), expected, sentinel);
            read(&older, expected, sentinel);
        }
    }
    global::clear_thread_cache().unwrap();
    read(&older, expected, sentinel);
    read(&product(&left, &right).unwrap(), expected, sentinel);
}

fn run(backend: global::PcuBackendChoice) {
    for float_underflow in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend,
            float_underflow,
            ..Default::default()
        })
        .unwrap();
        // These small stack inputs exceed the old selected inner-sixteen limit.
        #[allow(clippy::cast_possible_truncation)]
        // All generated values are exact finite dyadics within F32 range.
        let f32_value = |v: f64| v as f32;
        cohort::<f32, 17>(f32_value, f32::MAX);
        cohort::<f32, 33>(f32_value, f32::MAX);
        cohort::<f64, 17>(std::convert::identity, f64::MAX);
        cohort::<f64, 33>(std::convert::identity, f64::MAX);
    }
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_large_inner_source_contract() {
    run(global::PcuBackendChoice::Cpu);
}
macro_rules! provider {
    ($feature:literal, $name:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "Requires actual provider and independently qualified larger-inner selected offer"]
        fn $name() {
            run(global::PcuBackendChoice::$backend);
        }
    };
}
provider!("rocm", rocm_large_inner_source_contract, Rocm);
provider!("cuda", cuda_large_inner_source_contract, Cuda);
provider!("vulkan", vulkan_large_inner_source_contract, Vulkan);
provider!("metal", metal_large_inner_source_contract, Metal);
provider!("mlx", mlx_large_inner_source_contract, Mlx);
