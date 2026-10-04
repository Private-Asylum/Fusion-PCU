//! Nested ordinary functions retain distinct underflow contracts and original faults.
#![cfg(all(
    feature = "tensor",
    any(
        feature = "cpu",
        feature = "rocm",
        feature = "cuda",
        feature = "vulkan",
        feature = "metal",
        feature = "mlx"
    )
))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedFloat,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuScalar,
    PcuTensor,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};

#[pcu(flag(allow_gradual_underflow))]
fn gradual<T: PcuCheckedFloat>(
    left: &[T; 3],
    right: &[T; 3],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(left, right)
}

#[pcu(flag(strict), flag(reject_subnormal_result))]
fn mixed<T: PcuCheckedFloat>(
    lossy: &[T; 3],
    checked: &[T; 3],
    factor: &[T; 3],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    // A discarded checked effect still runs, under the helper's own policy.
    let _discarded = gradual(lossy, factor)?;
    pcu::mul(checked, factor)
}

fn verify<T: PcuScalar>(owner: &PcuTensor<T>, expected: [T; 3], sentinel: T) {
    let mut host = [sentinel; 5];
    owner.read_into(&mut host).unwrap();
    for (actual, expected) in host[..3].iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    for tail in &host[3..] {
        assert_eq!(tail.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}

fn check<T: PcuCheckedFloat>(smallest: T, normal_floor: T, half: T, one: T, zero: T) {
    let lossy = [one, smallest, one];
    let factor = [half; 3];
    let older = gradual(&lossy, &factor).unwrap();
    verify(&older, [half, zero, half], one);
    // The first multiply's inexact tiny result is explicitly permitted. The second
    // multiply's exact subnormal must instead fail under its declaring function's flag.
    let error = mixed(&lossy, &[normal_floor, one, one], &factor).unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert_eq!(fault.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
    // The permitted discarded effect would fault at lane one if caller flags leaked in.
    assert_eq!(fault.invocation_id, 0);
    assert!(!fault.recovered);
    verify(&older, [half, zero, half], one);
    let retry = mixed(&lossy, &[one; 3], &factor).unwrap();
    global::clear_thread_cache().unwrap();
    verify(&retry, [half; 3], one);
    verify(&older, [half, zero, half], one);
    // Clearing preparation must not merge local policies on recapture either.
    let recaptured = mixed(&lossy, &[one; 3], &factor).unwrap();
    verify(&recaptured, [half; 3], one);
}

fn run(backend: global::PcuBackendChoice) {
    global::configure(global::PcuExecutionPolicy {
        backend,
        float_underflow: PcuFloatUnderflowPolicy::IeeeAfterRounding,
        ..Default::default()
    })
    .unwrap();
    check(
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.5_f32,
        1.0_f32,
        0.0_f32,
    );
    check(
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        0.5_f64,
        1.0_f64,
        0.0_f64,
    );
    check(
        PcuF16Bits::from_bits(1),
        PcuF16Bits::from_bits(0x0400),
        PcuF16Bits::from_bits(0x3800),
        PcuF16Bits::from_bits(0x3c00),
        PcuF16Bits::from_bits(0),
    );
    check(
        PcuBf16Bits::from_bits(1),
        PcuBf16Bits::from_bits(0x0080),
        PcuBf16Bits::from_bits(0x3f00),
        PcuBf16Bits::from_bits(0x3f80),
        PcuBf16Bits::from_bits(0),
    );
    check(
        PcuF8E4M3FnBits::from_bits(1),
        PcuF8E4M3FnBits::from_bits(0x08),
        PcuF8E4M3FnBits::from_bits(0x30),
        PcuF8E4M3FnBits::from_bits(0x38),
        PcuF8E4M3FnBits::from_bits(0),
    );
    check(
        PcuF8E5M2Bits::from_bits(1),
        PcuF8E5M2Bits::from_bits(0x04),
        PcuF8E5M2Bits::from_bits(0x38),
        PcuF8E5M2Bits::from_bits(0x3c),
        PcuF8E5M2Bits::from_bits(0),
    );
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_stage_policy_source_contract() {
    run(global::PcuBackendChoice::Cpu);
}

macro_rules! provider {
    ($feature:literal, $name:ident, $backend:ident, $reason:literal) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = $reason]
        fn $name() {
            run(global::PcuBackendChoice::$backend);
        }
    };
}
provider!(
    "rocm",
    rocm_stage_policy_source_contract,
    Rocm,
    "Requires native ROCm"
);
provider!(
    "cuda",
    cuda_stage_policy_source_contract,
    Cuda,
    "Requires native CUDA"
);
provider!(
    "vulkan",
    vulkan_stage_policy_source_contract,
    Vulkan,
    "Requires native Vulkan"
);
provider!(
    "metal",
    metal_stage_policy_source_contract,
    Metal,
    "Requires native Metal"
);
provider!(
    "mlx",
    mlx_stage_policy_source_contract,
    Mlx,
    "Requires native MLX"
);
