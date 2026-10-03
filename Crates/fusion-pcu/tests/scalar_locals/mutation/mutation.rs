//! Ordinary mutable locals share the immutable chain's exact device operation stream.
use super::*;

#[pcu(invocations = N)]
fn chain<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let mut value = input[id];
    let original = value;
    value = value + value;
    value = value * original;
    output[id] = value;
}

#[pcu(invocations = 3)]
fn grid<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let mut value = input[id];
        value = value + input[0];
        value = value * value;
        output[id] = value;
        id += stride;
    }
}

#[pcu(invocations = N)]
fn overwritten_fault<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    denominator: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let mut value = input[id];
    value = value / denominator[id];
    value = input[id];
    output[id] = value;
}

fn verify<T: Sample>() {
    let bindings = chain_bindings::<T>();
    let mutation = chain_ir::<T, 7>(&bindings).unwrap();
    let reference = local_chain_ir::<T, 7>(&bindings).unwrap();
    mutation.with_ir(|mutation| {
        reference.with_ir(|reference| assert_eq!(mutation.ops, reference.ops));
    });
    let bindings = grid_bindings::<T>();
    let mutation = grid_ir::<T, 7>(&bindings).unwrap();
    let reference = grid_chain_ir::<T, 7>(&bindings).unwrap();
    mutation.with_ir(|mutation| {
        reference.with_ir(|reference| assert_eq!(mutation.ops, reference.ops));
    });

    let mut output = [T::ONE; 9];
    for (input, expected) in [(T::ONE, T::TWO), (T::TWO, T::EIGHT)] {
        chain::<T, 7>(&[input; 7], &mut output).unwrap();
        assert_eq!(&output[..7], &[expected; 7]);
        assert_eq!(&output[7..], &[T::ONE; 2]);
    }
    grid::<T, 7>(&[T::ONE; 7], &mut output).unwrap();
    assert_eq!(&output[..7], &[T::FOUR; 7]);
    let before = output;
    let error = overwritten_fault::<T, 7>(
        &[T::ONE; 7],
        &[T::ONE, T::ZERO, T::ONE, T::ONE, T::ONE, T::ONE, T::ONE],
        &mut output,
    )
    .unwrap_err();
    assert!(
        matches!(&error, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::DivideByZero
        && fault.invocation_id == 1 && !fault.recovered),
        "{error:?}"
    );
    assert_eq!(output, before);
    overwritten_fault::<T, 7>(&[T::TWO; 7], &[T::ONE; 7], &mut output).unwrap();
    assert_eq!(&output[..7], &[T::TWO; 7]);
    assert_eq!(&output[7..], &[T::ONE; 2]);
}

fn verify_backend(backend: PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    configure(PcuExecutionPolicy {
        backend,
        ..Default::default()
    })
    .unwrap();
    clear_thread_cache().unwrap();
    verify::<PcuF16Bits>();
    verify::<PcuBf16Bits>();
    verify::<PcuF8E4M3FnBits>();
    verify::<PcuF8E5M2Bits>();
    verify::<f32>();
    verify::<f64>();
    clear_thread_cache().unwrap();
    use_defaults().unwrap();
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_six_format_mutable_locals() {
    verify_backend(PcuBackendChoice::Cpu);
}

macro_rules! gate {
    ($feature:literal, $name:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native mutable-local source and overwritten-fault qualification"]
        fn $name() {
            verify_backend(PcuBackendChoice::$backend);
        }
    };
}
gate!("rocm", rocm_six_format_mutable_locals, Rocm);
gate!("cuda", cuda_six_format_mutable_locals, Cuda);
gate!("vulkan", vulkan_six_format_mutable_locals, Vulkan);
gate!("metal", metal_six_format_mutable_locals, Metal);
gate!("mlx", mlx_six_format_mutable_locals, Mlx);
