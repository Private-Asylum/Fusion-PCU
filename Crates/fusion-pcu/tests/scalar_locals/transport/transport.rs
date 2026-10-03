//! Genuine all-carrier source gate. Metadata acceptance is not device parity.

use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuRangePolicy,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};

#[pcu(invocations = N)]
fn ordered<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    stage[id] = original;
    let mut selected = stage[id];
    selected = *seed;
    output[id] = selected;
}

// Unused declarations must not require an artificial local to enter transport.
#[pcu(invocations = N)]
fn simple<T: PcuScalar, const N: usize>(
    ghost: &mut [T],
    output: &mut [T],
    unused: &[T],
    input: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[pcu(invocations = 3)]
fn simple_grid<T: PcuScalar, const N: usize>(input: &[T], ghost: &mut [T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}

#[pcu(invocations = 3)]
fn grid<T: PcuScalar, const N: usize>(input: &[T], stage: &mut [T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let value = input[id];
        stage[id] = value;
        let updated = stage[id];
        output[id] = updated;
        id += stride;
    }
}

#[pcu(invocations = N)]
fn retained_local<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    stage[id] = input[id];
    let saved = stage[id];
    stage[id] = *seed;
    output[id] = saved;
}

#[pcu(invocations = 3)]
fn retained_grid_local<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        stage[id] = input[id];
        let saved = stage[id];
        stage[id] = *seed;
        output[id] = saved;
        id += stride;
    }
}

#[pcu(invocations = N, flag(clamp_range))]
fn explicit_clamp<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let saved = input[id];
    output[id] = saved;
}

const fn sample<T: PcuScalar>(byte: u8) -> T {
    let mut value = core::mem::MaybeUninit::<T>::uninit();
    // SAFETY: PcuScalar is sealed; every carrier is padding-free and admits
    // every bit pattern. Fill exactly the complete aligned host representation
    // before forming a T. No uninitialized byte reference is constructed.
    unsafe {
        value
            .as_mut_ptr()
            .cast::<u8>()
            .write_bytes(byte, core::mem::size_of::<T>());
        value.assume_init()
    }
}

fn assert_bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.encode_le().as_ref(),
            expected.encode_le().as_ref(),
            "{index}: {:?}",
            T::TYPE
        );
    }
}

fn verify<T: PcuScalar>() {
    const N: usize = 47;
    let sentinel = sample::<T>(0xA5);
    let mut stage = [sentinel; N + 3];
    let mut output = [sentinel; N + 5];
    let mut ghost = [];
    // This deliberately includes floating NaN encodings. Transport must copy
    // their exact representation rather than treating them as arithmetic input.
    for (input_byte, seed_byte) in [(0x00_u8, 0xFF_u8), (0x55, 0xC3), (0xFF, 0x00)] {
        let input: [T; N + 7] = core::array::from_fn(|index| {
            sample::<T>(input_byte.wrapping_add(u8::try_from(index).unwrap()))
        });
        let seed = sample::<T>(seed_byte);
        simple::<T, N>(&mut ghost, &mut output, &[], &input).unwrap();
        assert_bits(&output[..N], &input[..N]);
        assert_bits(&output[N..], &[sentinel; 5]);
        simple_grid::<T, N>(&input, &mut ghost, &mut output).unwrap();
        assert_bits(&output[..N], &input[..N]);
        assert_bits(&output[N..], &[sentinel; 5]);
        ordered::<T, N>(&input, &seed, &mut ghost, &mut stage, &mut output).unwrap();
        assert_bits(&stage[..N], &input[..N]);
        assert_bits(&stage[N..], &[sentinel; 3]);
        assert_bits(&output[..N], &[seed; N]);
        assert_bits(&output[N..], &[sentinel; 5]);
        grid::<T, N>(&input, &mut stage, &mut output).unwrap();
        assert_bits(&stage[..N], &input[..N]);
        assert_bits(&output[..N], &input[..N]);
        assert_bits(&stage[N..], &[sentinel; 3]);
        assert_bits(&output[N..], &[sentinel; 5]);
        // A range policy is inert here: exact copies neither clamp bits nor
        // invent a recovered arithmetic fault, even for stored NaN payloads.
        explicit_clamp::<T, N>(&input, &mut output).unwrap();
        assert_bits(&output[..N], &input[..N]);
        assert_bits(&output[N..], &[sentinel; 5]);
        // An SSA value owns its bits. A load must not become a deferred view of
        // storage that a later store can overwrite before the value is used.
        retained_local::<T, N>(&input, &seed, &mut stage, &mut output).unwrap();
        assert_bits(&stage[..N], &[seed; N]);
        assert_bits(&output[..N], &input[..N]);
        assert_bits(&stage[N..], &[sentinel; 3]);
        assert_bits(&output[N..], &[sentinel; 5]);
        retained_grid_local::<T, N>(&input, &seed, &mut stage, &mut output).unwrap();
        assert_bits(&stage[..N], &[seed; N]);
        assert_bits(&output[..N], &input[..N]);
        assert_bits(&stage[N..], &[sentinel; 3]);
        assert_bits(&output[N..], &[sentinel; 5]);
    }
    let input = [sample::<T>(0x3C); N];
    let before_stage = stage;
    let before_output = output;
    assert!(
        ordered::<T, N>(
            &input,
            &sentinel,
            &mut ghost,
            &mut stage,
            &mut output[..N - 1]
        )
        .is_err()
    );
    assert_bits(&stage, &before_stage);
    assert_bits(&output, &before_output);
    ordered::<T, N>(&input, &sentinel, &mut ghost, &mut stage, &mut output).unwrap();
    assert_bits(&stage[..N], &input);
    assert_bits(&output[..N], &[sentinel; N]);
}

fn verify_backend(backend: PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for range_policy in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
        configure(PcuExecutionPolicy {
            backend,
            range_policy,
            ..Default::default()
        })
        .unwrap();
        clear_thread_cache().unwrap();
        verify::<u8>();
        verify::<i8>();
        verify::<u16>();
        verify::<i16>();
        verify::<u32>();
        verify::<i32>();
        verify::<u64>();
        verify::<i64>();
        verify::<u128>();
        verify::<i128>();
        verify::<PcuU256>();
        verify::<PcuI256>();
        verify::<PcuU512>();
        verify::<PcuI512>();
        verify::<PcuF16Bits>();
        verify::<PcuBf16Bits>();
        verify::<PcuF8E4M3FnBits>();
        verify::<PcuF8E5M2Bits>();
        verify::<f32>();
        verify::<f64>();
        verify::<PcuF128Bits>();
        verify::<PcuF256Bits>();
    }
    clear_thread_cache().unwrap();
    use_defaults().unwrap();
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_twenty_two_carrier_ordered_transport() {
    verify_backend(PcuBackendChoice::Cpu);
}

#[pcu(invocations = 7)]
fn unary_five_declarations<T: PcuCheckedFloat>(
    ghost0: &[T],
    output: &mut [T],
    ghost1: &T,
    input: &[T],
    ghost2: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ordinary_unary_uses_five_stack_arguments_without_a_transport_profile() {
    let _guard = POLICY_LOCK.lock().unwrap();
    configure(PcuExecutionPolicy {
        backend: PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    clear_thread_cache().unwrap();
    let mut output = [17.0_f32; 9];
    let input = [-1.0, 0.0, -0.0, 1.0, 2.0, 3.0, 4.0, f32::NAN];
    for _ in 0..3 {
        unary_five_declarations(&[], &mut output, &f32::NAN, &input, &[]).unwrap();
        assert_eq!(&output[..7], &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&output[7..], &[17.0; 2]);
        assert_eq!(output[2].to_bits(), 0);
    }
    clear_thread_cache().unwrap();
    use_defaults().unwrap();
}

macro_rules! transport_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native all-carrier ordered transport source qualification"]
        fn $test() {
            verify_backend(PcuBackendChoice::$backend);
        }
    };
}
transport_gate!("rocm", rocm_twenty_two_carrier_ordered_transport, Rocm);
transport_gate!("cuda", cuda_twenty_two_carrier_ordered_transport, Cuda);
transport_gate!(
    "vulkan",
    vulkan_twenty_two_carrier_ordered_transport,
    Vulkan
);
transport_gate!("metal", metal_twenty_two_carrier_ordered_transport, Metal);
transport_gate!("mlx", mlx_twenty_two_carrier_ordered_transport, Mlx);
