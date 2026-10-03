//! Ordinary source locals retain SSA reuse, observable faults and atomic publication.
#![cfg(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "vulkan",
    feature = "metal",
    feature = "mlx"
))]

use core::fmt::Debug;
use std::sync::Mutex;
use fusion_pcu::pcu;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[rustfmt::skip]
use fusion_pcu::global::{
    configure,
    clear_thread_cache,
    PcuBackendChoice,
    PcuExecutionError,
    PcuExecutionPolicy,
    use_defaults,
};

static POLICY_LOCK: Mutex<()> = Mutex::new(());

#[path = "helpers/helpers.rs"]
mod helpers;
#[path = "mutation/mutation.rs"]
mod mutation;
#[path = "transport/transport.rs"]
mod transport;
#[cfg(feature = "cpu")]
#[path = "typed/typed.rs"]
mod typed;

#[pcu(invocations = N)]
fn local_chain<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let value = input[id];
    let doubled = value + value;
    output[id] = doubled * value;
}

#[pcu(invocations = 3)]
fn grid_chain<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let value = input[id];
        let value = value + input[0];
        output[id] = value * value;
        id += stride;
    }
}

#[pcu(invocations = R * C)]
fn matrix_chain<T: PcuCheckedFloat, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    let row = id / C;
    let col = id % C;
    let value = input[row][col];
    let doubled = value + value;
    output[row][col] = doubled * value;
}

#[pcu(invocations = N, flag(reject_subnormal_result), flag(clamp_range))]
fn clamped_chain<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let value = input[id];
    let doubled = value + value;
    output[id] = doubled * value;
}

#[pcu(invocations = N)]
fn unused_division<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    denominator: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let _must_check = input[id] / denominator[id];
    let value = input[id];
    output[id] = value + value;
}

#[pcu(invocations = N)]
fn ordered_stores<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    stage[id] = original + original;
    let updated = stage[id];
    output[id] = updated * original;
}

#[pcu(invocations = 3)]
fn ordered_grid<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let original = input[id];
        output[id] = original + original;
        let updated = output[id];
        output[id] = updated * original;
        id += stride;
    }
}

#[pcu(invocations = N)]
fn cross_lane_stores<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
    let shared = output[0];
    output[id] = shared * input[id];
}

#[pcu(invocations = N, flag(reject_subnormal_result), flag(clamp_range))]
fn ordered_clamp<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    stage[id] = original + original;
    let updated = stage[id];
    output[id] = updated * original;
}

trait Sample: PcuCheckedFloat + PartialEq + Debug {
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const FOUR: Self;
    const EIGHT: Self;
    const MINIMUM: Self;
    const DOUBLE_MINIMUM: Self;
    const INVALID: Self;
}

macro_rules! encoded_sample {
    ($ty:ty, $one:expr, $two:expr, $four:expr, $eight:expr, $invalid:expr) => {
        impl Sample for $ty {
            const ZERO: Self = Self::from_bits(0);
            const ONE: Self = Self::from_bits($one);
            const TWO: Self = Self::from_bits($two);
            const FOUR: Self = Self::from_bits($four);
            const EIGHT: Self = Self::from_bits($eight);
            const MINIMUM: Self = Self::from_bits(1);
            const DOUBLE_MINIMUM: Self = Self::from_bits(2);
            const INVALID: Self = Self::from_bits($invalid);
        }
    };
}
encoded_sample!(PcuF16Bits, 0x3c00, 0x4000, 0x4400, 0x4800, 0x7c00);
encoded_sample!(PcuBf16Bits, 0x3f80, 0x4000, 0x4080, 0x4100, 0x7f80);
encoded_sample!(PcuF8E4M3FnBits, 0x38, 0x40, 0x48, 0x50, 0x7f);
encoded_sample!(PcuF8E5M2Bits, 0x3c, 0x40, 0x44, 0x48, 0x7c);

macro_rules! native_sample {
    ($ty:ty) => {
        impl Sample for $ty {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const FOUR: Self = 4.0;
            const EIGHT: Self = 8.0;
            const MINIMUM: Self = Self::from_bits(1);
            const DOUBLE_MINIMUM: Self = Self::from_bits(2);
            const INVALID: Self = Self::INFINITY;
        }
    };
}
native_sample!(f32);
native_sample!(f64);

fn verify<T: Sample>() {
    let mut output = [T::ONE; 9];
    local_chain::<T, 7>(&[T::ONE; 7], &mut output).unwrap();
    assert_eq!(&output[..7], &[T::TWO; 7]);
    assert_eq!(&output[7..], &[T::ONE; 2]);
    local_chain::<T, 7>(&[T::TWO; 7], &mut output).unwrap();
    assert_eq!(&output[..7], &[T::EIGHT; 7]);
    assert_eq!(&output[7..], &[T::ONE; 2]);

    grid_chain::<T, 7>(&[T::ONE; 7], &mut output).unwrap();
    assert_eq!(&output[..7], &[T::FOUR; 7]);
    assert_eq!(&output[7..], &[T::ONE; 2]);
    let mut matrix = [[T::ZERO; 3]; 2];
    matrix_chain(&[[T::TWO; 3]; 2], &mut matrix).unwrap();
    assert_eq!(matrix, [[T::EIGHT; 3]; 2]);

    let before = output;
    let fault = unused_division::<T, 7>(
        &[T::ONE; 7],
        &[T::ONE, T::ZERO, T::ONE, T::ONE, T::ONE, T::ONE, T::ONE],
        &mut output,
    )
    .unwrap_err();
    assert!(matches!(fault, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::DivideByZero
        && fault.invocation_id == 1 && !fault.recovered));
    assert_eq!(output, before); // An unused initializer cannot erase its checked operation.

    output.fill(T::ONE);
    let fault = clamped_chain::<T, 7>(&[T::MINIMUM; 7], &mut output).unwrap_err();
    assert!(matches!(fault, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow
        && fault.invocation_id == 0 && fault.recovered));
    assert_eq!(&output[..7], &[T::ZERO; 7]);
    assert_eq!(&output[7..], &[T::ONE; 2]);

    output.fill(T::ONE);
    let before = output;
    let fault = clamped_chain::<T, 7>(
        &[
            T::MINIMUM,
            T::INVALID,
            T::ONE,
            T::ONE,
            T::ONE,
            T::ONE,
            T::ONE,
        ],
        &mut output,
    )
    .unwrap_err();
    assert!(matches!(fault, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand
        && fault.invocation_id == 1 && !fault.recovered));
    assert_eq!(output, before); // A later fatal error defeats earlier useful recovery.
    local_chain::<T, 7>(&[T::TWO; 7], &mut output).unwrap();
    assert_eq!(&output[..7], &[T::EIGHT; 7]);
    verify_ordered::<T>();
}

fn verify_ordered<T: Sample>() {
    let mut output = [T::ONE; 9];
    let error = cross_lane_stores::<T, 7>(&[T::TWO; 7], &mut output).unwrap_err();
    assert!(
        matches!(&error, PcuExecutionError::NoCompatibleDevice { rejected, .. } if !rejected.is_empty()),
        "{error:?}"
    );
    assert_eq!(output, [T::ONE; 9]); // The source does not invent a cross-lane snapshot contract.
    // A single logical lane has no cross-lane dependency: the explicit load
    // must observe its own preceding store, not the caller's old bytes.
    cross_lane_stores::<T, 1>(&[T::TWO], &mut output).unwrap();
    assert_eq!(output[0], T::EIGHT);
    assert_eq!(&output[1..], &[T::ONE; 8]);
    let mut stage = [T::ONE; 9];
    ordered_stores::<T, 7>(&[T::TWO; 7], &mut stage, &mut output).unwrap();
    assert_eq!(&stage[..7], &[T::FOUR; 7]);
    assert_eq!(&output[..7], &[T::EIGHT; 7]);
    assert_eq!(&stage[7..], &[T::ONE; 2]);
    assert_eq!(&output[7..], &[T::ONE; 2]);
    ordered_grid::<T, 7>(&[T::ONE; 7], &mut output).unwrap();
    assert_eq!(&output[..7], &[T::TWO; 7]);
    assert_eq!(&output[7..], &[T::ONE; 2]);

    let before_stage = stage;
    let before_output = output;
    let fault = ordered_stores::<T, 7>(
        &[T::TWO, T::INVALID, T::ONE, T::ONE, T::ONE, T::ONE, T::ONE],
        &mut stage,
        &mut output,
    )
    .unwrap_err();
    assert!(matches!(fault, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand
        && fault.invocation_id == 1 && !fault.recovered));
    assert_eq!(stage, before_stage);
    assert_eq!(output, before_output);

    let fault = ordered_clamp::<T, 7>(&[T::MINIMUM; 7], &mut stage, &mut output).unwrap_err();
    assert!(matches!(fault, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::ArithmeticUnderflow && fault.recovered));
    // Tight subnormal policy reports the Add, retaining its defined gradual
    // recovery. The subsequent Mul independently rounds below minimum to zero.
    assert_eq!(&stage[..7], &[T::DOUBLE_MINIMUM; 7]);
    assert_eq!(&output[..7], &[T::ZERO; 7]);
    assert_eq!(&stage[7..], &[T::ONE; 2]);
    assert_eq!(&output[7..], &[T::ONE; 2]);
    let before_stage = stage;
    let before_output = output;
    let fault = ordered_clamp::<T, 7>(
        &[
            T::MINIMUM,
            T::INVALID,
            T::ONE,
            T::ONE,
            T::ONE,
            T::ONE,
            T::ONE,
        ],
        &mut stage,
        &mut output,
    )
    .unwrap_err();
    assert!(matches!(fault, PcuExecutionError::ArithmeticFault(fault)
        if fault.kind == PcuExecutionFaultKind::InvalidFloatingOperand && !fault.recovered));
    assert_eq!(stage, before_stage);
    assert_eq!(output, before_output);
}

fn verify_backend(backend: PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    configure(PcuExecutionPolicy {
        backend,
        ..PcuExecutionPolicy::default()
    })
    .unwrap();
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
fn six_format_locals_execute_once_preserve_faults_and_publish_atomically() {
    verify_backend(PcuBackendChoice::Cpu);
}

macro_rules! provider_gate {
    ($feature:literal, $test:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "required native composed locals/ordered-store source publication qualification"]
        fn $test() {
            verify_backend(PcuBackendChoice::$backend);
        }
    };
}
provider_gate!("rocm", rocm_six_format_locals_and_ordered_stores, Rocm);
provider_gate!("cuda", cuda_six_format_locals_and_ordered_stores, Cuda);
provider_gate!(
    "vulkan",
    vulkan_six_format_locals_and_ordered_stores,
    Vulkan
);
provider_gate!("metal", metal_six_format_locals_and_ordered_stores, Metal);
provider_gate!("mlx", mlx_six_format_locals_and_ordered_stores, Mlx);
