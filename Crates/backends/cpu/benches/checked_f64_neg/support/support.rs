//! Native oracle, complete boundary checks, and current-thread Rust allocator census.

use std::cell::Cell;
#[rustfmt::skip]
use std::alloc::{
    GlobalAlloc,
    Layout,
    System,
};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuHostArgumentError,
    PcuCpuHostError,
    PcuCpuPreparedNegError,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBindingRef,
    PcuCheckedFloat,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuPreparedHostKernel,
};

const INPUT: PcuBindingRef = PcuBindingRef::new(0, 0);
const OUTPUT: PcuBindingRef = PcuBindingRef::new(0, 1);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Allocations {
    alloc_calls: usize,
    allocated_bytes: usize,
    deallocations: usize,
    deallocated_bytes: usize,
    reallocations: usize,
    reallocated_bytes: usize,
}

std::thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<Allocations> = const { Cell::new(Allocations {
        alloc_calls: 0, allocated_bytes: 0,
        deallocations: 0, deallocated_bytes: 0,
        reallocations: 0, reallocated_bytes: 0,
    }) };
}

pub struct RustAllocator;

fn record(update: impl FnOnce(&mut Allocations)) {
    let enabled = TRACKING.try_with(Cell::get).unwrap_or(false);
    if enabled {
        let _ = COUNTS.try_with(|counts| {
            let mut value = counts.get();
            update(&mut value);
            counts.set(value);
        });
    }
}

// SAFETY: Every allocation operation delegates unchanged pointer/layout contracts to System.
// Census uses allocation-free thread-local Cells and is disabled during unrelated setup/timing.
unsafe impl GlobalAlloc for RustAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(|counts| {
            counts.alloc_calls += 1;
            counts.allocated_bytes += layout.size();
        });
        // SAFETY: The caller's valid layout is forwarded unchanged to System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(|counts| {
            counts.alloc_calls += 1;
            counts.allocated_bytes += layout.size();
        });
        // SAFETY: The caller's valid layout is forwarded unchanged to System.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record(|counts| {
            counts.deallocations += 1;
            counts.deallocated_bytes += layout.size();
        });
        // SAFETY: The caller's allocation pointer and original layout reach System unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(|counts| {
            counts.reallocations += 1;
            counts.reallocated_bytes += new_size;
        });
        // SAFETY: The caller's pointer, original layout and nonzero new size are unchanged.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

struct CensusGuard;
impl Drop for CensusGuard {
    fn drop(&mut self) {
        TRACKING.with(|tracking| tracking.set(false));
    }
}

pub fn census<T>(action: impl FnOnce() -> T) -> (T, Allocations) {
    COUNTS.with(|counts| counts.set(Allocations::default()));
    TRACKING.with(|tracking| assert!(!tracking.replace(true), "non-nested allocation scope"));
    let guard = CensusGuard;
    let result = action();
    drop(guard);
    (result, COUNTS.with(Cell::get))
}

pub fn invoke(
    prepared: &mut impl PcuPreparedHostKernel<Error = PcuCpuHostError>,
    input: &[f64],
    output: &mut [f64],
) -> Result<(), PcuCpuHostError> {
    prepared.call(&mut [
        PcuHostArgument::read(INPUT, input),
        PcuHostArgument::read_write(OUTPUT, output),
    ])
}

fn fault(invocation: usize, kind: PcuExecutionFaultKind) -> PcuCpuHostError {
    PcuCpuHostError::Neg(PcuCpuPreparedNegError::Fault(PcuExecutionFault {
        recovered: false,
        kind,
        invocation_id: u64::try_from(invocation).unwrap(),
    }))
}

pub fn native<const N: usize>(input: &[f64], output: &mut [f64]) -> Result<(), PcuCpuHostError> {
    // Match full host extent validation before numeric preflight and before any store.
    for (binding, count) in [(INPUT, input.len()), (OUTPUT, output.len())] {
        if count < N {
            return Err(PcuCpuHostError::Arguments(
                PcuCpuHostArgumentError::BufferTooSmall {
                    binding,
                    required_bytes: N * 8,
                    actual_bytes: count * 8,
                },
            ));
        }
    }
    for (invocation, value) in input[..N].iter().enumerate() {
        value
            .pcu_checked_neg()
            .map_err(|kind| fault(invocation, kind))?;
    }
    for (value, result) in input[..N].iter().zip(&mut output[..N]) {
        // Preflight cleared every selected checked exception. Exact sign-bit inversion does
        // not depend on host floating rounding, contraction, denormals or FTZ/DAZ mode.
        *result = f64::from_bits(value.to_bits() ^ (1 << 63));
    }
    Ok(())
}

fn assert_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

pub fn verify<const N: usize>(
    mut call: impl FnMut(&[f64], &mut [f64]) -> Result<(), PcuCpuHostError>,
) {
    let edges = [
        0,
        1 << 63,
        1,
        (1 << 63) | 1,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
    ];
    let mut input: std::vec::Vec<_> = (0..N)
        .map(|index| f64::from_bits(edges[index % edges.len()]))
        .collect();
    let mut expected = std::vec![41.0; N + 3];
    let mut actual = expected.clone();
    native::<N>(&input, &mut expected).unwrap();
    call(&input, &mut actual).unwrap();
    assert_bits(&actual, &expected);
    input.fill(3.0);
    native::<N>(&input, &mut expected).unwrap();
    call(&input, &mut actual).unwrap();
    assert_bits(&actual, &expected);
    for bits in [
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        0x7ff8_0000_0000_0001,
        0xfff0_0000_0000_0001,
    ] {
        input.fill(3.0);
        input[5] = f64::from_bits(bits);
        input[N - 1] = f64::NAN;
        actual.fill(47.0);
        expected.fill(47.0);
        assert_eq!(
            call(&input, &mut actual),
            native::<N>(&input, &mut expected)
        );
        assert_bits(&actual, &expected);
        assert_bits(&actual, &std::vec![47.0; N + 3]);
    }
    input.fill(2.0);
    native::<N>(&input, &mut expected).unwrap();
    call(&input, &mut actual).unwrap();
    assert_bits(&actual, &expected);
    assert_eq!(
        call(&input[..N - 1], &mut actual),
        native::<N>(&input[..N - 1], &mut expected)
    );
    assert_bits(&actual, &expected);
    assert_eq!(
        call(&input, &mut actual[..N - 1]),
        native::<N>(&input, &mut expected[..N - 1])
    );
    assert_bits(&actual, &expected);
}

pub const fn change(input: &mut [f64]) {
    input[0] = if input[0].to_bits() == 1.0_f64.to_bits() {
        2.0
    } else {
        1.0
    };
}
