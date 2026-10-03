//! Detached actual-read plans: typed metadata, empty unread arguments and transactional arithmetic.
#[path = "../../benches/low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "graph/graph.rs"]
mod graph;
#[path = "source/source.rs"]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use pcu_facade::{PcuCheckedFloat,PcuHostKernelBackend,PcuHostArgument,PcuPreparedHostKernel,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits};
use fusion_pcu_cpu::PcuCpuHostBackend;
fn same<T: PcuCheckedFloat>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn width<T: PcuCheckedFloat>(finite: impl Fn(f32) -> T + Copy) {
    let backend = PcuCpuHostBackend::scalar();
    let input = [1.0, 0.5, 2.0, 4.0, 1.0, 2.0, 0.5].map(finite);
    let empty: [T; 0] = [];
    let sentinel = finite(16.0);
    for schema in 0..5 {
        let expected = match schema {
            0 => [2.0, 1.0, 4.0, 8.0, 2.0, 4.0, 1.0],
            1 => [1.0, 0.25, 4.0, 16.0, 1.0, 4.0, 0.25],
            2 => [0.0; 7],
            3 => [1.0, 0.5, 2.0, 4.0, 1.0, 2.0, 0.5],
            _ => [1.0; 7],
        }
        .map(finite);
        let mut output = [sentinel; 9];
        let mut plan = graph::with(T::TYPE, 7, schema, |kernel| {
            backend.prepare_host_kernel(kernel).unwrap()
        });
        assert_eq!(
            plan.argument_count(),
            if schema == 0 || schema == 3 { 2 } else { 3 }
        );
        let mut call = |input: &[T], output: &mut [T]| {
            if schema == 0 || schema == 3 {
                plan.call(&mut [
                    PcuHostArgument::read_write(graph::OUTPUT, output),
                    PcuHostArgument::read(graph::INPUT, input),
                ])
            } else {
                plan.call(&mut [
                    PcuHostArgument::read(graph::UNUSED, &empty),
                    PcuHostArgument::read_write(graph::OUTPUT, output),
                    PcuHostArgument::read(graph::INPUT, input),
                ])
            }
        };
        call(&input, &mut output).unwrap();
        same(&output[..7], &expected);
        same(&output[7..], &[sentinel; 2]);
        let saved = output;
        assert!(call(&input[..6], &mut output).is_err());
        same(&output, &saved);
        assert!(call(&input, &mut output[..6]).is_err());
        same(&output, &saved);
        let counts = ffi::count_heap(|| {
            for _ in 0..64 {
                call(&input, &mut output).unwrap();
                same(&output, &saved);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        if schema == 4 {
            let mut zero = input;
            zero[2] = finite(0.0);
            zero[6] = finite(0.0);
            let fault = call(&zero, &mut output).unwrap_err().fault().unwrap();
            assert_eq!(fault.invocation_id, 2);
            assert!(!fault.recovered);
            same(&output, &saved);
            call(&input, &mut output).unwrap();
            same(&output, &saved);
        }
    }
    let mut prepared = source::independent_prepare::<T, 7, _>(&backend).unwrap();
    let mut output = [sentinel; 9];
    prepared(&input, &mut output).unwrap();
    same(&output[..7], &input);
    let mut prepared = source::repeated_prepare::<T, 7, _>(&backend).unwrap();
    prepared(&input, &empty, &mut output).unwrap();
}
#[test]
fn six_formats_read_roles_and_no_warm_heap() {
    width::<f32>(|v| v);
    width::<f64>(f64::from);
    width::<PcuF16Bits>(|v| PcuF16Bits::pcu_checked_from_f32(v).unwrap());
    width::<PcuBf16Bits>(|v| PcuBf16Bits::pcu_checked_from_f32(v).unwrap());
    width::<PcuF8E4M3FnBits>(|v| PcuF8E4M3FnBits::pcu_checked_from_f32(v).unwrap());
    width::<PcuF8E5M2Bits>(|v| PcuF8E5M2Bits::pcu_checked_from_f32(v).unwrap());
}
