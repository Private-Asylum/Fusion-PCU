//! Separate broadcast admission, exact raw-bit repetition and transactional host roles.
mod admission;
#[path = "sample/sample.rs"]
mod sample;
#[path = "source/source.rs"]
mod source;
use sample::{Sample, same};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuIdentity,
    PcuCpuHostBackend,
    PcuScalarIdentityReference,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuSynchronousHostDispatchBackend,
    PcuDispatchSubmission,
    PcuInvocationShape,
    PcuInvocationParameters,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    validate_scalar_identity_kernel,
    validate_scalar_broadcast_kernel,
};
#[allow(clippy::too_many_lines)] // One carrier corpus is replayed through separately admitted source, graph and reference boundaries.
fn check<T: Sample>() -> usize {
    admission::check::<T>();
    let cases = if T::HOST_SIZE <= 2 {
        1usize << (T::HOST_SIZE * 8)
    } else {
        T::HOST_SIZE * 8 + 2 + 1024
    };
    let sentinel = T::pattern(17);
    let backend = PcuCpuHostBackend::scalar();
    macro_rules! profile {
        ($prepare:ident,$ir:ident,$bindings:ident) => {{
            let mut prepared = source::$prepare::<T, 7, _>(&backend).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, 7>(&bindings).unwrap();
            let kernel = builder.ir();
            assert!(validate_scalar_identity_kernel(&kernel, T::TYPE).is_err());
            validate_scalar_broadcast_kernel(&kernel, T::TYPE).unwrap();
            let mut graph = PcuCpuIdentity.prepare_host_kernel(&kernel).unwrap();
            assert_eq!(graph.local_id(), 384 + T::TYPE as u32);
            let mut output = [sentinel; 10];
            let mut reference = [sentinel; 10];
            for seed in 0..cases {
                let input = T::pattern(seed);
                prepared(&input, &mut output).unwrap();
                same(&output[..7], &[input; 7]);
                same(&output[7..], &[sentinel; 3]);
                graph
                    .call(&mut [
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                        PcuHostArgument::read(
                            PcuBindingRef::new(0, 0),
                            core::slice::from_ref(&input),
                        ),
                    ])
                    .unwrap();
                same(&output[..7], &[input; 7]);
                PcuScalarIdentityReference
                    .run_host_direct(
                        PcuDispatchSubmission {
                            kernel: &kernel,
                            shape: PcuInvocationShape::invocations(
                                core::num::NonZeroU32::new(kernel.entry.logical_shape[0]).unwrap(),
                            ),
                        },
                        &mut [
                            PcuHostScalarBinding {
                                target: PcuBindingRef::new(0, 0),
                                slice: PcuHostScalarSlice::Read(core::slice::from_ref(&input)),
                            },
                            PcuHostScalarBinding {
                                target: PcuBindingRef::new(0, 1),
                                slice: PcuHostScalarSlice::ReadWrite(&mut reference),
                            },
                        ],
                        PcuInvocationParameters::empty(),
                    )
                    .unwrap();
                same(&reference[..7], &[input; 7]);
                same(&reference[7..], &[sentinel; 3]);
            }
            let before = output;
            let empty: [T; 0] = [];
            assert!(
                graph
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &empty),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output)
                    ])
                    .is_err()
            );
            same(&output, &before);
            let input = T::pattern(71);
            graph
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), core::slice::from_ref(&input)),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                ])
                .unwrap();
            same(&output[..7], &[input; 7]);
            assert!(prepared(&input, &mut output[..6]).is_err());
            same(&output[..7], &[input; 7]);
        }};
    }
    profile!(direct_prepare, direct_ir, direct_bindings);
    profile!(grid_prepare, grid_ir, grid_bindings);
    cases * 2
}
#[test]
fn all_twenty_two_carriers_complete_narrow_encodings_and_wide_bit_basis() {
    let count = check::<i8>()
        + check::<u8>()
        + check::<i16>()
        + check::<u16>()
        + check::<i32>()
        + check::<u32>()
        + check::<i64>()
        + check::<u64>()
        + check::<i128>()
        + check::<u128>()
        + check::<PcuI256>()
        + check::<PcuU256>()
        + check::<PcuI512>()
        + check::<PcuU512>()
        + check::<PcuF16Bits>()
        + check::<PcuBf16Bits>()
        + check::<PcuF8E4M3FnBits>()
        + check::<PcuF8E5M2Bits>()
        + check::<f32>()
        + check::<f64>()
        + check::<PcuF128Bits>()
        + check::<PcuF256Bits>();
    assert_eq!(count, 559_992);
}
