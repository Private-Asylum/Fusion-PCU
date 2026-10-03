//! Annotated representation-only copies and equivalent native controls at identical host boundaries.
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
use fusion_pcu_cpu::PcuCpuIdentity;
#[rustfmt::skip]
use pcu_facade::{
    PcuF8E4M3FnBits, PcuF8E5M2Bits,
    PcuBf16Bits,PcuF16Bits,PcuF128Bits,PcuF256Bits,PcuU256,PcuI256,PcuU512,PcuI512,
    PcuHostArgument,PcuBindingRef,PcuHostKernelBackend,PcuPreparedHostKernel,
};
#[path = "../../tests/prepared_identity/source/source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;
fn benchmarks(criterion: &mut Criterion) {
    macro_rules! case {
        ($ty:ty,$values:expr,$count:expr) => {{
            let call = source::copy_prepare::<$ty, $count, _>(&PcuCpuIdentity).unwrap();
            let bindings = source::copy_bindings::<$ty>();
            let builder = source::copy_ir::<$ty, $count>(&bindings).unwrap();
            let mut graph = PcuCpuIdentity.prepare_host_kernel(&builder.ir()).unwrap();
            support::compare::<$ty, $count>(
                criterion,
                call,
                move |input, output| {
                    graph.call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                    ])
                },
                $values,
            );
        }};
    }
    macro_rules! widths {
        ($count:expr) => {
            case!(
                PcuF8E4M3FnBits,
                [
                    PcuF8E4M3FnBits::from_bits(0x7f),
                    PcuF8E4M3FnBits::from_bits(0x80)
                ],
                $count
            );
            case!(
                PcuF8E5M2Bits,
                [
                    PcuF8E5M2Bits::from_bits(0x7d),
                    PcuF8E5M2Bits::from_bits(0x80)
                ],
                $count
            );
            case!(i8, [1, 2], $count);
            case!(u8, [1, 2], $count);
            case!(i16, [1, 2], $count);
            case!(u16, [1, 2], $count);
            case!(i32, [1, 2], $count);
            case!(u32, [1, 2], $count);
            case!(i64, [1, 2], $count);
            case!(u64, [1, 2], $count);
            case!(i128, [1, 2], $count);
            case!(u128, [1, 2], $count);
            case!(f32, [f32::from_bits(0x7f80_0001), -0.0], $count);
            case!(f64, [f64::from_bits(0x7ff0_0000_0000_0001), -0.0], $count);
            case!(
                PcuF16Bits,
                [PcuF16Bits::from_bits(0x7c01), PcuF16Bits::from_bits(0x8000)],
                $count
            );
            case!(
                PcuBf16Bits,
                [
                    PcuBf16Bits::from_bits(0x7f81),
                    PcuBf16Bits::from_bits(0x8000)
                ],
                $count
            );
            case!(
                PcuU256,
                [[1; 4], [u64::MAX; 4]].map(PcuU256::from_limbs_le),
                $count
            );
            case!(
                PcuI256,
                [[1; 4], [u64::MAX; 4]].map(PcuI256::from_limbs_le),
                $count
            );
            case!(
                PcuU512,
                [[1; 8], [u64::MAX; 8]].map(PcuU512::from_limbs_le),
                $count
            );
            case!(
                PcuI512,
                [[1; 8], [u64::MAX; 8]].map(PcuI512::from_limbs_le),
                $count
            );
            case!(
                PcuF128Bits,
                [[1; 2], [u64::MAX; 2]].map(PcuF128Bits::from_limbs_le),
                $count
            );
            case!(
                PcuF256Bits,
                [[1; 4], [u64::MAX; 4]].map(PcuF256Bits::from_limbs_le),
                $count
            );
        };
    }
    widths!(16);
    widths!(4096);
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
