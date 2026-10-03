//! Genuine annotated binary32/binary64 arithmetic beside exact checked native controls.
#[rustfmt::skip]
use criterion::{Criterion, criterion_group, criterion_main};
use fusion_pcu_cpu::PcuCpuCheckedBinary;
#[rustfmt::skip]
use pcu_facade::{PcuBindingRef, PcuDispatchFloatBinaryOp, PcuHostArgument, PcuHostKernelBackend, PcuPreparedHostKernel};
#[path = "../../tests/prepared_binary/source/source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;

fn benchmarks(criterion: &mut Criterion) {
    macro_rules! case {
        ($module:ident, $ty:ty, $prepare:ident, $ir:ident, $bindings:ident, $op:ident, $count:expr) => {{
            let backend = PcuCpuCheckedBinary::<$ty>::new();
            let call = source::$module::$prepare::<$count, _>(&backend).unwrap();
            let bindings = source::$module::$bindings();
            let builder = source::$module::$ir::<$count>(&bindings).unwrap();
            let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
            support::compare::<$ty, $count>(
                criterion,
                stringify!($ty),
                PcuDispatchFloatBinaryOp::$op,
                call,
                move |lhs, rhs, output| {
                    graph.call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), lhs),
                        PcuHostArgument::read(PcuBindingRef::new(0, 1), rhs),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
                    ])
                },
                [0.0, 1.0, 2.0, 3.0, <$ty>::NAN],
            );
        }};
    }
    macro_rules! width {
        ($module:ident,$ty:ty,$count:expr) => {
            case!($module, $ty, add_prepare, add_ir, add_bindings, Add, $count);
            case!($module, $ty, sub_prepare, sub_ir, sub_bindings, Sub, $count);
            case!($module, $ty, mul_prepare, mul_ir, mul_bindings, Mul, $count);
            case!($module, $ty, div_prepare, div_ir, div_bindings, Div, $count);
        };
    }
    width!(f32_source, f32, 16);
    width!(f64_source, f64, 16);
    width!(f32_source, f32, 4096);
    width!(f64_source, f64, 4096);
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
