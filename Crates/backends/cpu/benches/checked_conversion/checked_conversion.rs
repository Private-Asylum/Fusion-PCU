//! Actual annotated mixed-width casts, explicit graph diagnostics and checked native controls.
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
use fusion_pcu_cpu::PcuCpuCheckedConversion;
#[rustfmt::skip]
use pcu_facade::{PcuCheckedFloatConversion,PcuCheckedFloatWidening,PcuHostKernelBackend,PcuPreparedHostKernel,PcuHostArgument,PcuBindingRef};
#[path = "../../tests/prepared_conversion/source/source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;
fn benchmarks(criterion: &mut Criterion) {
    macro_rules! case {
        ($source:ty,$destination:ty,$prepare:ident,$ir:ident,$bindings:ident,$evaluate:expr,$count:expr) => {{
            let call = source::$prepare::<$count, _>(&PcuCpuCheckedConversion).unwrap();
            let bindings = source::$bindings();
            let builder = source::$ir::<$count>(&bindings).unwrap();
            let mut graph = PcuCpuCheckedConversion
                .prepare_host_kernel(&builder.ir())
                .unwrap();
            support::compare::<$source, $destination, $count>(
                criterion,
                call,
                move |input, output| {
                    graph.call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                    ])
                },
                $evaluate,
                [1.25, 2.5, <$source>::NAN],
                99.0,
            );
        }};
    }
    case!(
        f32,
        f64,
        widen_prepare,
        widen_ir,
        widen_bindings,
        PcuCheckedFloatWidening::pcu_checked_to_f64,
        16
    );
    case!(
        f64,
        f32,
        narrow_prepare,
        narrow_ir,
        narrow_bindings,
        PcuCheckedFloatConversion::pcu_checked_to_f32,
        16
    );
    case!(
        f32,
        f64,
        widen_prepare,
        widen_ir,
        widen_bindings,
        PcuCheckedFloatWidening::pcu_checked_to_f64,
        4096
    );
    case!(
        f64,
        f32,
        narrow_prepare,
        narrow_ir,
        narrow_bindings,
        PcuCheckedFloatConversion::pcu_checked_to_f32,
        4096
    );
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
