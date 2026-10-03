//! Preuploaded logical inputs, private fresh output, required terminal gate and immutable owner replacement.
#[rustfmt::skip]
use std::{hint::black_box,rc::Rc};
#[rustfmt::skip]
use pcu_facade::{global,PcuCheckedFloat,PcuScalar,PcuTensor,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuDispatchFloatUnaryOp as Op,PcuFloatUnderflowPolicy as Policy,PcuRangePolicy as Range,PcuHostArgument,PcuBindingRef};
#[rustfmt::skip]
use fusion_pcu_mlx::{MlxRuntime,MlxSession,MlxEncodedArray,MlxEncodedCompletion,MlxPreparedHostKernel,MlxCheckedUnaryControl};
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[path = "../../checked_unary/support/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../checked_unary/census/census.rs"]
mod census;
#[path = "../../checked_unary/graph/graph.rs"]
mod graph;
#[path = "../source/source.rs"]
mod source;
trait Sample: PcuCheckedFloat {
    fn value(value: f32) -> Self;
}
macro_rules! samples{($($ty:ty),+)=>{$(impl Sample for $ty{fn value(value:f32)->Self{Self::pcu_checked_from_f32(value).unwrap()}})+};}
samples!(PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits);
struct Published {
    array: MlxEncodedArray,
    shape: Rc<[usize]>,
}
fn completed(result: MlxEncodedCompletion, shape: &Rc<[usize]>) -> Published {
    let (array, recovered) = result.into_parts();
    assert!(
        recovered.is_none(),
        "finite semantic bank unexpectedly recovered"
    );
    Published {
        array,
        shape: Rc::clone(shape),
    }
}
fn same<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(
        PcuHostArgument::read(PcuBindingRef::new(0, 0), actual).bytes(),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), expected).bytes()
    );
}
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
// One frozen four-peer replacement boundary retains independent changed-bank/tail preflight plus caller census.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Criterion registers the same mutable primary context; census does not use it.
fn compare<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MlxSession,
    op: Op,
    policy: Policy,
    range: Range,
    mut prepared: MlxPreparedHostKernel,
    mut ordinary: impl FnMut(&PcuTensor<T>, &mut PcuTensor<T>),
) {
    let prepare = |kernel: &pcu_facade::PcuDispatchKernelIr<'_>| {
        session.prepare_unary_host_kernel(kernel).unwrap()
    };
    let mut explicit = if range == Range::Reject {
        graph::fixture::<T, _>(u32::try_from(N).unwrap(), op, policy, false, prepare)
    } else {
        graph::fixture_profile::<T, _>(
            u32::try_from(N).unwrap(),
            op,
            policy,
            range,
            false,
            false,
            prepare,
        )
    };
    let mut native: MlxCheckedUnaryControl = session
        .prepare_checked_unary_control(T::TYPE, op, policy, range, N, false)
        .unwrap();
    let sentinel = T::value(7.0);
    let hosts: [Vec<T>; 3] = std::array::from_fn(|bank| {
        (0..N)
            .map(|i| match i % 4 {
                0 => T::value(0.0),
                1 => T::value(-0.0),
                2 => T::value(f32::from(u8::try_from(bank + 1).unwrap())),
                _ => T::value(-(f32::from(u8::try_from(bank + 1).unwrap()))),
            })
            .collect()
    });
    let expected: [Vec<T>; 3] = std::array::from_fn(|bank| {
        let mut values = hosts[bank]
            .iter()
            .map(|&value| {
                match op {
                    Op::Neg => value.pcu_checked_neg_with_policy(policy),
                    Op::Relu => value.pcu_checked_relu_with_policy(policy),
                }
                .unwrap()
            })
            .collect::<Vec<_>>();
        values.extend([sentinel; 3]);
        values
    });
    let inputs: [MlxEncodedArray; 3] =
        std::array::from_fn(|bank| session.upload_encoded(&hosts[bank]).unwrap());
    let ordinary_inputs: [PcuTensor<T>; 3] =
        std::array::from_fn(|bank| source::retain(&hosts[bank]).unwrap());
    let mut ordinary_output = source::retain(&hosts[0]).unwrap();
    let shape: Rc<[usize]> = Rc::from([N]);
    let mut output = Published {
        array: session.upload_encoded(&hosts[0]).unwrap(),
        shape: Rc::clone(&shape),
    };
    let mut readback = vec![sentinel; N + 3];
    let mut execute = |route, bank: usize, verify| {
        match route {
            0 => {
                output = completed(prepared.execute_resident(&inputs[bank]).unwrap(), &shape);
                output.array.read_into(&mut readback).unwrap();
            }
            1 => {
                ordinary(
                    black_box(&ordinary_inputs[bank]),
                    black_box(&mut ordinary_output),
                );
                ordinary_output.read_into(&mut readback).unwrap();
            }
            2 => {
                output = completed(explicit.execute_resident(&inputs[bank]).unwrap(), &shape);
                output.array.read_into(&mut readback).unwrap();
            }
            3 => {
                output = completed(native.execute_resident(&inputs[bank]).unwrap(), &shape);
                output.array.read_into(&mut readback).unwrap();
            }
            _ => unreachable!("registered encoded peer"),
        }
        if verify {
            same(&readback, &expected[bank]);
            assert_eq!(&*output.shape, &[N]);
            assert_eq!(ordinary_output.shape(), &[N]);
        }
        black_box(&readback);
        black_box(&output);
        black_box(&ordinary_output);
    };
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "mlx_encoded_private_replacement_Unspecified/{:?}/{op:?}/{policy:?}/{range:?}",
        T::TYPE
    ));
    for (route, name) in [
        "source_ir_prepared",
        "ordinary_pcu",
        "explicit_graph",
        "native_owned_kernel",
    ]
    .into_iter()
    .enumerate()
    {
        for bank in 0..3 {
            execute(route, bank, true);
        }
        execute(route, 0, true);
        #[cfg(feature = "allocation-census")]
        census::census(
            &format!("{:?}/{op:?}/{policy:?}/{range:?}/{name}/{N}", T::TYPE),
            || {
                execute(route, 1, false);
            },
        );
        #[cfg(not(feature = "allocation-census"))]
        {
            let mut bank = 0;
            group.bench_function(BenchmarkId::new(name, N), |bench| {
                bench.iter(|| {
                    bank = (bank + 1) % 3;
                    execute(route, bank, false);
                });
            });
        }
        for bank in 0..3 {
            execute(route, bank, true);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
}
#[allow(clippy::too_many_lines)] // All twelve independently annotated op/underflow/range peers are registered explicitly.
fn cases<T: Sample, const N: usize>(criterion: &mut Criterion) {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    macro_rules! run {
        ($op:ident,$policy:ident,$range:ident,$entry:ident,$ir:ident,$bindings:ident) => {{
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let prepared = session.prepare_unary_host_kernel(&builder.ir()).unwrap();
            compare::<T, N>(
                criterion,
                &session,
                Op::$op,
                Policy::$policy,
                Range::$range,
                prepared,
                |input, output| source::$entry::<T, N>(input, output).unwrap(),
            );
        }};
    }
    run!(
        Neg,
        IeeeAfterRounding,
        Reject,
        negate,
        negate_ir,
        negate_bindings
    );
    run!(
        Relu,
        IeeeAfterRounding,
        Reject,
        relu,
        relu_ir,
        relu_bindings
    );
    run!(
        Neg,
        AllowGradualUnderflow,
        Reject,
        negate_gradual,
        negate_gradual_ir,
        negate_gradual_bindings
    );
    run!(
        Relu,
        AllowGradualUnderflow,
        Reject,
        relu_gradual,
        relu_gradual_ir,
        relu_gradual_bindings
    );
    run!(
        Neg,
        RejectSubnormalResult,
        Reject,
        negate_tight,
        negate_tight_ir,
        negate_tight_bindings
    );
    run!(
        Relu,
        RejectSubnormalResult,
        Reject,
        relu_tight,
        relu_tight_ir,
        relu_tight_bindings
    );
    run!(
        Neg,
        IeeeAfterRounding,
        Clamp,
        negate_clamp,
        negate_clamp_ir,
        negate_clamp_bindings
    );
    run!(
        Relu,
        IeeeAfterRounding,
        Clamp,
        relu_clamp,
        relu_clamp_ir,
        relu_clamp_bindings
    );
    run!(
        Neg,
        AllowGradualUnderflow,
        Clamp,
        negate_gradual_clamp,
        negate_gradual_clamp_ir,
        negate_gradual_clamp_bindings
    );
    run!(
        Relu,
        AllowGradualUnderflow,
        Clamp,
        relu_gradual_clamp,
        relu_gradual_clamp_ir,
        relu_gradual_clamp_bindings
    );
    run!(
        Neg,
        RejectSubnormalResult,
        Clamp,
        negate_tight_clamp,
        negate_tight_clamp_ir,
        negate_tight_clamp_bindings
    );
    run!(
        Relu,
        RejectSubnormalResult,
        Clamp,
        relu_tight_clamp,
        relu_tight_clamp_ir,
        relu_tight_clamp_bindings
    );
}
pub fn run(criterion: &mut Criterion) {
    activity::guard();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Mlx,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    cases::<PcuF16Bits, 257>(criterion);
    cases::<PcuF16Bits, 65537>(criterion);
    cases::<PcuBf16Bits, 257>(criterion);
    cases::<PcuBf16Bits, 65537>(criterion);
    cases::<PcuF8E4M3FnBits, 257>(criterion);
    cases::<PcuF8E4M3FnBits, 65537>(criterion);
    cases::<PcuF8E5M2Bits, 257>(criterion);
    cases::<PcuF8E5M2Bits, 65537>(criterion);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
