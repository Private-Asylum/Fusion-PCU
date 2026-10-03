//! Matched fresh input/output/status/completion/prefix-read/drop binary peers.
#[rustfmt::skip]
use fusion_pcu_metal::{MetalSession,MetalPreparedFloatBinaryKernel,MetalPreparedFloatBinary,MetalBuffer,MetalError,MetalHostKernelError};
#[rustfmt::skip]
use pcu_facade::{PcuCheckedFloat,PcuExecutionError,PcuFloatUnderflowPolicy as Policy,PcuDispatchFloatBinaryOp as Op,PcuHostArgument,PcuBindingRef,PcuRangePolicy as Range};
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
use criterion::Criterion;
use std::hint::black_box;
#[path = "../../support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../checked_neg/census/census.rs"]
mod census;
#[path = "../graph/graph.rs"]
mod graph;
#[path = "../source/source.rs"]
mod source;
trait Sample: PcuCheckedFloat {
    fn value(value: f32) -> Self;
}
impl Sample for f32 {
    fn value(value: f32) -> Self {
        value
    }
}
impl Sample for f64 {
    fn value(value: f32) -> Self {
        Self::from(value)
    }
}
macro_rules! low_sample {
    ($ty:ty) => {
        impl Sample for $ty {
            fn value(value: f32) -> Self {
                Self::pcu_checked_from_f32(value).unwrap()
            }
        }
    };
}
low_sample!(pcu_facade::PcuF16Bits);
low_sample!(pcu_facade::PcuBf16Bits);
low_sample!(pcu_facade::PcuF8E4M3FnBits);
low_sample!(pcu_facade::PcuF8E5M2Bits);
trait DeviceMap {
    fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
        bytes: usize,
    ) -> Result<(), MetalError>;
}
impl DeviceMap for MetalPreparedFloatBinaryKernel {
    fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
        _bytes: usize,
    ) -> Result<(), MetalError> {
        Self::execute_into(self, inputs, output)
    }
}
impl DeviceMap for MetalPreparedFloatBinary {
    fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
        bytes: usize,
    ) -> Result<(), MetalError> {
        Self::execute_into(self, inputs, output, bytes)
    }
}
fn host_call<T: PcuCheckedFloat>(
    session: &MetalSession,
    map: &impl DeviceMap,
    input: &[T],
    right: &[T],
    output: &mut [T],
    count: usize,
) -> Result<(), MetalError> {
    if output.len() < count {
        return Err(MetalError::InvalidExtent);
    }
    let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
    let device = session.upload_bytes(bytes.bytes())?;
    let right_argument = PcuHostArgument::read(PcuBindingRef::new(0, 1), right);
    let rhs = session.upload_bytes(right_argument.bytes())?;
    let bytes = count
        .checked_mul(T::HOST_SIZE)
        .ok_or(MetalError::InvalidExtent)?;
    let result = session.allocate_zeroed_bytes(bytes)?;
    let recovered = match map.execute_into([&device, &rhs], &result, bytes) {
        Ok(()) => None,
        Err(MetalError::Arithmetic(fault)) if fault.recovered => Some(fault),
        Err(error) => return Err(error),
    };
    let mut destination =
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output[..count]);
    result.read_into_bytes(destination.bytes_mut().expect("mutable prefix"))?;
    recovered.map_or(Ok(()), |fault| Err(MetalError::Arithmetic(fault)))
}
#[allow(clippy::too_many_lines)] // One matching four-peer boundary retains bank/tail checks and separate census.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Criterion requires mutability in the noninstrumented peer.
fn operation<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
    op: Op,
    policy: Policy,
    range: Range,
    mut prepared: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), MetalHostKernelError>,
    mut ordinary: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    let prepare = |kernel: &pcu_facade::PcuDispatchKernelIr<'_>| {
        session.prepare_float_binary_kernel(kernel).unwrap()
    };
    let count = u32::try_from(N).unwrap();
    let map = graph::fixture_profile::<T, _>(count, op, policy, range, false, [false; 2], prepare);
    let native = session
        .prepare_checked_float_binary_with_range(T::TYPE, op, policy, range)
        .unwrap();
    let sentinel = T::value(91.0);
    let banks = [1.0, 2.0, 3.0].map(|phase| {
        let input: Vec<T> = (0..N)
            .map(|i| {
                T::value(match i % 4 {
                    0 => -phase,
                    1 => phase,
                    2 => 0.0,
                    _ => -0.0,
                })
            })
            .collect();
        let right: Vec<T> = (0..N)
            .map(|i| T::value(if i.is_multiple_of(2) { 1.0 } else { 2.0 }))
            .collect();
        let mut expected: Vec<T> = input
            .iter()
            .zip(&right)
            .map(|(&value, &rhs)| {
                match op {
                    Op::Add => value.pcu_checked_add_with_policy(rhs, policy),
                    Op::Sub => value.pcu_checked_sub_with_policy(rhs, policy),
                    Op::Mul => value.pcu_checked_mul_with_policy(rhs, policy),
                    Op::Div => value.pcu_checked_div_with_policy(rhs, policy),
                }
                .unwrap()
            })
            .collect();
        expected.push(sentinel);
        let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 1), &expected)
            .bytes()
            .to_vec();
        (input, right, bytes)
    });
    let mut output = vec![sentinel; N + 1];
    let mut execute = |route, bank: usize, verify| {
        let (input, right, expected) = &banks[bank];
        match route {
            0 => prepared(black_box(input), black_box(right), black_box(&mut output)).unwrap(),
            1 => ordinary(black_box(input), black_box(right), black_box(&mut output)).unwrap(),
            2 => host_call(
                session,
                &map,
                black_box(input),
                black_box(right),
                black_box(&mut output),
                N,
            )
            .unwrap(),
            3 => host_call(
                session,
                &native,
                black_box(input),
                black_box(right),
                black_box(&mut output),
                N,
            )
            .unwrap(),
            _ => unreachable!("registered binary peer"),
        }
        if verify {
            assert_eq!(
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &output).bytes(),
                expected
            );
        }
        black_box(&output);
    };
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    #[cfg(not(feature = "allocation-census"))]
    let mut group = criterion.benchmark_group(format!(
        "host_Unspecified_{:?}_{op:?}_{policy:?}_{range:?}_copied_terminal_readback",
        T::TYPE
    ));
    for (route, name) in [
        "source_prepared",
        "ordinary_pcu",
        "neutral_graph",
        "native_checker",
    ]
    .into_iter()
    .enumerate()
    {
        for bank in 0..banks.len() {
            execute(route, bank, true);
        }
        // Every filtered registration gets its own initialized and checked input bank.
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
                    bank = (bank + 1) % banks.len();
                    execute(route, bank, false);
                });
            });
        }
        for bank in 0..banks.len() {
            execute(route, bank, true);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
}
#[allow(clippy::too_many_lines)] // Register every independent op/underflow/range source specialization as a matching peer.
fn cases<T: Sample, const N: usize>(criterion: &mut Criterion) {
    let session = MetalSession::open(0).unwrap();
    macro_rules! run {
        ($op:ident,$policy:ident,$range:ident,$source:ident,$prepare:ident) => {
            operation::<T, N>(
                criterion,
                &session,
                Op::$op,
                Policy::$policy,
                Range::$range,
                source::$prepare::<T, N, _>(&session).unwrap(),
                source::$source::<T, N>,
            );
        };
    }
    run!(
        Add,
        IeeeAfterRounding,
        Reject,
        add_ieee_reject,
        add_ieee_reject_prepare
    );
    run!(
        Add,
        IeeeAfterRounding,
        Clamp,
        add_ieee_clamp,
        add_ieee_clamp_prepare
    );
    run!(
        Add,
        AllowGradualUnderflow,
        Reject,
        add_gradual_reject,
        add_gradual_reject_prepare
    );
    run!(
        Add,
        AllowGradualUnderflow,
        Clamp,
        add_gradual_clamp,
        add_gradual_clamp_prepare
    );
    run!(
        Add,
        RejectSubnormalResult,
        Reject,
        add_tight_reject,
        add_tight_reject_prepare
    );
    run!(
        Add,
        RejectSubnormalResult,
        Clamp,
        add_tight_clamp,
        add_tight_clamp_prepare
    );
    run!(
        Sub,
        IeeeAfterRounding,
        Reject,
        sub_ieee_reject,
        sub_ieee_reject_prepare
    );
    run!(
        Sub,
        IeeeAfterRounding,
        Clamp,
        sub_ieee_clamp,
        sub_ieee_clamp_prepare
    );
    run!(
        Sub,
        AllowGradualUnderflow,
        Reject,
        sub_gradual_reject,
        sub_gradual_reject_prepare
    );
    run!(
        Sub,
        AllowGradualUnderflow,
        Clamp,
        sub_gradual_clamp,
        sub_gradual_clamp_prepare
    );
    run!(
        Sub,
        RejectSubnormalResult,
        Reject,
        sub_tight_reject,
        sub_tight_reject_prepare
    );
    run!(
        Sub,
        RejectSubnormalResult,
        Clamp,
        sub_tight_clamp,
        sub_tight_clamp_prepare
    );
    run!(
        Mul,
        IeeeAfterRounding,
        Reject,
        mul_ieee_reject,
        mul_ieee_reject_prepare
    );
    run!(
        Mul,
        IeeeAfterRounding,
        Clamp,
        mul_ieee_clamp,
        mul_ieee_clamp_prepare
    );
    run!(
        Mul,
        AllowGradualUnderflow,
        Reject,
        mul_gradual_reject,
        mul_gradual_reject_prepare
    );
    run!(
        Mul,
        AllowGradualUnderflow,
        Clamp,
        mul_gradual_clamp,
        mul_gradual_clamp_prepare
    );
    run!(
        Mul,
        RejectSubnormalResult,
        Reject,
        mul_tight_reject,
        mul_tight_reject_prepare
    );
    run!(
        Mul,
        RejectSubnormalResult,
        Clamp,
        mul_tight_clamp,
        mul_tight_clamp_prepare
    );
    run!(
        Div,
        IeeeAfterRounding,
        Reject,
        div_ieee_reject,
        div_ieee_reject_prepare
    );
    run!(
        Div,
        IeeeAfterRounding,
        Clamp,
        div_ieee_clamp,
        div_ieee_clamp_prepare
    );
    run!(
        Div,
        AllowGradualUnderflow,
        Reject,
        div_gradual_reject,
        div_gradual_reject_prepare
    );
    run!(
        Div,
        AllowGradualUnderflow,
        Clamp,
        div_gradual_clamp,
        div_gradual_clamp_prepare
    );
    run!(
        Div,
        RejectSubnormalResult,
        Reject,
        div_tight_reject,
        div_tight_reject_prepare
    );
    run!(
        Div,
        RejectSubnormalResult,
        Clamp,
        div_tight_clamp,
        div_tight_clamp_prepare
    );
}
pub fn run(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: checked binary Metal needs actual hardware");
        return;
    }
    activity::guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    macro_rules! formats {
        ($ty:ty) => {
            cases::<$ty, 257>(criterion);
            cases::<$ty, 65_537>(criterion);
        };
    }
    formats!(f32);
    formats!(f64);
    formats!(pcu_facade::PcuF16Bits);
    formats!(pcu_facade::PcuBf16Bits);
    formats!(pcu_facade::PcuF8E4M3FnBits);
    formats!(pcu_facade::PcuF8E5M2Bits);
    pcu_facade::global::clear_thread_cache().unwrap();
    activity::guard();
}
