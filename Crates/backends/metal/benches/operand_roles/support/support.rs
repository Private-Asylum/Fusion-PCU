//! Successful dyadic repeated-input boundaries; fault/publication cases remain native gate proofs.
#[rustfmt::skip]
use pcu_facade::{
    PcuScalar,
    PcuCheckedFloat,
    PcuCheckedInteger,
    PcuDispatchFloatBinaryOp as FloatOp,
    PcuDispatchIntegerBinaryOp as IntegerOp,
    PcuFloatUnderflowPolicy as Underflow,
    PcuRangePolicy as Range,
    PcuHostArgument,
    PcuBindingRef,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuExecutionError,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalSession,
    MetalError,
    MetalHostKernelError,
    MetalIntegerOp,
    MetalPreparedIntegerControl,
    MetalPreparedFloatBinary,
};
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
use std::hint::black_box;
#[path = "activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../portable_unary/census/census.rs"]
mod census;
#[path = "graph/graph.rs"]
mod graph;
#[path = "../source/source.rs"]
mod source;
#[derive(Clone, Copy, Debug)]
enum Kind {
    Integer(IntegerOp),
    Float(FloatOp, Underflow),
}
trait Sample: PcuScalar {
    fn small(value: u8) -> Self;
}
macro_rules! integers {($($ty:ty=>$width:literal),+)=>{$(
    impl Sample for $ty {fn small(value:u8)->Self {let mut bytes=[0;$width];bytes[0]=value;Self::decode_le(bytes)}}
)+};}
integers!(i8=>1,u8=>1,i16=>2,u16=>2,i32=>4,u32=>4,i64=>8,u64=>8,i128=>16,u128=>16,PcuI256=>32,PcuU256=>32,PcuI512=>64,PcuU512=>64);
impl Sample for f32 {
    fn small(value: u8) -> Self {
        Self::from(value)
    }
}
impl Sample for f64 {
    fn small(value: u8) -> Self {
        Self::from(value)
    }
}
macro_rules! floats {
    ($($ty:ty=>$word:ty,$bias:expr,$fraction:expr);+)=>{$(
        impl Sample for $ty {
            fn small(value:u8)->Self {
                assert!(value<=7);
                if value==0 {return Self::from_bits(0);}
                let exponent=value.ilog2();
                let fraction=(u32::from(value)-(1<<exponent))<<$fraction;
                let bits=(($bias+exponent)<<$fraction)|(fraction>>exponent);
                Self::from_bits(<$word>::try_from(bits).unwrap())
            }
        }
    )+};
}
floats!(PcuF16Bits=>u16,15,10;PcuBf16Bits=>u16,127,7;PcuF8E4M3FnBits=>u8,7,3;PcuF8E5M2Bits=>u8,15,2);
#[cfg(feature = "allocation-census")]
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "allocation-census")]
fn score(_: &pcu_facade::global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    1
}
enum Native {
    Integer(MetalPreparedIntegerControl),
    Float(MetalPreparedFloatBinary),
}
impl Native {
    fn prepare(
        session: &MetalSession,
        kind: Kind,
        range: Range,
        count: usize,
        scalar: pcu_facade::PcuScalarType,
    ) -> Self {
        match kind {
            Kind::Integer(op) => Self::Integer(
                session
                    .prepare_checked_integer_control(
                        scalar,
                        match op {
                            IntegerOp::Add => MetalIntegerOp::Add,
                            IntegerOp::Sub => MetalIntegerOp::Subtract,
                            IntegerOp::Mul => MetalIntegerOp::Multiply,
                        },
                        range,
                        count,
                        [false; 2],
                    )
                    .unwrap(),
            ),
            Kind::Float(op, uf) => Self::Float(
                session
                    .prepare_checked_float_binary_with_range(scalar, op, uf, range)
                    .unwrap(),
            ),
        }
    }
    fn call<T: PcuScalar>(
        &self,
        session: &MetalSession,
        input: &[T],
        output: &mut [T],
        count: usize,
    ) -> Result<(), MetalError> {
        let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 2), input);
        let buffer = session.upload_bytes(bytes.bytes())?;
        let (completed, recovered) = match self {
            Self::Integer(native) => native.execute_completed_control([&buffer; 2])?,
            Self::Float(native) => {
                native.execute_completed_control([&buffer; 2], count * T::HOST_SIZE)?
            }
        };
        let mut destination =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output[..count]);
        completed.read_into_bytes(destination.bytes_mut().ok_or(MetalError::InvalidExtent)?)?;
        recovered.map_or(Ok(()), |fault| Err(MetalError::Arithmetic(fault)))
    }
}
fn banks<T: Sample>(count: usize, kind: Kind) -> Vec<(Vec<T>, Vec<u8>)> {
    (0..64_u8)
        .map(|phase| {
            let values: Vec<u8> = (0..count)
                .map(|index| {
                    let bit = (phase >> (index % 6)) & 1;
                    if matches!(kind, Kind::Integer(_)) {
                        2 + bit
                    } else {
                        1 + bit
                    }
                })
                .collect();
            let input = values.iter().copied().map(T::small).collect();
            let mut expected: Vec<T> = values
                .into_iter()
                .map(|value| {
                    T::small(match kind {
                        Kind::Integer(IntegerOp::Add) | Kind::Float(FloatOp::Add, _) => {
                            value + value
                        }
                        Kind::Integer(IntegerOp::Sub) | Kind::Float(FloatOp::Sub, _) => 0,
                        Kind::Integer(IntegerOp::Mul) | Kind::Float(FloatOp::Mul, _) => {
                            value * value
                        }
                        Kind::Float(FloatOp::Div, _) => 1,
                    })
                })
                .collect();
            expected.extend([T::small(7); 3]);
            let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 1), &expected)
                .bytes()
                .to_vec();
            (input, bytes)
        })
        .collect()
}
#[allow(clippy::too_many_lines)]
// Four physical peers retain independent oracle/observer boundaries beside their exact registration.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // The ordinary Criterion branch needs this same mutable reference.
fn operation<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
    kind: Kind,
    range: Range,
    mut prepared: impl FnMut(&[T], &mut [T], &[T]) -> Result<(), MetalHostKernelError>,
    mut ordinary: impl FnMut(&[T], &mut [T], &[T]) -> Result<(), PcuExecutionError>,
) {
    let mut explicit = graph::fixture::<T, _>(N, kind, range, |ir| {
        session.prepare_host_kernel(ir).unwrap()
    });
    let native = Native::prepare(session, kind, range, N, T::TYPE);
    let banks = banks::<T>(N, kind);
    for (index, (input, _)) in banks.iter().enumerate() {
        let argument = PcuHostArgument::read(PcuBindingRef::new(0, 2), input);
        let bytes = argument.bytes();
        for (prior, _) in &banks[..index] {
            assert_ne!(
                bytes,
                PcuHostArgument::read(PcuBindingRef::new(0, 2), prior).bytes()
            );
        }
    }
    let mut output = vec![T::small(7); N + 3];
    let mut execute = |route, bank: usize, verify| {
        if route == 4 {
            assert_eq!(
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &output).bytes(),
                &banks[bank].1
            );
            return;
        }
        let (input, expected) = &banks[bank];
        match route {
            0 => prepared(&[], black_box(&mut output), black_box(input)).unwrap(),
            1 => ordinary(&[], black_box(&mut output), black_box(input)).unwrap(),
            2 => explicit
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &[] as &[T]),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), black_box(&mut output)),
                    PcuHostArgument::read(PcuBindingRef::new(0, 2), black_box(input)),
                ])
                .unwrap(),
            3 => native
                .call(session, black_box(input), black_box(&mut output), N)
                .unwrap(),
            _ => unreachable!("registered peer"),
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
        "metal_repeated_roles/{:?}/{kind:?}/{range:?}",
        T::TYPE
    ));
    for (route, name) in [
        "source_prepared",
        "ordinary_pcu",
        "explicit_ir",
        "direct_native",
    ]
    .into_iter()
    .enumerate()
    {
        for bank in 0..64 {
            execute(route, bank, true);
        }
        #[cfg(feature = "allocation-census")]
        {
            let before = SCORES.load(std::sync::atomic::Ordering::Relaxed);
            let mut total = census::Census::default();
            for bank in 0..64 {
                let ((), counts) = census::measure(|| execute(route, bank, false));
                total.alloc_calls += counts.alloc_calls;
                total.realloc_calls += counts.realloc_calls;
                total.dealloc_calls += counts.dealloc_calls;
                total.requested_bytes += counts.requested_bytes;
                // Do not duplicate the measured call to check it: inspect its completed bytes outside capture.
                execute(4, bank, true);
            }
            let scores = SCORES.load(std::sync::atomic::Ordering::Relaxed) - before;
            println!(
                "Rust allocation census/metal_repeated_roles/{:?}/{kind:?}/{range:?}/{N}/{name}: calls=64 alloc={}, realloc={}, dealloc={}, requested_bytes={}, invocation_score_callbacks={scores} (caller Rust only; native/device heap unknown)",
                T::TYPE,
                total.alloc_calls,
                total.realloc_calls,
                total.dealloc_calls,
                total.requested_bytes
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        {
            let mut bank = 0;
            group.bench_function(BenchmarkId::new(name, N), |bench| {
                bench.iter(|| {
                    bank = (bank + 1) % 64;
                    execute(route, bank, false);
                });
            });
        }
        for bank in 0..64 {
            execute(route, bank, true);
        }
    }
    #[cfg(not(feature = "allocation-census"))]
    group.finish();
}
fn float_cases<T: Sample + PcuCheckedFloat, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
) {
    float_add_cases::<T, N>(criterion, session);
    float_sub_cases::<T, N>(criterion, session);
    float_mul_cases::<T, N>(criterion, session);
    float_div_cases::<T, N>(criterion, session);
}
fn float_add_cases<T: Sample + PcuCheckedFloat, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
) {
    macro_rules! run {
        ($uf:ident,$range:ident,$name:ident,$prepare:ident) => {
            operation::<T, N>(
                criterion,
                session,
                Kind::Float(FloatOp::Add, Underflow::$uf),
                Range::$range,
                source::$prepare::<T, N, _>(session).unwrap(),
                source::$name::<T, N>,
            );
        };
    }
    run!(
        IeeeAfterRounding,
        Reject,
        float_add_ieee_reject,
        float_add_ieee_reject_prepare
    );
    run!(
        IeeeAfterRounding,
        Clamp,
        float_add_ieee_clamp,
        float_add_ieee_clamp_prepare
    );
    run!(
        AllowGradualUnderflow,
        Reject,
        float_add_gradual_reject,
        float_add_gradual_reject_prepare
    );
    run!(
        AllowGradualUnderflow,
        Clamp,
        float_add_gradual_clamp,
        float_add_gradual_clamp_prepare
    );
    run!(
        RejectSubnormalResult,
        Reject,
        float_add_tight_reject,
        float_add_tight_reject_prepare
    );
    run!(
        RejectSubnormalResult,
        Clamp,
        float_add_tight_clamp,
        float_add_tight_clamp_prepare
    );
}
fn float_sub_cases<T: Sample + PcuCheckedFloat, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
) {
    macro_rules! run {
        ($uf:ident,$range:ident,$name:ident,$prepare:ident) => {
            operation::<T, N>(
                criterion,
                session,
                Kind::Float(FloatOp::Sub, Underflow::$uf),
                Range::$range,
                source::$prepare::<T, N, _>(session).unwrap(),
                source::$name::<T, N>,
            );
        };
    }
    run!(
        IeeeAfterRounding,
        Reject,
        float_sub_ieee_reject,
        float_sub_ieee_reject_prepare
    );
    run!(
        IeeeAfterRounding,
        Clamp,
        float_sub_ieee_clamp,
        float_sub_ieee_clamp_prepare
    );
    run!(
        AllowGradualUnderflow,
        Reject,
        float_sub_gradual_reject,
        float_sub_gradual_reject_prepare
    );
    run!(
        AllowGradualUnderflow,
        Clamp,
        float_sub_gradual_clamp,
        float_sub_gradual_clamp_prepare
    );
    run!(
        RejectSubnormalResult,
        Reject,
        float_sub_tight_reject,
        float_sub_tight_reject_prepare
    );
    run!(
        RejectSubnormalResult,
        Clamp,
        float_sub_tight_clamp,
        float_sub_tight_clamp_prepare
    );
}
fn float_mul_cases<T: Sample + PcuCheckedFloat, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
) {
    macro_rules! run {
        ($uf:ident,$range:ident,$name:ident,$prepare:ident) => {
            operation::<T, N>(
                criterion,
                session,
                Kind::Float(FloatOp::Mul, Underflow::$uf),
                Range::$range,
                source::$prepare::<T, N, _>(session).unwrap(),
                source::$name::<T, N>,
            );
        };
    }
    run!(
        IeeeAfterRounding,
        Reject,
        float_mul_ieee_reject,
        float_mul_ieee_reject_prepare
    );
    run!(
        IeeeAfterRounding,
        Clamp,
        float_mul_ieee_clamp,
        float_mul_ieee_clamp_prepare
    );
    run!(
        AllowGradualUnderflow,
        Reject,
        float_mul_gradual_reject,
        float_mul_gradual_reject_prepare
    );
    run!(
        AllowGradualUnderflow,
        Clamp,
        float_mul_gradual_clamp,
        float_mul_gradual_clamp_prepare
    );
    run!(
        RejectSubnormalResult,
        Reject,
        float_mul_tight_reject,
        float_mul_tight_reject_prepare
    );
    run!(
        RejectSubnormalResult,
        Clamp,
        float_mul_tight_clamp,
        float_mul_tight_clamp_prepare
    );
}
fn float_div_cases<T: Sample + PcuCheckedFloat, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
) {
    macro_rules! run {
        ($uf:ident,$range:ident,$name:ident,$prepare:ident) => {
            operation::<T, N>(
                criterion,
                session,
                Kind::Float(FloatOp::Div, Underflow::$uf),
                Range::$range,
                source::$prepare::<T, N, _>(session).unwrap(),
                source::$name::<T, N>,
            );
        };
    }
    run!(
        IeeeAfterRounding,
        Reject,
        float_div_ieee_reject,
        float_div_ieee_reject_prepare
    );
    run!(
        IeeeAfterRounding,
        Clamp,
        float_div_ieee_clamp,
        float_div_ieee_clamp_prepare
    );
    run!(
        AllowGradualUnderflow,
        Reject,
        float_div_gradual_reject,
        float_div_gradual_reject_prepare
    );
    run!(
        AllowGradualUnderflow,
        Clamp,
        float_div_gradual_clamp,
        float_div_gradual_clamp_prepare
    );
    run!(
        RejectSubnormalResult,
        Reject,
        float_div_tight_reject,
        float_div_tight_reject_prepare
    );
    run!(
        RejectSubnormalResult,
        Clamp,
        float_div_tight_clamp,
        float_div_tight_clamp_prepare
    );
}

fn integer_cases<T: Sample + PcuCheckedInteger, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
) {
    operation::<T, N>(
        criterion,
        session,
        Kind::Integer(IntegerOp::Add),
        Range::Reject,
        source::integer_add_integer_reject_prepare::<T, N, _>(session).unwrap(),
        source::integer_add_integer_reject::<T, N>,
    );
    operation::<T, N>(
        criterion,
        session,
        Kind::Integer(IntegerOp::Add),
        Range::Clamp,
        source::integer_add_integer_clamp_prepare::<T, N, _>(session).unwrap(),
        source::integer_add_integer_clamp::<T, N>,
    );
    operation::<T, N>(
        criterion,
        session,
        Kind::Integer(IntegerOp::Sub),
        Range::Reject,
        source::integer_sub_integer_reject_prepare::<T, N, _>(session).unwrap(),
        source::integer_sub_integer_reject::<T, N>,
    );
    operation::<T, N>(
        criterion,
        session,
        Kind::Integer(IntegerOp::Sub),
        Range::Clamp,
        source::integer_sub_integer_clamp_prepare::<T, N, _>(session).unwrap(),
        source::integer_sub_integer_clamp::<T, N>,
    );
    operation::<T, N>(
        criterion,
        session,
        Kind::Integer(IntegerOp::Mul),
        Range::Reject,
        source::integer_mul_integer_reject_prepare::<T, N, _>(session).unwrap(),
        source::integer_mul_integer_reject::<T, N>,
    );
    operation::<T, N>(
        criterion,
        session,
        Kind::Integer(IntegerOp::Mul),
        Range::Clamp,
        source::integer_mul_integer_clamp_prepare::<T, N, _>(session).unwrap(),
        source::integer_mul_integer_clamp::<T, N>,
    );
}
pub fn run(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: actual Metal required");
        return;
    }
    activity::guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        #[cfg(feature = "allocation-census")]
        score_invocation: Some(score),
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let session = MetalSession::open(0).unwrap();
    macro_rules! floats {($($ty:ty),+)=>{$(float_cases::<$ty,65>(criterion,&session);float_cases::<$ty,4096>(criterion,&session);)+};}
    macro_rules! integers {($($ty:ty),+)=>{$(integer_cases::<$ty,65>(criterion,&session);integer_cases::<$ty,4096>(criterion,&session);)+};}
    floats!(
        f32,
        f64,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits
    );
    integers!(
        i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, PcuI256, PcuU256, PcuI512, PcuU512
    );
    pcu_facade::global::clear_thread_cache().unwrap();
    activity::guard();
}
