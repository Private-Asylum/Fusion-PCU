//! Fresh staged input, private payload/status, terminal prefix readback and drop for every peer.
#[rustfmt::skip]
use pcu_facade::{global,PcuScalar,PcuHostArgument,PcuBindingRef,PcuExecutionError,PcuU256,PcuI256,PcuU512,PcuI512,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits};
#[rustfmt::skip]
use fusion_pcu_metal::{MetalSession,MetalPreparedCarrierKernel,MetalPreparedCarrierControl,MetalBuffer,MetalError,MetalHostKernelError};
use criterion::Criterion;
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
use std::hint::black_box;
#[path = "../../support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../../checked_neg/census/census.rs"]
mod census;
#[path = "../../../tests/source/carrier/sample/sample.rs"]
mod sample;
#[path = "../../../tests/source/carrier/source/source.rs"]
mod source;
use sample::Sample;
trait Map {
    fn write(&self, input: &MetalBuffer, output: &MetalBuffer) -> Result<(), MetalError>;
}
impl Map for MetalPreparedCarrierKernel {
    fn write(&self, input: &MetalBuffer, output: &MetalBuffer) -> Result<(), MetalError> {
        self.execute_into(input, output)
    }
}
impl Map for MetalPreparedCarrierControl {
    fn write(&self, input: &MetalBuffer, output: &MetalBuffer) -> Result<(), MetalError> {
        self.execute_into(input, output)
    }
}
fn host<T: PcuScalar, const N: usize>(
    session: &MetalSession,
    map: &impl Map,
    input: &[T],
    output: &mut [T],
) -> Result<(), MetalError> {
    if output.len() < N {
        return Err(MetalError::InvalidExtent);
    }
    let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
    let staged = session.upload_bytes(bytes.bytes())?;
    let result = session.allocate_zeroed_bytes(N * T::HOST_SIZE)?;
    map.write(&staged, &result)?;
    let mut destination = PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output[..N]);
    result.read_into_bytes(destination.bytes_mut().expect("exclusive prefix"))
}
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Independent matched four-peer boundary plus bank/tail preflight and separate census.
#[cfg_attr(feature = "allocation-census", allow(clippy::needless_pass_by_ref_mut))] // Primary Criterion registration requires mutable context; census uses the same signature.
fn compare<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    session: &MetalSession,
    profile: &str,
    broadcast: bool,
    map: &MetalPreparedCarrierKernel,
    mut prepared: impl FnMut(&[T], &mut [T]) -> Result<(), MetalHostKernelError>,
    mut ordinary: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    let native = session
        .prepare_carrier_control(T::TYPE, N, broadcast)
        .unwrap();
    let sentinel = T::pattern(619);
    let banks: [(Vec<T>, Vec<T>); 3] = std::array::from_fn(|phase| {
        let input = (0..if broadcast { 1 } else { N })
            .map(|i| T::pattern(i + phase * 277))
            .collect::<Vec<_>>();
        let mut expected = if broadcast {
            vec![input[0]; N]
        } else {
            input.clone()
        };
        expected.extend([sentinel; 3]);
        (input, expected)
    });
    let mut output = vec![sentinel; N + 3];
    let mut execute = |route, bank: usize, verify| {
        let (input, expected) = &banks[bank];
        match route {
            0 => prepared(black_box(input), black_box(&mut output)).unwrap(),
            1 => ordinary(black_box(input), black_box(&mut output)).unwrap(),
            2 => host::<T, N>(session, map, black_box(input), black_box(&mut output)).unwrap(),
            3 => host::<T, N>(session, &native, black_box(input), black_box(&mut output)).unwrap(),
            _ => unreachable!("registered carrier peer"),
        }
        if verify {
            assert_eq!(
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &output).bytes(),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), expected).bytes()
            );
        }
        black_box(&output);
    };
    #[cfg(feature = "allocation-census")]
    let _ = criterion;
    #[cfg(not(feature = "allocation-census"))]
    let mut group =
        criterion.benchmark_group(format!("metal_carrier_Unspecified/{:?}/{profile}", T::TYPE));
    for (route, name) in [
        "source_prepared",
        "ordinary_pcu",
        "lowered_graph",
        "native_byte_kernel",
    ]
    .into_iter()
    .enumerate()
    {
        for bank in 0..3 {
            execute(route, bank, true);
        }
        execute(route, 0, true);
        #[cfg(feature = "allocation-census")]
        census::census(&format!("{:?}/{profile}/{name}/{N}", T::TYPE), || {
            execute(route, 1, false);
        });
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
fn cases<T: Sample, const N: usize>(criterion: &mut Criterion) {
    let session = MetalSession::open(0).unwrap();
    macro_rules! dense {
        ($entry:ident,$prepare:ident,$ir:ident,$bindings:ident) => {{
            let mut prepared = source::$prepare::<T, N, _>(&session).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let map = session.prepare_carrier_kernel(&builder.ir()).unwrap();
            compare::<T, N>(
                criterion,
                &session,
                stringify!($entry),
                false,
                &map,
                move |input, output| prepared(input, output),
                |input, output| source::$entry::<T, N>(input, output),
            );
        }};
    }
    dense!(direct, direct_prepare, direct_ir, direct_bindings);
    dense!(grid, grid_prepare, grid_ir, grid_bindings);
    macro_rules! broadcast {
        ($entry:ident,$prepare:ident,$ir:ident,$bindings:ident) => {{
            let mut prepared = source::$prepare::<T, N, _>(&session).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let map = session.prepare_carrier_kernel(&builder.ir()).unwrap();
            compare::<T, N>(
                criterion,
                &session,
                stringify!($entry),
                true,
                &map,
                move |input, output| prepared(&input[0], output),
                |input, output| source::$entry::<T, N>(&input[0], output),
            );
        }};
    }
    broadcast!(
        broadcast,
        broadcast_prepare,
        broadcast_ir,
        broadcast_bindings
    );
    broadcast!(
        grid_broadcast,
        grid_broadcast_prepare,
        grid_broadcast_ir,
        grid_broadcast_bindings
    );
}
pub fn run(criterion: &mut Criterion) {
    activity::guard();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Metal,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    macro_rules! widths{($($ty:ty),+)=>{$(cases::<$ty,257>(criterion);cases::<$ty,65537>(criterion);)+};}
    widths!(
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
        u128,
        i128,
        PcuU256,
        PcuI256,
        PcuU512,
        PcuI512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
