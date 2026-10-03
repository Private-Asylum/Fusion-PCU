//! Matched host-publication boundary: actual annotated source, neutral graph, native checker.
//! Cold compile is outside samples. Each call allocates one shared input, fresh output and
//! fresh private status, resets every record, waits for terminal completion, inspects all status
//! records in place and publishes the successful output prefix. No resident or native-ALU claim is made.

#[rustfmt::skip]
#[cfg(not(feature = "allocation-census"))]
use std::hint::black_box;
#[rustfmt::skip]
use std::{
    time::Duration,
};
#[rustfmt::skip]
#[cfg(not(feature = "allocation-census"))]
use criterion::BenchmarkId;
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuKernelId,
    PcuRangePolicy,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalBuffer,
    MetalError,
    MetalPreparedF32Kernel,
    MetalPreparedF32Unary,
    MetalSession,
};

#[path = "source.rs"]
mod source;

#[cfg(feature = "allocation-census")]
#[path = "census/census.rs"]
mod census;

trait DeviceMap {
    fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError>;
}
impl DeviceMap for MetalPreparedF32Kernel {
    fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        Self::execute(self, input)
    }
}
impl DeviceMap for MetalPreparedF32Unary {
    fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        Self::execute(self, input)
    }
}
fn host_call(
    session: &MetalSession,
    executable: &impl DeviceMap,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), MetalError> {
    if output.len() < input.len() {
        return Err(MetalError::InvalidExtent);
    }
    let length = input.len();
    let input = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
    let input = session.upload_bytes(input.bytes())?;
    let result = executable.execute(&input)?;
    let mut destination =
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output[..length]);
    result.read_into_bytes(destination.bytes_mut().expect("mutable host destination"))
}
fn graph<const N: usize>(session: &MetalSession) -> MetalPreparedF32Kernel {
    let bindings = [
        PcuBinding::scalar::<f32>(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<f32>(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
    ];
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: PcuValueType::f32(),
            op: PcuDispatchFloatUnaryOp::Neg,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    session
        .prepare_f32_unary_kernel(&PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "checked_neg",
                logical_shape: [u32::try_from(N).unwrap(), 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::FLOAT32,
            feature_caps: PcuDispatchFeatureCaps::empty(),
        })
        .unwrap()
}
trait CheckedFault {
    fn checked_fault(self) -> Option<pcu_facade::PcuExecutionFault>;
}
impl CheckedFault for PcuHostDispatchError<MetalError> {
    fn checked_fault(self) -> Option<pcu_facade::PcuExecutionFault> {
        if let Self::Backend(MetalError::Arithmetic(fault)) = self {
            Some(fault)
        } else {
            None
        }
    }
}
impl CheckedFault for pcu_facade::PcuExecutionError {
    fn checked_fault(self) -> Option<pcu_facade::PcuExecutionFault> {
        if let Self::ArithmeticFault(fault) = self {
            Some(fault)
        } else {
            None
        }
    }
}
fn validate<const N: usize, E: std::fmt::Debug + CheckedFault>(
    session: &MetalSession,
    source: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), E>,
    graph: &MetalPreparedF32Kernel,
    native: &MetalPreparedF32Unary,
) {
    let mut input = vec![1.0_f32; N];
    let special = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0xbf80_0000,
    ];
    for (value, bits) in input.iter_mut().zip(special) {
        *value = f32::from_bits(bits);
    }
    let expected: Vec<u32> = input
        .iter()
        .map(|value| value.to_bits() ^ 0x8000_0000)
        .chain([91.0_f32.to_bits()])
        .collect();
    let mut output = vec![91.0; N + 1];
    source(&input, &mut output).unwrap();
    assert_eq!(
        output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    host_call(session, graph, &input, &mut output).unwrap();
    assert_eq!(
        output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    host_call(session, native, &input, &mut output).unwrap();
    assert_eq!(
        output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    for bits in [0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0x7f80_0001] {
        input[2] = f32::from_bits(bits);
        input[N - 1] = f32::INFINITY;
        let source_error = source(&input, &mut output).unwrap_err();
        let source_fault = source_error
            .checked_fault()
            .expect("unexpected source failure");
        for result in [
            host_call(session, graph, &input, &mut output),
            host_call(session, native, &input, &mut output),
        ] {
            let Err(MetalError::Arithmetic(fault)) = result else {
                panic!("missing native checked fault");
            };
            assert_eq!(fault, source_fault);
            assert_eq!(fault.invocation_id, 2);
        }
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
    input.fill(2.0);
    source(&input, &mut output).unwrap();
    host_call(session, graph, &input, &mut output).unwrap();
    host_call(session, native, &input, &mut output).unwrap();
    assert!(
        output[..N]
            .iter()
            .all(|value| value.to_bits() == (-2.0_f32).to_bits())
    );
}
#[cfg_attr(
    feature = "allocation-census",
    allow(
        clippy::needless_pass_by_ref_mut,
        reason = "Census keeps the primary Criterion callback signature without timing."
    )
)]
fn cases<const N: usize>(criterion: &mut Criterion) {
    let session = MetalSession::open(0).unwrap();
    println!(
        "Metal canonical boundary: {:?}; N={N}; fresh shared input/output/status, terminal wait, full status inspection + host selection, host output prefix",
        session.facts()
    );
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy {
        backend: pcu_facade::global::PcuBackendChoice::Metal,
        ..pcu_facade::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let mut source = source::negate_prepare::<N, _>(&session).unwrap();
    let graph = graph::<N>(&session);
    let native = session
        .prepare_f32_neg(PcuFloatUnderflowPolicy::IeeeAfterRounding)
        .unwrap();
    validate::<N, _>(&session, &mut source, &graph, &native);
    let mut global = |input: &[f32], output: &mut [f32]| source::negate::<N>(input, output);
    // The global source route has the same physical buffers/status/publication as its peers.
    // Preflight also warms its immutable thread-local specialization before Criterion timing.
    validate::<N, _>(&session, &mut global, &graph, &native);
    let mut input = vec![1.0_f32; N];
    let mut output = vec![91.0_f32; N + 1];
    #[cfg(feature = "allocation-census")]
    {
        let _ = criterion;
        input[0] = f32::from_bits(input[0].to_bits() ^ 1);
        census::census(&format!("source/{N}"), || {
            source(&input, &mut output).unwrap();
        });
        input[0] = f32::from_bits(input[0].to_bits() ^ 1);
        census::census(&format!("global_source/{N}"), || {
            global(&input, &mut output).unwrap();
        });
        input[0] = f32::from_bits(input[0].to_bits() ^ 1);
        census::census(&format!("graph/{N}"), || {
            host_call(&session, &graph, &input, &mut output).unwrap();
        });
        input[0] = f32::from_bits(input[0].to_bits() ^ 1);
        census::census(&format!("native_checker/{N}"), || {
            host_call(&session, &native, &input, &mut output).unwrap();
        });
    }
    #[cfg(not(feature = "allocation-census"))]
    {
        let mut group = criterion.benchmark_group("host_f32_neg");
        group.bench_function(BenchmarkId::new("source", N), |bench| {
            bench.iter(|| {
                input[0] = f32::from_bits(input[0].to_bits() ^ 1);
                source(black_box(&input), black_box(&mut output)).unwrap();
                black_box(&output);
            });
        });
        group.bench_function(BenchmarkId::new("global_source", N), |bench| {
            bench.iter(|| {
                input[0] = f32::from_bits(input[0].to_bits() ^ 1);
                global(black_box(&input), black_box(&mut output)).unwrap();
                black_box(&output);
            });
        });
        group.bench_function(BenchmarkId::new("graph", N), |bench| {
            bench.iter(|| {
                input[0] = f32::from_bits(input[0].to_bits() ^ 1);
                host_call(&session, &graph, black_box(&input), black_box(&mut output)).unwrap();
                black_box(&output);
            });
        });
        group.bench_function(BenchmarkId::new("native_checker", N), |bench| {
            bench.iter(|| {
                input[0] = f32::from_bits(input[0].to_bits() ^ 1);
                host_call(&session, &native, black_box(&input), black_box(&mut output)).unwrap();
                black_box(&output);
            });
        });
        group.finish();
    }
}
fn checked_neg(criterion: &mut Criterion) {
    if !cfg!(target_os = "macos") {
        println!("SKIP: Metal hardware benchmark requires macOS; no fallback timing");
        return;
    }
    let inventory = std::process::Command::new("/usr/sbin/ioreg")
        .args(["-r", "-c", "AGXAccelerator", "-l"])
        .output()
        .expect("GPU activity guard could not read ioreg");
    assert!(
        inventory.status.success(),
        "GPU activity guard command failed"
    );
    let inventory = String::from_utf8(inventory.stdout).expect("GPU activity is not UTF-8");
    let activity = inventory
        .split("\"Device Utilization %\"=")
        .nth(1)
        .expect("GPU activity counter unavailable");
    let activity: u32 = activity
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .expect("GPU activity counter malformed");
    println!("Metal activity guard before benchmark: {activity}%");
    assert!(
        activity <= 5,
        "GPU is busy; refusing contaminated benchmark"
    );
    cases::<257>(criterion);
    cases::<65_536>(criterion);
}
criterion_group! {
    name = benches;
    config = Criterion::default().confidence_level(0.95).sample_size(30).warm_up_time(Duration::from_millis(250)).measurement_time(Duration::from_secs(1));
    targets = checked_neg
}
criterion_main!(benches);
