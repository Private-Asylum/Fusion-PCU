//! Matched genuine-source, explicit graph and independently compiled native checked unary peers.
#[path = "../checked_unary/ffi/ffi.rs"]
#[allow(dead_code)] // Private diagnostics are exercised by the independent encoding test.
mod ffi;
#[path = "../../../cpu/tests/low_unary/graph/graph.rs"]
mod graph;
#[path = "../../tests/checked_unary/oracle/oracle.rs"]
#[allow(dead_code)]
mod oracle;
#[path = "../../../cpu/tests/portable_unary/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
use oracle::Float;
#[rustfmt::skip]
use pcu_facade::{
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
use std::hint::black_box;
use std::sync::OnceLock;
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuRangePolicy,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuDispatchFloatUnaryOp,
    PcuFloatUnderflowPolicy,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscovery,
    PcuStableDeviceIdentity,
    PcuTargetDescriptor,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanDiscovery,
};

static SELECTED_IDENTITY: OnceLock<PcuStableDeviceIdentity> = OnceLock::new();
static COLD_SCORES: AtomicUsize = AtomicUsize::new(0);

fn matched_device_score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    assert_eq!(
        candidate.facts.stable_identity.as_ref(),
        SELECTED_IDENTITY.get()
    );
    COLD_SCORES.fetch_add(1, Ordering::Relaxed);
    0
}

fn selected_backend() -> PcuVulkanBackend {
    let discovery = PcuVulkanDiscovery::discover().expect("physical Vulkan discovery");
    let reference = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Device,
        id: 0,
    };
    let readiness = PcuProviderReadiness {
        status: PcuProviderStatus::Unavailable,
        reason: None,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness,
    }];
    discovery.providers(&mut providers).unwrap();
    let mut targets = [PcuTargetDescriptor {
        reference,
        name: "",
        readiness,
    }];
    discovery
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .unwrap();
    let mut devices = [PcuDeviceDescriptor {
        reference,
        target: reference,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    }];
    assert!(
        discovery
            .devices(targets[0].reference, &mut devices)
            .unwrap()
            > 0,
        "a physical GPU is required"
    );
    let device = devices[0].reference;
    SELECTED_IDENTITY
        .set(
            discovery
                .device_facts(device)
                .unwrap()
                .stable_identity
                .expect("physical device UUID"),
        )
        .unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        device: Some(device.id),
        score_invocation: Some(matched_device_score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    PcuVulkanBackend::open(&discovery, device).expect("selected physical Vulkan GPU")
}

fn require_gpu_idle() {
    if std::env::args().any(|argument| argument == "--test") {
        println!("Criterion semantic smoke only: no statistical timing samples");
        return;
    }
    let path = std::env::var_os("PCU_VULKAN_GPU_BUSY_PATH").map_or_else(
        || std::path::PathBuf::from("/sys/class/drm/card1/device/gpu_busy_percent"),
        std::path::PathBuf::from,
    );
    let mut idle = 0;
    for _ in 0..200 {
        let busy: u32 = std::fs::read_to_string(&path)
            .expect("GPU idle guard requires a device activity counter")
            .trim()
            .parse()
            .expect("GPU activity counter is an integer percent");
        idle = if busy <= 5 { idle + 1 } else { 0 };
        if idle == 3 {
            println!(
                "GPU idle guard admitted <=5% activity at {}",
                path.display()
            );
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!(
        "GPU idle guard did not reach three <=5% readings in ten seconds at {}",
        path.display()
    );
}

fn fault(error: fusion_pcu_vulkan::PcuVulkanError) -> PcuExecutionFault {
    match error {
        fusion_pcu_vulkan::PcuVulkanError::Fault(f) => f,
        other => panic!("unexpected {other:?}"),
    }
}
fn ordinary_fault(error: global::PcuExecutionError) -> PcuExecutionFault {
    match error {
        global::PcuExecutionError::ArithmeticFault(f) => f,
        other => panic!("unexpected {other:?}"),
    }
}
macro_rules! workload {($run:ident,$entry:ident,$prepare:ident,$op:ident,$op_code:literal,$uf:ident,$uf_code:literal,$clamp:literal)=>{
#[allow(clippy::too_many_lines,clippy::significant_drop_tightening)] // Four matched actual boundaries remain adjacent; finish consumes the Criterion group.
fn $run<T:Float,const N:usize>(criterion:&mut Criterion,backend:&PcuVulkanBackend){
 let mut input=vec![T::from_raw(1);N];let sentinel=T::from_raw(17);let mut output=vec![sentinel;N+3];let range=if $clamp{PcuRangePolicy::Clamp}else{PcuRangePolicy::Reject};let policy=PcuFloatUnderflowPolicy::$uf;
 let mut prepared=source::$prepare::<T,N,_>(backend).unwrap();let graph=graph::Graph::new(T::TYPE,u32::try_from(N).unwrap(),PcuDispatchFloatUnaryOp::$op,policy,range);let mut explicit=graph.with(|kernel|{let mut kernel=*kernel;kernel.numerical_requirements.numerical_options.reproducibility=pcu_facade::PcuReproducibility::PortableV1;if policy==PcuFloatUnderflowPolicy::RejectSubnormalResult{kernel.numerical_requirements.numerical_mode=pcu_facade::PcuNumericalMode::Strict;}backend.prepare_host_kernel(&kernel)}).unwrap();let mut native=ffi::NativeUnary::new(SELECTED_IDENTITY.get().unwrap().clone(),u32::try_from(N).unwrap(),T::FORMAT,$op_code,$uf_code).unwrap();
 let underflow=policy==PcuFloatUnderflowPolicy::RejectSubnormalResult;let notice=PcuExecutionFault{recovered:$clamp,invocation_id:0,kind:PcuExecutionFaultKind::ArithmeticUnderflow};let expected_result=if underflow{Err(notice)}else{Ok(())};
 let mut run=|route:usize,input:&[T],output:&mut[T]|match route {0=>prepared(input,output).map_err(fault),1=>source::$entry::<T,N>(input,output).map_err(ordinary_fault),2=>explicit.call(&mut[PcuHostArgument::read(PcuBindingRef::new(0,0),input),PcuHostArgument::read_write(PcuBindingRef::new(0,1),output)]).map_err(fault),_=>native.call(ffi::bytes(input),ffi::bytes_mut(output),$clamp).unwrap().map_or(Ok(()),Err)};
 for route in 0..4{for raw in [1,2]{input[0]=T::from_raw(raw);output.fill(sentinel);assert_eq!(run(route,&input,&mut output),expected_result);let bits=if underflow&&!$clamp{17}else{oracle::evaluate::<T>(raw,PcuDispatchFloatUnaryOp::$op,policy).unwrap().0};assert_eq!(output[0].raw(),bits);assert!(output[N..].iter().all(|v|v.raw()==17));}}
 let warm_scores=COLD_SCORES.load(Ordering::Relaxed);let mut group=criterion.benchmark_group(format!("vulkan_portable_unary_{:?}_{:?}_{policy:?}_{range:?}",T::TYPE,PcuDispatchFloatUnaryOp::$op));group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
 for (route,label) in ["prepared_annotated_source","ordinary_annotated_source","explicit_graph_diagnostic","native_ash_u32_kernel"].into_iter().enumerate(){let counts=ffi::count_heap(||{for _ in 0..64{input[0]=T::from_raw(if input[0].raw()==1{2}else{1});assert_eq!(run(route,&input,&mut output),expected_result);}});assert_eq!((counts.allocations,counts.reallocations,counts.frees),(0,0,0));println!("census {label} {:?} {:?} {policy:?} {range:?} N{N} calls64 allocations0 reallocations0 frees0",T::TYPE,PcuDispatchFloatUnaryOp::$op);group.bench_function(BenchmarkId::new(label,N),|bench|bench.iter(||{input[0]=T::from_raw(if input[0].raw()==1{2}else{1});assert_eq!(run(route,black_box(&input),black_box(&mut output)),expected_result);black_box(&output);}));}group.finish();assert_eq!(COLD_SCORES.load(Ordering::Relaxed),warm_scores);
}
};}
workload!(
    neg_reject_ieee,
    neg_reject_ieee,
    neg_reject_ieee_prepare,
    Neg,
    0,
    IeeeAfterRounding,
    0,
    false
);
workload!(
    neg_clamp_ieee,
    neg_clamp_ieee,
    neg_clamp_ieee_prepare,
    Neg,
    0,
    IeeeAfterRounding,
    0,
    true
);
workload!(
    neg_reject_gradual,
    neg_reject_gradual,
    neg_reject_gradual_prepare,
    Neg,
    0,
    AllowGradualUnderflow,
    2,
    false
);
workload!(
    neg_clamp_gradual,
    neg_clamp_gradual,
    neg_clamp_gradual_prepare,
    Neg,
    0,
    AllowGradualUnderflow,
    2,
    true
);
workload!(
    neg_reject_strict,
    neg_reject_strict,
    neg_reject_strict_prepare,
    Neg,
    0,
    RejectSubnormalResult,
    1,
    false
);
workload!(
    neg_clamp_strict,
    neg_clamp_strict,
    neg_clamp_strict_prepare,
    Neg,
    0,
    RejectSubnormalResult,
    1,
    true
);
workload!(
    relu_reject_ieee,
    relu_reject_ieee,
    relu_reject_ieee_prepare,
    Relu,
    1,
    IeeeAfterRounding,
    0,
    false
);
workload!(
    relu_clamp_ieee,
    relu_clamp_ieee,
    relu_clamp_ieee_prepare,
    Relu,
    1,
    IeeeAfterRounding,
    0,
    true
);
workload!(
    relu_reject_gradual,
    relu_reject_gradual,
    relu_reject_gradual_prepare,
    Relu,
    1,
    AllowGradualUnderflow,
    2,
    false
);
workload!(
    relu_clamp_gradual,
    relu_clamp_gradual,
    relu_clamp_gradual_prepare,
    Relu,
    1,
    AllowGradualUnderflow,
    2,
    true
);
workload!(
    relu_reject_strict,
    relu_reject_strict,
    relu_reject_strict_prepare,
    Relu,
    1,
    RejectSubnormalResult,
    1,
    false
);
workload!(
    relu_clamp_strict,
    relu_clamp_strict,
    relu_clamp_strict_prepare,
    Relu,
    1,
    RejectSubnormalResult,
    1,
    true
);
#[allow(clippy::too_many_lines)] // Explicit twelve-profile calls prevent accidental admission or peer omissions.
fn benchmark(criterion: &mut Criterion) {
    require_gpu_idle();
    let backend = selected_backend();
    neg_reject_ieee::<PcuF16Bits, 1>(criterion, &backend);
    neg_reject_ieee::<PcuF16Bits, 65>(criterion, &backend);
    neg_clamp_ieee::<PcuF16Bits, 1>(criterion, &backend);
    neg_clamp_ieee::<PcuF16Bits, 65>(criterion, &backend);
    neg_reject_gradual::<PcuF16Bits, 1>(criterion, &backend);
    neg_reject_gradual::<PcuF16Bits, 65>(criterion, &backend);
    neg_clamp_gradual::<PcuF16Bits, 1>(criterion, &backend);
    neg_clamp_gradual::<PcuF16Bits, 65>(criterion, &backend);
    neg_reject_strict::<PcuF16Bits, 1>(criterion, &backend);
    neg_reject_strict::<PcuF16Bits, 65>(criterion, &backend);
    neg_clamp_strict::<PcuF16Bits, 1>(criterion, &backend);
    neg_clamp_strict::<PcuF16Bits, 65>(criterion, &backend);
    relu_reject_ieee::<PcuF16Bits, 1>(criterion, &backend);
    relu_reject_ieee::<PcuF16Bits, 65>(criterion, &backend);
    relu_clamp_ieee::<PcuF16Bits, 1>(criterion, &backend);
    relu_clamp_ieee::<PcuF16Bits, 65>(criterion, &backend);
    relu_reject_gradual::<PcuF16Bits, 1>(criterion, &backend);
    relu_reject_gradual::<PcuF16Bits, 65>(criterion, &backend);
    relu_clamp_gradual::<PcuF16Bits, 1>(criterion, &backend);
    relu_clamp_gradual::<PcuF16Bits, 65>(criterion, &backend);
    relu_reject_strict::<PcuF16Bits, 1>(criterion, &backend);
    relu_reject_strict::<PcuF16Bits, 65>(criterion, &backend);
    relu_clamp_strict::<PcuF16Bits, 1>(criterion, &backend);
    relu_clamp_strict::<PcuF16Bits, 65>(criterion, &backend);
    neg_reject_ieee::<PcuBf16Bits, 1>(criterion, &backend);
    neg_reject_ieee::<PcuBf16Bits, 65>(criterion, &backend);
    neg_clamp_ieee::<PcuBf16Bits, 1>(criterion, &backend);
    neg_clamp_ieee::<PcuBf16Bits, 65>(criterion, &backend);
    neg_reject_gradual::<PcuBf16Bits, 1>(criterion, &backend);
    neg_reject_gradual::<PcuBf16Bits, 65>(criterion, &backend);
    neg_clamp_gradual::<PcuBf16Bits, 1>(criterion, &backend);
    neg_clamp_gradual::<PcuBf16Bits, 65>(criterion, &backend);
    neg_reject_strict::<PcuBf16Bits, 1>(criterion, &backend);
    neg_reject_strict::<PcuBf16Bits, 65>(criterion, &backend);
    neg_clamp_strict::<PcuBf16Bits, 1>(criterion, &backend);
    neg_clamp_strict::<PcuBf16Bits, 65>(criterion, &backend);
    relu_reject_ieee::<PcuBf16Bits, 1>(criterion, &backend);
    relu_reject_ieee::<PcuBf16Bits, 65>(criterion, &backend);
    relu_clamp_ieee::<PcuBf16Bits, 1>(criterion, &backend);
    relu_clamp_ieee::<PcuBf16Bits, 65>(criterion, &backend);
    relu_reject_gradual::<PcuBf16Bits, 1>(criterion, &backend);
    relu_reject_gradual::<PcuBf16Bits, 65>(criterion, &backend);
    relu_clamp_gradual::<PcuBf16Bits, 1>(criterion, &backend);
    relu_clamp_gradual::<PcuBf16Bits, 65>(criterion, &backend);
    relu_reject_strict::<PcuBf16Bits, 1>(criterion, &backend);
    relu_reject_strict::<PcuBf16Bits, 65>(criterion, &backend);
    relu_clamp_strict::<PcuBf16Bits, 1>(criterion, &backend);
    relu_clamp_strict::<PcuBf16Bits, 65>(criterion, &backend);
    neg_reject_ieee::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    neg_reject_ieee::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    neg_clamp_ieee::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    neg_clamp_ieee::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    neg_reject_gradual::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    neg_reject_gradual::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    neg_clamp_gradual::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    neg_clamp_gradual::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    neg_reject_strict::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    neg_reject_strict::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    neg_clamp_strict::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    neg_clamp_strict::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    relu_reject_ieee::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    relu_reject_ieee::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    relu_clamp_ieee::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    relu_clamp_ieee::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    relu_reject_gradual::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    relu_reject_gradual::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    relu_clamp_gradual::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    relu_clamp_gradual::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    relu_reject_strict::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    relu_reject_strict::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    relu_clamp_strict::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    relu_clamp_strict::<PcuF8E4M3FnBits, 65>(criterion, &backend);
    neg_reject_ieee::<PcuF8E5M2Bits, 1>(criterion, &backend);
    neg_reject_ieee::<PcuF8E5M2Bits, 65>(criterion, &backend);
    neg_clamp_ieee::<PcuF8E5M2Bits, 1>(criterion, &backend);
    neg_clamp_ieee::<PcuF8E5M2Bits, 65>(criterion, &backend);
    neg_reject_gradual::<PcuF8E5M2Bits, 1>(criterion, &backend);
    neg_reject_gradual::<PcuF8E5M2Bits, 65>(criterion, &backend);
    neg_clamp_gradual::<PcuF8E5M2Bits, 1>(criterion, &backend);
    neg_clamp_gradual::<PcuF8E5M2Bits, 65>(criterion, &backend);
    neg_reject_strict::<PcuF8E5M2Bits, 1>(criterion, &backend);
    neg_reject_strict::<PcuF8E5M2Bits, 65>(criterion, &backend);
    neg_clamp_strict::<PcuF8E5M2Bits, 1>(criterion, &backend);
    neg_clamp_strict::<PcuF8E5M2Bits, 65>(criterion, &backend);
    relu_reject_ieee::<PcuF8E5M2Bits, 1>(criterion, &backend);
    relu_reject_ieee::<PcuF8E5M2Bits, 65>(criterion, &backend);
    relu_clamp_ieee::<PcuF8E5M2Bits, 1>(criterion, &backend);
    relu_clamp_ieee::<PcuF8E5M2Bits, 65>(criterion, &backend);
    relu_reject_gradual::<PcuF8E5M2Bits, 1>(criterion, &backend);
    relu_reject_gradual::<PcuF8E5M2Bits, 65>(criterion, &backend);
    relu_clamp_gradual::<PcuF8E5M2Bits, 1>(criterion, &backend);
    relu_clamp_gradual::<PcuF8E5M2Bits, 65>(criterion, &backend);
    relu_reject_strict::<PcuF8E5M2Bits, 1>(criterion, &backend);
    relu_reject_strict::<PcuF8E5M2Bits, 65>(criterion, &backend);
    relu_clamp_strict::<PcuF8E5M2Bits, 1>(criterion, &backend);
    relu_clamp_strict::<PcuF8E5M2Bits, 65>(criterion, &backend);
    neg_reject_ieee::<f32, 1>(criterion, &backend);
    neg_reject_ieee::<f32, 65>(criterion, &backend);
    neg_clamp_ieee::<f32, 1>(criterion, &backend);
    neg_clamp_ieee::<f32, 65>(criterion, &backend);
    neg_reject_gradual::<f32, 1>(criterion, &backend);
    neg_reject_gradual::<f32, 65>(criterion, &backend);
    neg_clamp_gradual::<f32, 1>(criterion, &backend);
    neg_clamp_gradual::<f32, 65>(criterion, &backend);
    neg_reject_strict::<f32, 1>(criterion, &backend);
    neg_reject_strict::<f32, 65>(criterion, &backend);
    neg_clamp_strict::<f32, 1>(criterion, &backend);
    neg_clamp_strict::<f32, 65>(criterion, &backend);
    relu_reject_ieee::<f32, 1>(criterion, &backend);
    relu_reject_ieee::<f32, 65>(criterion, &backend);
    relu_clamp_ieee::<f32, 1>(criterion, &backend);
    relu_clamp_ieee::<f32, 65>(criterion, &backend);
    relu_reject_gradual::<f32, 1>(criterion, &backend);
    relu_reject_gradual::<f32, 65>(criterion, &backend);
    relu_clamp_gradual::<f32, 1>(criterion, &backend);
    relu_clamp_gradual::<f32, 65>(criterion, &backend);
    relu_reject_strict::<f32, 1>(criterion, &backend);
    relu_reject_strict::<f32, 65>(criterion, &backend);
    relu_clamp_strict::<f32, 1>(criterion, &backend);
    relu_clamp_strict::<f32, 65>(criterion, &backend);
    neg_reject_ieee::<f64, 1>(criterion, &backend);
    neg_reject_ieee::<f64, 65>(criterion, &backend);
    neg_clamp_ieee::<f64, 1>(criterion, &backend);
    neg_clamp_ieee::<f64, 65>(criterion, &backend);
    neg_reject_gradual::<f64, 1>(criterion, &backend);
    neg_reject_gradual::<f64, 65>(criterion, &backend);
    neg_clamp_gradual::<f64, 1>(criterion, &backend);
    neg_clamp_gradual::<f64, 65>(criterion, &backend);
    neg_reject_strict::<f64, 1>(criterion, &backend);
    neg_reject_strict::<f64, 65>(criterion, &backend);
    neg_clamp_strict::<f64, 1>(criterion, &backend);
    neg_clamp_strict::<f64, 65>(criterion, &backend);
    relu_reject_ieee::<f64, 1>(criterion, &backend);
    relu_reject_ieee::<f64, 65>(criterion, &backend);
    relu_clamp_ieee::<f64, 1>(criterion, &backend);
    relu_clamp_ieee::<f64, 65>(criterion, &backend);
    relu_reject_gradual::<f64, 1>(criterion, &backend);
    relu_reject_gradual::<f64, 65>(criterion, &backend);
    relu_clamp_gradual::<f64, 1>(criterion, &backend);
    relu_clamp_gradual::<f64, 65>(criterion, &backend);
    relu_reject_strict::<f64, 1>(criterion, &backend);
    relu_reject_strict::<f64, 65>(criterion, &backend);
    relu_clamp_strict::<f64, 1>(criterion, &backend);
    relu_clamp_strict::<f64, 65>(criterion, &backend);
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
