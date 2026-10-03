//! Matched genuine-source, graph and independently owned native packed arithmetic peers.
#[path = "../../tests/clamped_binary/source/source.rs"]
#[allow(dead_code)]
mod clamp;
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../../spirv/tests/checked_binary/support/support.rs"]
mod graph;
#[path = "../../../cpu/tests/low_precision/oracle/oracle.rs"]
#[allow(dead_code)]
mod oracle;
#[path = "../../../cpu/tests/low_precision/source/source.rs"]
#[allow(dead_code)]
mod reject;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
use oracle::Low;
#[rustfmt::skip]
use pcu_facade::{PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuScalarType};
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
    PcuDispatchFloatBinaryOp,
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

fn format<T: Low>() -> u32 {
    match T::TYPE {
        PcuScalarType::F16 => 0,
        PcuScalarType::BF16 => 1,
        PcuScalarType::F8E4M3FN => 2,
        PcuScalarType::F8E5M2 => 3,
        _ => unreachable!("closed low benchmark types"),
    }
}
fn fault(error: fusion_pcu_vulkan::PcuVulkanError) -> PcuExecutionFault {
    match error {
        fusion_pcu_vulkan::PcuVulkanError::Fault(fault) => fault,
        other => panic!("unexpected {other:?}"),
    }
}
fn ordinary_fault(error: global::PcuExecutionError) -> PcuExecutionFault {
    match error {
        global::PcuExecutionError::ArithmeticFault(fault) => fault,
        other => panic!("unexpected {other:?}"),
    }
}
macro_rules! workload {($run:ident,$family:ident,$entry:ident,$prepare:ident,$op:ident,$code:literal,$clamped:literal)=> {
#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // Four actual boundaries stay together; Criterion finish consumes the group before score assertions.
fn $run<T:Low,const N:usize>(criterion:&mut Criterion,backend:&PcuVulkanBackend) {
 let f=T::FORMAT;let one=u16::try_from(f.bias).unwrap()<<f.fraction;let two=u16::try_from(f.bias+1).unwrap()<<f.fraction;let half=u16::try_from(f.bias-1).unwrap()<<f.fraction;
 let initial=if $clamped {f.max}else{two};let rhs=if $clamped {match $code {0=>f.max,1=>f.sign|f.max,2=>two,_=>half}}else{one};let mut left=vec![T::from_bits(initial);N];let right=vec![T::from_bits(rhs);N];let sentinel=T::from_bits(17);let mut output=vec![sentinel;N+3];
 let range=if $clamped {PcuRangePolicy::Clamp}else{PcuRangePolicy::Reject};let mut source=$family::$prepare::<T,N,_>(backend).unwrap();let mut g=graph::Graph::new(u32::try_from(N).unwrap(),PcuDispatchFloatBinaryOp::$op,PcuFloatUnderflowPolicy::default());g.scalar=T::TYPE;g.range=range;let mut explicit=g.with(|kernel|backend.prepare_host_kernel(kernel)).unwrap();let mut native=ffi::NativeBinary::new(SELECTED_IDENTITY.get().unwrap().clone(),u32::try_from(N).unwrap(),$code,0,format::<T>()).unwrap();
 let notice=PcuExecutionFault{recovered:true,invocation_id:0,kind:PcuExecutionFaultKind::ArithmeticOverflow};let expected_result=if $clamped {Err(notice)}else{Ok(())};
 let expected=core::array::from_fn::<_,2,_>(|i|f.evaluate_clamped(if $clamped {initial-u16::try_from(i).unwrap()}else{initial^u16::try_from(i).unwrap()},rhs,PcuDispatchFloatBinaryOp::$op,PcuFloatUnderflowPolicy::default()).unwrap().0);
 let mut run=|route:usize,left:&[T],output:&mut[T]|match route {0=>source(left,&right,output).map_err(fault),1=>$family::$entry::<T,N>(left,&right,output).map_err(ordinary_fault),2=>explicit.call(&mut[PcuHostArgument::read(PcuBindingRef::new(0,0),left),PcuHostArgument::read(PcuBindingRef::new(0,1),&right),PcuHostArgument::read_write(PcuBindingRef::new(0,2),output)]).map_err(fault),_=>if $clamped {native.call_clamped(ffi::bytes(left),ffi::bytes(&right),ffi::bytes_mut(output)).unwrap().map_or(Ok(()),Err)}else{native.call(ffi::bytes(left),ffi::bytes(&right),ffi::bytes_mut(output),None).unwrap();Ok(())}};
 for route in 0..4 {for changed in 0..2 {left[0]=T::from_bits(if $clamped {initial-u16::try_from(changed).unwrap()}else{initial^u16::try_from(changed).unwrap()});assert_eq!(run(route,&left,&mut output),expected_result);assert_eq!(output[0].bits(),expected[changed]);assert!(output[N..].iter().all(|v|*v==sentinel));}}
 let warm_scores=COLD_SCORES.load(Ordering::Relaxed);
 let mut group=criterion.benchmark_group(format!("vulkan_low_{:?}_{:?}_{:?}",T::TYPE,PcuDispatchFloatBinaryOp::$op,range));group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
 for (route,label) in ["prepared_annotated_source","ordinary_annotated_source","explicit_graph_diagnostic","native_ash_packed_kernel"].into_iter().enumerate() {let counts=ffi::count_heap(||{for _ in 0..64 {left[0]=T::from_bits(if left[0].bits()==initial {if $clamped {initial-1}else{initial^1}}else{initial});assert_eq!(run(route,&left,&mut output),expected_result);}});assert_eq!((counts.allocations,counts.reallocations,counts.frees),(0,0,0));println!("census {label} {:?} {:?} {range:?} N{N} calls64 allocations0 reallocations0 frees0",T::TYPE,PcuDispatchFloatBinaryOp::$op);group.bench_function(BenchmarkId::new(label,N),|bench|bench.iter(||{left[0]=T::from_bits(if left[0].bits()==initial {if $clamped {initial-1}else{initial^1}}else{initial});assert_eq!(run(route,black_box(&left),black_box(&mut output)),expected_result);black_box(&output);}));}
 group.finish();assert_eq!(COLD_SCORES.load(Ordering::Relaxed),warm_scores);
}
};}
workload!(add_reject, reject, add, add_prepare, Add, 0, false);
workload!(add_clamp, clamp, add, add_prepare, Add, 0, true);
workload!(sub_reject, reject, sub, sub_prepare, Sub, 1, false);
workload!(sub_clamp, clamp, sub, sub_prepare, Sub, 1, true);
workload!(mul_reject, reject, mul, mul_prepare, Mul, 2, false);
workload!(mul_clamp, clamp, mul, mul_prepare, Mul, 2, true);
workload!(div_reject, reject, div, div_prepare, Div, 3, false);
workload!(div_clamp, clamp, div, div_prepare, Div, 3, true);
fn benchmark(criterion: &mut Criterion) {
    require_gpu_idle();
    let backend = selected_backend();
    add_reject::<PcuF16Bits, 1>(criterion, &backend);
    add_reject::<PcuF16Bits, 4096>(criterion, &backend);
    add_clamp::<PcuF16Bits, 1>(criterion, &backend);
    add_clamp::<PcuF16Bits, 4096>(criterion, &backend);
    sub_reject::<PcuF16Bits, 1>(criterion, &backend);
    sub_reject::<PcuF16Bits, 4096>(criterion, &backend);
    sub_clamp::<PcuF16Bits, 1>(criterion, &backend);
    sub_clamp::<PcuF16Bits, 4096>(criterion, &backend);
    mul_reject::<PcuF16Bits, 1>(criterion, &backend);
    mul_reject::<PcuF16Bits, 4096>(criterion, &backend);
    mul_clamp::<PcuF16Bits, 1>(criterion, &backend);
    mul_clamp::<PcuF16Bits, 4096>(criterion, &backend);
    div_reject::<PcuF16Bits, 1>(criterion, &backend);
    div_reject::<PcuF16Bits, 4096>(criterion, &backend);
    div_clamp::<PcuF16Bits, 1>(criterion, &backend);
    div_clamp::<PcuF16Bits, 4096>(criterion, &backend);
    add_reject::<PcuBf16Bits, 1>(criterion, &backend);
    add_reject::<PcuBf16Bits, 4096>(criterion, &backend);
    add_clamp::<PcuBf16Bits, 1>(criterion, &backend);
    add_clamp::<PcuBf16Bits, 4096>(criterion, &backend);
    sub_reject::<PcuBf16Bits, 1>(criterion, &backend);
    sub_reject::<PcuBf16Bits, 4096>(criterion, &backend);
    sub_clamp::<PcuBf16Bits, 1>(criterion, &backend);
    sub_clamp::<PcuBf16Bits, 4096>(criterion, &backend);
    mul_reject::<PcuBf16Bits, 1>(criterion, &backend);
    mul_reject::<PcuBf16Bits, 4096>(criterion, &backend);
    mul_clamp::<PcuBf16Bits, 1>(criterion, &backend);
    mul_clamp::<PcuBf16Bits, 4096>(criterion, &backend);
    div_reject::<PcuBf16Bits, 1>(criterion, &backend);
    div_reject::<PcuBf16Bits, 4096>(criterion, &backend);
    div_clamp::<PcuBf16Bits, 1>(criterion, &backend);
    div_clamp::<PcuBf16Bits, 4096>(criterion, &backend);
    add_reject::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    add_reject::<PcuF8E4M3FnBits, 4096>(criterion, &backend);
    add_clamp::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    add_clamp::<PcuF8E4M3FnBits, 4096>(criterion, &backend);
    sub_reject::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    sub_reject::<PcuF8E4M3FnBits, 4096>(criterion, &backend);
    sub_clamp::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    sub_clamp::<PcuF8E4M3FnBits, 4096>(criterion, &backend);
    mul_reject::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    mul_reject::<PcuF8E4M3FnBits, 4096>(criterion, &backend);
    mul_clamp::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    mul_clamp::<PcuF8E4M3FnBits, 4096>(criterion, &backend);
    div_reject::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    div_reject::<PcuF8E4M3FnBits, 4096>(criterion, &backend);
    div_clamp::<PcuF8E4M3FnBits, 1>(criterion, &backend);
    div_clamp::<PcuF8E4M3FnBits, 4096>(criterion, &backend);
    add_reject::<PcuF8E5M2Bits, 1>(criterion, &backend);
    add_reject::<PcuF8E5M2Bits, 4096>(criterion, &backend);
    add_clamp::<PcuF8E5M2Bits, 1>(criterion, &backend);
    add_clamp::<PcuF8E5M2Bits, 4096>(criterion, &backend);
    sub_reject::<PcuF8E5M2Bits, 1>(criterion, &backend);
    sub_reject::<PcuF8E5M2Bits, 4096>(criterion, &backend);
    sub_clamp::<PcuF8E5M2Bits, 1>(criterion, &backend);
    sub_clamp::<PcuF8E5M2Bits, 4096>(criterion, &backend);
    mul_reject::<PcuF8E5M2Bits, 1>(criterion, &backend);
    mul_reject::<PcuF8E5M2Bits, 4096>(criterion, &backend);
    mul_clamp::<PcuF8E5M2Bits, 1>(criterion, &backend);
    mul_clamp::<PcuF8E5M2Bits, 4096>(criterion, &backend);
    div_reject::<PcuF8E5M2Bits, 1>(criterion, &backend);
    div_reject::<PcuF8E5M2Bits, 4096>(criterion, &backend);
    div_clamp::<PcuF8E5M2Bits, 1>(criterion, &backend);
    div_clamp::<PcuF8E5M2Bits, 4096>(criterion, &backend);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
