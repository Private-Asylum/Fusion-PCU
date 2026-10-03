//! Independent representation samples prove faults, signed zero and private publication.
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanPreparedTensorGraph,
    PcuVulkanTensorInput,
    PcuVulkanTensorError,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::TensorError;
use super::{bits, device, profiles, sample::Sample, source};

trait Format: Sample {
    fn raw(bits: u64) -> Self;
    const MAX: u64;
    const INVALID: u64;
    const SIGN: u64;
}
macro_rules! formats {($($ty:ty,$max:literal,$invalid:literal,$sign:literal);+)=>{$(
    impl Format for $ty {
        fn raw(bits:u64)->Self { Self::from_bits(bits.try_into().unwrap()) }
        const MAX:u64=$max;
        const INVALID:u64=$invalid;
        const SIGN:u64=$sign;
    }
)+};}
formats!(PcuF16Bits,0x7bff,0x7c00,0x8000; PcuBf16Bits,0x7f7f,0x7f80,0x8000;
    PcuF8E4M3FnBits,0x7e,0x7f,0x80; PcuF8E5M2Bits,0x7b,0x7c,0x80;
    f32,0x7f7f_ffff,0x7f80_0000,0x8000_0000;
    f64,0x7fef_ffff_ffff_ffff,0x7ff0_0000_0000_0000,0x8000_0000_0000_0000);

fn fault<T: Format>(
    backend: &PcuVulkanBackend,
    op: u32,
    policy: PcuFloatUnderflowPolicy,
    left: &[T],
    right: &[T],
    lane: usize,
    kind: PcuExecutionFaultKind,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        float_underflow: policy,
        ..Default::default()
    })
    .unwrap();
    let (graph, output) = profiles::graph::<T>(left.len(), op, policy);
    let mut plan = PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &graph, &[output]).unwrap();
    let args = [
        PcuVulkanTensorInput::Host(left),
        PcuVulkanTensorInput::Host(right),
    ];
    assert!(matches!(plan.execute_owned(&args[..if op==4 {1}else{2}]),
        Err(PcuVulkanTensorError::Graph(TensorError::ArithmeticFault {
            value,element_index,kind:actual })) if value==output && element_index==lane && actual==kind));
    let error = source::call(op, left, right).err().unwrap();
    let actual = error.arithmetic_fault().unwrap();
    assert_eq!(
        (actual.invocation_id, actual.kind, actual.recovered),
        (u64::try_from(lane).unwrap(), kind, false)
    );
    let valid = [T::small(2); 65];
    let args = [
        PcuVulkanTensorInput::Host(&valid),
        PcuVulkanTensorInput::Host(&valid),
    ];
    plan.execute_owned(&args[..if op == 4 { 1 } else { 2 }])
        .unwrap();
    source::call(op, &valid, &valid).unwrap();
}

fn format<T: Format>(backend: &PcuVulkanBackend) {
    let mut left = [T::small(1); 65];
    let mut right = [T::small(2); 65];
    left[1] = T::raw(T::MAX);
    right[1] = T::small(2);
    left[3] = T::raw(T::INVALID);
    fault(
        backend,
        2,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        &left,
        &right,
        1,
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    left[1] = T::raw(T::INVALID);
    left[3] = T::raw(T::MAX);
    right[3] = T::small(2);
    fault(
        backend,
        2,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        &left,
        &right,
        1,
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    left = [T::small(1); 65];
    right = [T::small(2); 65];
    right[7] = T::raw(0);
    fault(
        backend,
        3,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        &left,
        &right,
        7,
        PcuExecutionFaultKind::DivideByZero,
    );
    left[0] = T::raw(1);
    right = [T::small(2); 65];
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        fault(
            backend,
            3,
            policy,
            &left,
            &right,
            0,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
    }
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        float_underflow: PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ..Default::default()
    })
    .unwrap();
    let owner = source::div(&left, &right).unwrap();
    let mut observed = [T::small(11); 68];
    owner.read_into(&mut observed).unwrap();
    bits::same(&observed[..1], &[T::raw(0)]);
    left = [T::raw(T::SIGN); 65];
    let owner = source::relu(&left).unwrap();
    owner.read_into(&mut observed).unwrap();
    bits::same(&observed[..65], &[T::raw(0); 65]);
    bits::same(&observed[65..], &[T::small(11); 3]);
    left[9] = T::raw(T::SIGN | T::INVALID);
    fault(
        backend,
        4,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        &left,
        &right,
        9,
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    // A checked unused node remains selected; a private failure cannot invalidate input/siblings.
    left = [T::small(1); 65];
    right = [T::small(2); 65];
    right[7] = T::raw(0);
    let retained = source::retain(&left).unwrap();
    let sibling = source::retain(&left).unwrap();
    let error = source::checked_unused(&retained, &right).err().unwrap();
    assert_eq!(
        error.arithmetic_fault().unwrap().kind,
        PcuExecutionFaultKind::DivideByZero
    );
    retained.read_into(&mut observed).unwrap();
    bits::same(&observed[..65], &left);
    sibling.read_into(&mut observed).unwrap();
    bits::same(&observed[..65], &left);
    right[7] = T::small(2);
    source::checked_unused(&retained, &right)
        .unwrap()
        .read_into(&mut observed)
        .unwrap();
    bits::same(&observed[..65], &left);
}
#[test]
#[ignore = "requires actual Vulkan GPU; six-format checked pointwise fault/owner laws"]
fn six_float_fault_ordering_signed_zero_underflow_and_unused_effects() {
    let (backend, _) = device::selected();
    format::<PcuF16Bits>(&backend);
    format::<PcuBf16Bits>(&backend);
    format::<PcuF8E4M3FnBits>(&backend);
    format::<PcuF8E5M2Bits>(&backend);
    format::<f32>(&backend);
    format::<f64>(&backend);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
