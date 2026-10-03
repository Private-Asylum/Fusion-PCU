//! Real source and prepared scalar borrows retain their exact native session and public bytes.
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../../../cpu/tests/checked_div_rem/source/source.rs"]
#[allow(dead_code)]
mod div_rem;
#[path = "../../../cpu/tests/scalar_tensor/source/source.rs"]
#[allow(dead_code)]
mod owned;
#[path = "../scalar_transport/sample/sample.rs"]
mod sample;
#[path = "../scalar_transport/source/source.rs"]
#[allow(dead_code)]
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    pcu,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuRangePolicy,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanArgument,
    PcuVulkanBackend,
    PcuVulkanError,
};
#[rustfmt::skip]
use sample::{
    same,
    Sample,
};

#[pcu(invocations=65, crate_path=::pcu_facade)]
fn add<T: PcuCheckedFloat>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}

#[pcu(invocations=65, crate_path=::pcu_facade)]
fn output_first<T: PcuCheckedFloat>(output: &mut [T], left: &[T], right: &[T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}
#[pcu(crate_path=::pcu_facade)]
fn matrix(input: &[[f32; 3]; 2]) -> Result<pcu_facade::PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

fn policy(range_policy: PcuRangePolicy) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        range_policy,
        ..Default::default()
    })
    .unwrap();
}

fn carrier<T: Sample>(backend: &PcuVulkanBackend, foreign: &PcuVulkanBackend) {
    const N: usize = 65;
    let input: Vec<T> = (0..N + 3).map(|i| T::pattern(i * 19 + 31)).collect();
    let sentinel = T::pattern(91);
    let bank = vec![sentinel; N + 3];
    let input_owner = owned::identity(&input).unwrap();
    let mut output_owner = owned::identity(&bank).unwrap();
    let sibling = owned::identity(&output_owner).unwrap();
    let mut output = bank.clone();
    source::dense::<T, N>(&input_owner, &mut output_owner).unwrap();
    output_owner.read_into(&mut output).unwrap();
    same(&output[..N], &input[..N]);
    same(&output[N..], &bank[N..]); // Odd byte prefixes deliberately never round into tails.
    sibling.read_into(&mut output).unwrap();
    same(&output, &bank);
    global::clear_thread_cache().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Automatic,
        ..Default::default()
    })
    .unwrap();
    source::dense_grid::<T, N>(&input_owner, &mut output_owner).unwrap();
    output_owner.read_into(&mut output).unwrap();
    same(&output[..N], &input[..N]);
    same(&output[N..], &bank[N..]);
    policy(PcuRangePolicy::Reject);
    // Explicit IR shares exact arithmetic/transport admission while borrowing its own actual root.
    let bindings = source::dense_bindings::<T>();
    let graph = source::dense_ir::<T, N>(&bindings).unwrap();
    let mut prepared = backend.prepare_mixed_kernel(&graph.ir()).unwrap();
    let native_input = backend.upload_owned(&input).unwrap();
    let mut native_output = backend.upload_owned(&bank).unwrap();
    prepared
        .call(&mut [
            native_output.write_argument(PcuBindingRef::new(0, 1)),
            native_input.read_argument(PcuBindingRef::new(0, 0)),
        ])
        .unwrap();
    assert!(prepared.last_call_may_have_written());
    assert!(!prepared.last_call_completion_uncertain());
    native_output.read_into(&mut output).unwrap();
    same(&output[..N], &input[..N]);
    same(&output[N..], &bank[N..]);
    let wrong = foreign.upload_owned(&input).unwrap();
    assert!(matches!(
        prepared.call(&mut [
            wrong.read_argument(PcuBindingRef::new(0, 0)),
            native_output.write_argument(PcuBindingRef::new(0, 1))
        ]),
        Err(PcuVulkanError::InvalidArguments)
    ));
    assert!(!prepared.last_call_may_have_written());
    assert!(!prepared.last_call_completion_uncertain());
    native_output.read_into(&mut output).unwrap();
    same(&output[..N], &input[..N]);
    let empty: [T; 0] = [];
    assert!(matches!(
        prepared.call(&mut [
            PcuVulkanArgument::host(PcuHostArgument::read(PcuBindingRef::new(0, 0), &empty)),
            native_output.write_argument(PcuBindingRef::new(0, 1))
        ]),
        Err(PcuVulkanError::InvalidArguments)
    ));
    assert!(!prepared.last_call_may_have_written());
    prepared
        .call(&mut [
            native_input.read_argument(PcuBindingRef::new(0, 0)),
            PcuVulkanArgument::host(PcuHostArgument::read_write(
                PcuBindingRef::new(0, 1),
                &mut output,
            )),
        ])
        .unwrap();
    same(&output[..N], &input[..N]);
    same(&output[N..], &bank[N..]);
    drop(input_owner);
    global::clear_thread_cache().unwrap();
    output_owner.read_into(&mut output).unwrap();
    same(&output[..N], &input[..N]);
    same(&output[N..], &bank[N..]);
}

#[test]
#[ignore = "requires actual Vulkan GPU and coordinated exclusive correctness window"]
fn twenty_two_carrier_source_prepared_odd_prefix_and_foreign_root() {
    policy(PcuRangePolicy::Reject);
    let (backend, _) = device::selected();
    let foreign = PcuVulkanBackend::new().unwrap();
    macro_rules! carriers {($($ty:ty),+ $(,)?)=>{$(carrier::<$ty>(&backend,&foreign);)+};}
    carriers!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        PcuI256,
        PcuU256,
        PcuI512,
        PcuU512,
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

#[test]
#[ignore = "requires actual Vulkan GPU and coordinated exclusive correctness window"]
fn fatal_precommit_preserves_ready_and_recovered_commit_is_observable() {
    policy(PcuRangePolicy::Reject);
    let bank = [17.0_f32; 68];
    let mut owner = owned::identity(&bank).unwrap();
    let sibling = owned::identity(&owner).unwrap();
    let matrix_owner = matrix(&[[1.0; 3]; 2]).unwrap();
    let mut left = [f32::MAX; 65];
    let right = [f32::MAX; 65];
    let mut output = bank;
    let error = add(&left, &right, &mut owner).unwrap_err();
    assert_eq!(
        error.arithmetic_fault().unwrap().kind,
        PcuExecutionFaultKind::ArithmeticOverflow
    );
    owner.read_into(&mut output).unwrap();
    same(&output, &bank);
    policy(PcuRangePolicy::Clamp);
    let error = add(&left, &right, &mut owner).unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert!(fault.recovered);
    assert_eq!(fault.invocation_id, 0);
    owner.read_into(&mut output).unwrap();
    assert_eq!(output[..65], [f32::MAX; 65]);
    assert_eq!(output[65..], [17.0; 3]);
    // A later fatal lane dominates earlier recovered lanes, preserving the entire old Ready value.
    left[7] = f32::NAN;
    let error = add(&left, &right, &mut owner).unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert!(!fault.recovered);
    assert_eq!(fault.invocation_id, 7);
    owner.read_into(&mut output).unwrap();
    assert_eq!(output[..65], [f32::MAX; 65]);
    assert_eq!(output[65..], [17.0; 3]);
    assert!(output_first(&mut owner, &matrix_owner, &right).is_err());
    owner.read_into(&mut output).unwrap();
    assert_eq!(output[..65], [f32::MAX; 65]);
    left.fill(1.0);
    let error = add(&left, &[], &mut owner).unwrap_err();
    assert!(error.arithmetic_fault().is_none());
    owner.read_into(&mut output).unwrap();
    assert_eq!(output[..65], [f32::MAX; 65]);
    add(&left, &[2.0; 65], &mut owner).unwrap();
    owner.read_into(&mut output).unwrap();
    assert_eq!(output[..65], [3.0; 65]);
    assert_eq!(output[65..], [17.0; 3]);
    sibling.read_into(&mut output).unwrap();
    same(&output, &bank);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[test]
#[ignore = "requires actual Vulkan GPU and coordinated exclusive correctness window"]
fn paired_div_rem_publication_mixed_outputs_and_fatal_retry() {
    policy(PcuRangePolicy::Reject);
    let left = [17_i16; 65];
    let mut right = [3_i16; 65];
    let bank = [91_i16; 68];
    let input = owned::identity(&left).unwrap();
    let mut quotient = owned::identity(&bank).unwrap();
    let mut remainder = owned::identity(&bank).unwrap();
    div_rem::i16_source::direct::<65>(&input, &right, &mut quotient, &mut remainder).unwrap();
    let mut q = bank;
    let mut r = bank;
    quotient.read_into(&mut q).unwrap();
    remainder.read_into(&mut r).unwrap();
    assert_eq!(q[..65], [5; 65]);
    assert_eq!(r[..65], [2; 65]);
    assert_eq!(q[65..], [91; 3]);
    assert_eq!(r[65..], [91; 3]);
    right[7] = 0;
    let error = div_rem::i16_source::direct::<65>(&input, &right, &mut quotient, &mut remainder)
        .unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert_eq!(fault.invocation_id, 7);
    assert_eq!(fault.kind, PcuExecutionFaultKind::DivideByZero);
    quotient.read_into(&mut q).unwrap();
    remainder.read_into(&mut r).unwrap();
    assert_eq!(q[..65], [5; 65]);
    assert_eq!(r[..65], [2; 65]);
    let mut host = bank;
    assert!(div_rem::i16_source::direct::<65>(&input, &right, &mut quotient, &mut host).is_err());
    assert_eq!(host, bank);
    right.fill(4);
    div_rem::i16_source::direct::<65>(&input, &right, &mut quotient, &mut host).unwrap();
    quotient.read_into(&mut q).unwrap();
    assert_eq!(q[..65], [4; 65]);
    assert_eq!(host[..65], [1; 65]);
    assert_eq!(host[65..], [91; 3]);
    assert_eq!(q[65..], [91; 3]);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[path = "float/float.rs"]
mod float;
