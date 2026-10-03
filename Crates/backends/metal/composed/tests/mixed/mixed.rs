//! Actual resident leases and private host sibling publication, separate from aggregate admission.
#[rustfmt::skip]
use crate::{
    MetalMemoryProvider,
    MetalMemoryResource,
    MetalMixedHostArgument,
    MetalSession,
};
use super::source;
#[path = "fault/fault.rs"]
mod fault;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceArgument,
    PcuDeviceBuffer,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryHostAccess,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuScalar,
};
fn bytes<T: PcuScalar>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn owner<T: PcuScalar>(
    provider: &mut MetalMemoryProvider,
    values: &[T],
) -> PcuDeviceBuffer<T, MetalMemoryResource> {
    let bytes = bytes(values);
    let mut resource = provider
        .allocate(PcuMemoryAllocationRequest {
            pool: PcuMemoryPoolId(121),
            size_bytes: u64::try_from(bytes.len()).unwrap(),
            alignment_bytes: 1,
            access: PcuMemoryAccess::ReadWrite,
            host_access: PcuMemoryHostAccess::TransferOnly,
            require_device_local: false,
        })
        .unwrap();
    provider.transfer_to(&mut resource, 0, &bytes).unwrap();
    PcuDeviceBuffer::new(resource, values.len())
}
fn read<T: PcuScalar>(
    provider: &mut MetalMemoryProvider,
    owner: &PcuDeviceBuffer<T, MetalMemoryResource>,
) -> Vec<u8> {
    let mut bytes = vec![0; owner.len() * usize::from(T::TYPE.bit_width()) / 8];
    provider
        .transfer_from(owner.resource(), 0, &mut bytes)
        .unwrap();
    bytes
}
fn verify<T: PcuScalar>(
    session: &MetalSession,
    foreign: &MetalSession,
    ir: &PcuDispatchKernelIr<'_>,
    input: &[T; 7],
    seed: T,
    sentinel: T,
    expected: &[T; 7],
) {
    let mut prepared = session
        .composed_host_backend()
        .prepare_host_kernel(ir)
        .unwrap();
    let targets: [_; 5] = core::array::from_fn(|slot| ir.bindings[slot].reference());
    let mut provider = session.memory_provider(PcuMemoryPoolId(121));
    let mut other = foreign.memory_provider(PcuMemoryPoolId(121));
    let input_owner = owner(&mut provider, input);
    let seed_owner = owner(&mut provider, &[seed]);
    let mut ghost = owner(&mut other, &[sentinel]);
    for mode in 0..3 {
        let mut stage_owner = owner(&mut provider, &[sentinel; 10]);
        let mut output_owner = owner(&mut provider, &[sentinel; 11]);
        let mut host_stage = [sentinel; 10];
        let mut host_output = [sentinel; 11];
        let input_arg = if mode == 1 {
            MetalMixedHostArgument::Host(PcuHostArgument::read(targets[1], input))
        } else {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(targets[1], &input_owner))
        };
        let seed_arg = if mode == 2 {
            MetalMixedHostArgument::Host(PcuHostArgument::read_scalar(targets[2], &seed))
        } else {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(targets[2], &seed_owner))
        };
        let stage_arg = if mode == 1 {
            MetalMixedHostArgument::Host(PcuHostArgument::read_write(targets[3], &mut host_stage))
        } else {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                targets[3],
                &mut stage_owner,
            ))
        };
        let output_arg = if mode == 2 {
            MetalMixedHostArgument::Host(PcuHostArgument::read_write(targets[4], &mut host_output))
        } else {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                targets[4],
                &mut output_owner,
            ))
        };
        prepared
            .call_mixed(&mut [
                output_arg,
                seed_arg,
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                    targets[0], &mut ghost,
                )),
                input_arg,
                stage_arg,
            ])
            .unwrap();
        assert!(prepared.last_call_may_have_written());
        assert!(!prepared.last_call_completion_uncertain());
        let stage = if mode == 1 {
            bytes(&host_stage)
        } else {
            read(&mut provider, &stage_owner)
        };
        let output = if mode == 2 {
            bytes(&host_output)
        } else {
            read(&mut provider, &output_owner)
        };
        let width = usize::from(T::TYPE.bit_width()) / 8;
        assert_eq!(&stage[..7 * width], &bytes(input));
        assert_eq!(&stage[7 * width..], &bytes(&[sentinel; 3]));
        assert_eq!(&output[..7 * width], &bytes(expected));
        assert_eq!(&output[7 * width..], &bytes(&[sentinel; 4]));
        assert_eq!(read(&mut provider, &input_owner), bytes(input));
        assert_eq!(read(&mut other, &ghost), bytes(&[sentinel]));
    }
    let mut stage = owner(&mut provider, &[sentinel; 10]);
    let mut short = [sentinel; 6];
    assert!(matches!(
        prepared.call_mixed(&mut [
            MetalMixedHostArgument::Host(PcuHostArgument::read_write(
                targets[0],
                &mut [] as &mut [T]
            )),
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(targets[1], &input_owner)),
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(targets[2], &seed_owner)),
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(targets[3], &mut stage)),
            MetalMixedHostArgument::Host(PcuHostArgument::read_write(targets[4], &mut short)),
        ]),
        Err(PcuHostDispatchError::BufferTooSmall(_))
    ));
    assert!(!prepared.last_call_may_have_written());
    assert_eq!(read(&mut provider, &stage), bytes(&[sentinel; 10]));
    assert_eq!(bytes(&short), bytes(&[sentinel; 6]));
}
#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "requires authentic Metal checked mixed publication"
)]
fn ten_integer_two_float_actual_leases_mixed_directions_tails_and_ignored_foreign_owner() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    macro_rules! integer { ($($ty:ty),*) => { $(
        for phase in [0_u8, 1, 2] {
            let input: [$ty; 7] = core::array::from_fn(|index|
                <$ty>::try_from(1_u8 + u8::try_from(index).unwrap() + phase).unwrap());
            let expected: [$ty; 7] = core::array::from_fn(|lane|
                (input[lane] + 2) * input[lane] - input[lane]);
            source::integer_ir::<$ty, 7>(&source::integer_bindings::<$ty>()).unwrap().with_ir(|ir|
                verify(&session, &foreign, ir, &input, 2, 19, &expected));
        }
    )* }; }
    integer!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
    macro_rules! floating { ($($ty:ty),*) => { $(
        for phase in [0_u8, 1, 2] {
            let input: [$ty; 7] = core::array::from_fn(|index|
                <$ty>::from(1_u8 + u8::try_from(index).unwrap() + phase));
            let expected: [$ty; 7] = core::array::from_fn(|lane| {
                let value = u16::from(1_u8 + u8::try_from(lane).unwrap() + phase);
                <$ty>::from((value + 2) * value + value)
            });
            source::floating_ir::<$ty, 7>(&source::floating_bindings::<$ty>()).unwrap().with_ir(|ir|
                verify(&session, &foreign, ir, &input, 2.0, 19.0, &expected));
        }
    )* }; }
    floating!(f32, f64);
}
