//! Real neutral owned leases and typed device calls over the same ordered plan.
#[rustfmt::skip]
use core::num::NonZeroU32;
#[rustfmt::skip]
use crate::{
    MetalDiscovery,
    MetalMemoryResource,
    MetalOwnedDispatchBackend,
};
use super::source;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDeviceArgument,
    PcuDispatchKernelIr,
    PcuDispatchSubmission,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuMemoryPoolId,
    PcuObjectKind,
    PcuOwnedCompletion,
    PcuOwnedDispatchBackend,
    PcuOwnedDispatchMemorySession,
    PcuDeviceKernelBackend,
    PcuPreparedDeviceKernel,
    PcuPreparedOwnedDispatch,
    PcuScalar,
};
fn open() -> MetalOwnedDispatchBackend {
    let discovery = MetalDiscovery::discover().unwrap();
    discovery
        .open_owned_device(discovery.reference(PcuObjectKind::Device, 0))
        .unwrap()
}
fn verify<T: PcuScalar>(
    backend: &MetalOwnedDispatchBackend,
    foreign: &MetalOwnedDispatchBackend,
    ir: &PcuDispatchKernelIr<'_>,
    input: &[T; 7],
    seed: T,
    sentinel: T,
    expected: &[T; 7],
) {
    let targets: [_; 5] = core::array::from_fn(|slot| ir.bindings[slot].reference());
    let submitted = PcuDispatchSubmission {
        kernel: ir,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(ir.entry.logical_shape[0]).unwrap()),
    };
    let owned = backend
        .prepare_dispatch_owned_direct(submitted, PcuInvocationParameters::empty())
        .unwrap();
    assert_eq!(owned.binding_schema().len(), 4);
    assert!(
        owned
            .binding_schema()
            .iter()
            .all(|role| role.target != targets[0])
    );
    let mut device = backend.prepare_device_kernel(ir).unwrap();
    let pool = PcuMemoryPoolId(121);
    let input_owner = backend.upload_buffer(pool, input).unwrap();
    let seed_owner = backend.upload_buffer(pool, &[seed]).unwrap();
    let mut ghost = foreign.upload_buffer(pool, &[sentinel]).unwrap();
    for device_call in [false, true] {
        let mut stage = backend.upload_buffer(pool, &[sentinel; 10]).unwrap();
        let mut output = backend.upload_buffer(pool, &[sentinel; 11]).unwrap();
        if device_call {
            device
                .call(&mut [
                    PcuDeviceArgument::read_write(targets[4], &mut output),
                    PcuDeviceArgument::read(targets[2], &seed_owner),
                    PcuDeviceArgument::read_write(targets[0], &mut ghost),
                    PcuDeviceArgument::read(targets[1], &input_owner),
                    PcuDeviceArgument::read_write(targets[3], &mut stage),
                ])
                .unwrap();
        } else {
            let actual: [&MetalMemoryResource; 4] = [
                input_owner.resource(),
                seed_owner.resource(),
                stage.resource(),
                output.resource(),
            ];
            let bindings = actual
                .iter()
                .enumerate()
                .map(|(slot, resource)| {
                    let declared = &ir.bindings[slot + 1];
                    assert!(matches!(declared.binding_type, PcuBindingType::Value(_)));
                    backend
                        .bind(
                            declared.reference(),
                            declared.access,
                            declared.binding_type,
                            resource,
                        )
                        .unwrap()
                })
                .collect();
            let mut completion = owned.submit_owned_direct(bindings).unwrap();
            assert_eq!(completion.wait().unwrap(), PcuCompletionOutcome::Succeeded);
        }
        let mut stage_host = [sentinel; 10];
        let mut output_host = [sentinel; 11];
        backend
            .download_buffer(pool, &stage, &mut stage_host)
            .unwrap();
        backend
            .download_buffer(pool, &output, &mut output_host)
            .unwrap();
        let encode = |values: &[T]| {
            values
                .iter()
                .flat_map(|value| value.encode_le().as_ref().to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(encode(&stage_host[..7]), encode(input));
        assert_eq!(encode(&stage_host[7..]), encode(&[sentinel; 3]));
        assert_eq!(encode(&output_host[..7]), encode(expected));
        assert_eq!(encode(&output_host[7..]), encode(&[sentinel; 4]));
        let mut unchanged = [sentinel; 7];
        backend
            .download_buffer(pool, &input_owner, &mut unchanged)
            .unwrap();
        assert_eq!(encode(&unchanged), encode(input));
        let mut old_ghost = [sentinel; 1];
        foreign
            .download_buffer(pool, &ghost, &mut old_ghost)
            .unwrap();
        assert_eq!(encode(&old_ghost), encode(&[sentinel]));
    }
}
#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "requires authentic Metal owned and typed device ordered effects"
)]
fn ten_integer_two_float_owned_and_device_actual_resources_keep_tails_and_foreign_ghost() {
    let backend = open();
    let foreign = open();
    macro_rules! integer { ($($ty:ty),*) => { $(
        for phase in [0_u8, 1, 2] {
            let input: [$ty; 7] = core::array::from_fn(|lane| <$ty>::try_from(1 + u8::try_from(lane).unwrap() + phase).unwrap());
            let expected: [$ty; 7] = core::array::from_fn(|lane| (input[lane] + 2) * input[lane] - input[lane]);
            source::integer_ir::<$ty, 7>(&source::integer_bindings::<$ty>()).unwrap().with_ir(|ir|
                verify(&backend, &foreign, ir, &input, 2, 19, &expected));
        }
    )* }; }
    integer!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
    macro_rules! floating { ($($ty:ty),*) => { $(
        for phase in [0_u8, 1, 2] {
            let input: [$ty; 7] = core::array::from_fn(|lane| <$ty>::from(1 + u8::try_from(lane).unwrap() + phase));
            let expected: [$ty; 7] = core::array::from_fn(|lane| {
                let value = u16::from(1 + u8::try_from(lane).unwrap() + phase);
                <$ty>::from((value + 2) * value + value)
            });
            source::floating_ir::<$ty, 7>(&source::floating_bindings::<$ty>()).unwrap().with_ir(|ir|
                verify(&backend, &foreign, ir, &input, 2.0, 19.0, &expected));
        }
    )* }; }
    floating!(f32, f64);
}

#[path = "fault/fault.rs"]
mod fault;
