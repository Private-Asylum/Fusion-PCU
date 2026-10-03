//! Genuine source-retained device kernels, actual owned submission and ordinary facade.

#[rustfmt::skip]
use pcu_facade::{
    PcuBindingAccess,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDispatchSubmission,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuOwnedCompletion,
    PcuOwnedDispatchBackend,
    PcuOwnedDispatchMemorySession,
    PcuPreparedOwnedDispatch,
    PcuRuntimeDiscovery,
    PcuValueType,
};
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalDiscovery,
    MetalOwnedDispatchBackend,
};
use super::*;
#[path = "../../../../runtime/transport/tests/fixture/fixture.rs"]
mod fixture;

fn backend() -> MetalOwnedDispatchBackend {
    let discovery = MetalDiscovery::discover().unwrap();
    let mut providers = [pcu_facade::PcuProviderDescriptor {
        id: pcu_facade::PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: pcu_facade::PcuProviderReadiness {
            status: pcu_facade::PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    discovery.providers(&mut providers).unwrap();
    discovery
        .open_owned_device(pcu_facade::PcuObjectRef {
            provider: providers[0].id,
            generation: providers[0].generation,
            kind: pcu_facade::PcuObjectKind::Device,
            id: 0,
        })
        .unwrap()
}

fn resident<T: Sample>(backend: &MetalOwnedDispatchBackend, foreign: &MetalOwnedDispatchBackend) {
    let pool = PcuMemoryPoolId(195);
    let mut direct = saved_prepare_device::<T, 17, _>(backend).unwrap();
    let mut grid = saved_grid_prepare_device::<T, 17, _>(backend).unwrap();
    for phase in 0..3 {
        let values: Vec<T> = (0..24).map(|lane| T::raw(lane, phase)).collect();
        let seed_values: Vec<T> = (0..5).map(|lane| T::raw(lane + 1, phase)).collect();
        let input = backend.upload_buffer(pool, &values).unwrap();
        let seed = backend.upload_buffer(pool, &seed_values).unwrap();
        let mut ghost = foreign.upload_buffer(pool, &[T::raw(5, phase)]).unwrap();
        for grid_mode in [false, true] {
            let mut stage = vec![T::raw(9, phase); 20];
            let mut output = vec![T::raw(11, phase); 22];
            let old_stage = stage.clone();
            let old_output = output.clone();
            let mut stage_owner = backend.upload_buffer(pool, &stage).unwrap();
            let mut output_owner = backend.upload_buffer(pool, &output).unwrap();
            if grid_mode {
                grid(
                    &input,
                    &seed,
                    &mut ghost,
                    &mut stage_owner,
                    &mut output_owner,
                )
                .unwrap();
            } else {
                direct(
                    &input,
                    &seed,
                    &mut ghost,
                    &mut stage_owner,
                    &mut output_owner,
                )
                .unwrap();
            }
            backend
                .download_buffer(pool, &stage_owner, &mut stage)
                .unwrap();
            backend
                .download_buffer(pool, &output_owner, &mut output)
                .unwrap();
            assert_eq!(
                bytes(&stage),
                [bytes(&seed_values[..1]).repeat(17), bytes(&old_stage[17..])].concat()
            );
            assert_eq!(
                bytes(&output),
                [bytes(&values[..17]), bytes(&old_output[17..])].concat()
            );
            let mut short = backend.upload_buffer(pool, &old_output[..16]).unwrap();
            let refused = if grid_mode {
                grid(&input, &seed, &mut ghost, &mut stage_owner, &mut short)
            } else {
                direct(&input, &seed, &mut ghost, &mut stage_owner, &mut short)
            };
            assert!(matches!(
                refused,
                Err(fusion_pcu_metal::MetalOwnedDispatchError::Binding(
                    pcu_facade::PcuOwnedDispatchBindingError::BufferTooSmall { .. }
                ))
            ));
            let mut retained = old_stage.clone();
            backend
                .download_buffer(pool, &stage_owner, &mut retained)
                .unwrap();
            assert_eq!(bytes(&retained), bytes(&stage));
        }
        let mut retained_input = values.clone();
        backend
            .download_buffer(pool, &input, &mut retained_input)
            .unwrap();
        assert_eq!(bytes(&retained_input), bytes(&values));
    }
}

fn owned<T: Sample>(backend: &MetalOwnedDispatchBackend) {
    let pool = PcuMemoryPoolId(195);
    for grid in [false, true] {
        fixture::visit(
            T::TYPE,
            grid,
            pcu_facade::PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            |ir| {
                let shape = PcuInvocationShape::invocations(
                    core::num::NonZeroU32::new(ir.entry.logical_shape[0]).unwrap(),
                );
                let prepared = backend
                    .prepare_dispatch_owned_direct(
                        PcuDispatchSubmission { kernel: ir, shape },
                        PcuInvocationParameters::empty(),
                    )
                    .unwrap();
                let schema = prepared.binding_schema();
                assert_eq!(schema.len(), 4);
                assert!(!schema.iter().any(|role| role.target == fixture::GHOST));
                assert_eq!(
                    schema
                        .iter()
                        .find(|role| role.target == fixture::SEED)
                        .unwrap()
                        .min_required_bytes,
                    T::HOST_SIZE as u64
                );
                for phase in 0..3 {
                    let input_values: Vec<T> = (0..24).map(|lane| T::raw(lane, phase)).collect();
                    let seed_values: Vec<T> = (0..5).map(|lane| T::raw(lane + 1, phase)).collect();
                    let mut stage = vec![T::raw(9, phase); 20];
                    let mut output = vec![T::raw(11, phase); 22];
                    let old_stage = stage.clone();
                    let old_output = output.clone();
                    let owners = [
                        backend.upload_buffer(pool, &input_values).unwrap(),
                        backend.upload_buffer(pool, &stage).unwrap(),
                        backend.upload_buffer(pool, &seed_values).unwrap(),
                        backend.upload_buffer(pool, &output).unwrap(),
                    ];
                    let bindings = [
                        fixture::INPUT,
                        fixture::STAGE,
                        fixture::SEED,
                        fixture::OUTPUT,
                    ]
                    .iter()
                    .enumerate()
                    .map(|(slot, &target)| {
                        backend
                            .bind(
                                target,
                                if slot == 1 || slot == 3 {
                                    PcuBindingAccess::ReadWrite
                                } else {
                                    PcuBindingAccess::ReadOnly
                                },
                                PcuBindingType::Value(PcuValueType::Scalar(T::TYPE)),
                                owners[slot].resource(),
                            )
                            .unwrap()
                    })
                    .collect();
                    let mut completion = prepared.submit_owned_direct(bindings).unwrap();
                    assert_eq!(completion.wait().unwrap(), PcuCompletionOutcome::Succeeded);
                    drop(completion);
                    backend
                        .download_buffer(pool, &owners[1], &mut stage)
                        .unwrap();
                    backend
                        .download_buffer(pool, &owners[3], &mut output)
                        .unwrap();
                    assert_eq!(
                        bytes(&stage),
                        [bytes(&seed_values[..1]).repeat(17), bytes(&old_stage[17..])].concat()
                    );
                    assert_eq!(
                        bytes(&output),
                        [bytes(&input_values[..17]), bytes(&old_output[17..])].concat()
                    );
                }
            },
        );
    }
}

fn escaped<T: Sample>() {
    let selected = backend();
    let pool = PcuMemoryPoolId(195);
    let mut source = saved_prepare_device::<T, 17, _>(&selected).unwrap();
    let input_values: Vec<T> = (0..24).map(|lane| T::raw(lane, 71)).collect();
    let seed_values = [T::raw(1, 71)];
    let input = selected.upload_buffer(pool, &input_values).unwrap();
    let seed = selected.upload_buffer(pool, &seed_values).unwrap();
    let mut ghost = selected.upload_buffer(pool, &[T::raw(5, 71)]).unwrap();
    let stage_values = [T::raw(9, 71); 20];
    let output_values = [T::raw(11, 71); 22];
    let mut stage = selected.upload_buffer(pool, &stage_values).unwrap();
    let mut output = selected.upload_buffer(pool, &output_values).unwrap();
    let mut provider = selected.memory_provider(pool);
    drop(selected);
    source(&input, &seed, &mut ghost, &mut stage, &mut output).unwrap();
    drop(source);
    drop(input);
    drop(seed);
    drop(ghost);
    let mut stage_bytes = vec![0; 20 * T::HOST_SIZE];
    let mut output_bytes = vec![0; 22 * T::HOST_SIZE];
    provider
        .transfer_from(stage.resource(), 0, &mut stage_bytes)
        .unwrap();
    provider
        .transfer_from(output.resource(), 0, &mut output_bytes)
        .unwrap();
    assert_eq!(
        stage_bytes,
        [bytes(&seed_values).repeat(17), bytes(&stage_values[17..])].concat()
    );
    assert_eq!(
        output_bytes,
        [bytes(&input_values[..17]), bytes(&output_values[17..])].concat()
    );
}

#[test]
#[ignore = "Requires actual Metal owned/device byte transport and authentic sessions."]
fn all_twenty_two_owned_submission_and_annotated_device_saved_ssa() {
    let _policy = crate::source_policy_guard();
    pcu_facade::global::configure(pcu_facade::global::PcuExecutionPolicy::default()).unwrap();
    let selected = backend();
    let foreign = backend();
    macro_rules! run {($($ty:ty),+)=>{$(resident::<$ty>(&selected,&foreign);owned::<$ty>(&selected);escaped::<$ty>();)+};}
    run!(
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
        f32,
        f64,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        PcuF128Bits,
        PcuF256Bits
    );
}

#[test]
#[ignore = "Requires actual ordinary Metal all22 transport source admission."]
fn ordinary_all_twenty_two_saved_ssa_and_untouched_mutable_metadata() {
    let _policy = crate::source_policy_guard();
    for policy in requests() {
        pcu_facade::global::configure(policy).unwrap();
        macro_rules! run {($($ty:ty),+)=>{$(ordinary::<$ty>();)+};}
        run!(
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
            f32,
            f64,
            PcuF16Bits,
            PcuBf16Bits,
            PcuF8E4M3FnBits,
            PcuF8E5M2Bits,
            PcuF128Bits,
            PcuF256Bits
        );
    }
}
fn ordinary<T: Sample>() {
    for phase in 0..3 {
        let input: Vec<T> = (0..24).map(|lane| T::raw(lane, phase)).collect();
        let seed = T::raw(1, phase);
        let mut stage = vec![T::raw(9, phase); 20];
        let mut output = vec![T::raw(11, phase); 22];
        let old_stage = stage.clone();
        let old_output = output.clone();
        saved::<T, 17>(&input, &seed, &mut [], &mut stage, &mut output).unwrap();
        assert_eq!(
            bytes(&stage),
            [bytes(&[seed]).repeat(17), bytes(&old_stage[17..])].concat()
        );
        assert_eq!(
            bytes(&output),
            [bytes(&input[..17]), bytes(&old_output[17..])].concat()
        );
        let before = bytes(&stage);
        let mut short = vec![T::raw(3, phase); 16];
        assert!(saved::<T, 17>(&input, &seed, &mut [], &mut stage, &mut short).is_err());
        assert_eq!(bytes(&stage), before);
        saved_grid::<T, 17>(&input, &seed, &mut [], &mut stage, &mut output).unwrap();
        assert_eq!(
            bytes(&stage),
            [bytes(&[seed]).repeat(17), bytes(&old_stage[17..])].concat()
        );
        assert_eq!(
            bytes(&output),
            [bytes(&input[..17]), bytes(&old_output[17..])].concat()
        );
    }
    pcu_facade::global::clear_thread_cache().unwrap();
}
