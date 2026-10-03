//! Independent all-carrier source/graph/native ownership and transaction controls.
#[path = "bytes/bytes.rs"]
mod bytes;
#[path = "graph/graph.rs"]
mod graph;
#[path = "policy/policy.rs"]
mod policy;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingAccess,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuPreparedHostKernel,
    PcuReproducibility,
    PcuScalar,
};
#[rustfmt::skip]
use fusion_pcu_cpu::{
    PcuCpuHostBackend,
    PcuCpuPreparedHost,
};
const N: usize = 47;
const fn arguments<'a, T: PcuScalar>(
    input: &'a [T],
    seed: &'a T,
    ghost: &'a mut [T],
    stage: &'a mut [T],
    output: &'a mut [T],
) -> [PcuHostArgument<'a>; 5] {
    [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), core::slice::from_ref(seed)),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), ghost),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 3), stage),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 4), output),
    ]
}
fn width<T: PcuScalar>(requirements: PcuImplementationRequirements) {
    let backend = PcuCpuHostBackend::scalar();
    for grid in [false, true] {
        let mut plan = graph::with(
            T::TYPE,
            u32::try_from(N).unwrap(),
            grid,
            requirements,
            |kernel| backend.prepare_host_kernel(kernel),
        )
        .unwrap();
        assert!(matches!(plan, PcuCpuPreparedHost::Transport(_)));
        let mut stage = [bytes::sample::<T>(0xA5, 0); N + 3];
        let mut output = [bytes::sample::<T>(0x5A, 0); N + 5];
        let mut expected_stage = stage;
        let mut expected_output = output;
        for bank in [0, 0x55, 0xFF] {
            let input: [T; N + 7] = core::array::from_fn(|lane| bytes::sample(bank, lane));
            let seed = bytes::sample::<T>(bank ^ 0xC3, 19);
            let mut ghost = [];
            bytes::native::<T, N>(&input, &seed, &mut expected_stage, &mut expected_output);
            plan.call(&mut arguments(
                &input,
                &seed,
                &mut ghost,
                &mut stage,
                &mut output,
            ))
            .unwrap();
            bytes::equal(&stage, &expected_stage);
            bytes::equal(&output, &expected_output);
            if grid {
                source::grid::<T, N>(&input, &seed, &mut ghost, &mut stage, &mut output).unwrap();
            } else {
                source::ordered::<T, N>(&input, &seed, &mut ghost, &mut stage, &mut output)
                    .unwrap();
            }
            bytes::equal(&stage, &expected_stage);
            bytes::equal(&output, &expected_output);
            // An explicit local Clamp flag remains representation-only transport, including
            // when the ambient caller range is Reject. There is no arithmetic notice to recover.
            if grid {
                source::clamped_grid::<T, N>(&input, &seed, &mut ghost, &mut stage, &mut output)
                    .unwrap();
            } else {
                source::clamped_ordered::<T, N>(&input, &seed, &mut ghost, &mut stage, &mut output)
                    .unwrap();
            }
            bytes::equal(&stage, &expected_stage);
            bytes::equal(&output, &expected_output);
            let before_stage = stage;
            let before_output = output;
            assert!(
                plan.call(&mut arguments(
                    &input,
                    &seed,
                    &mut ghost,
                    &mut stage,
                    &mut output[..N - 1]
                ))
                .is_err()
            );
            bytes::equal(&stage, &before_stage);
            bytes::equal(&output, &before_output);
            plan.call(&mut arguments(
                &input,
                &seed,
                &mut ghost,
                &mut stage,
                &mut output,
            ))
            .unwrap();
            let wrong_ghost = [bytes::sample::<T>(0xF7, 0)];
            let mut args = arguments(&input, &seed, &mut ghost, &mut stage, &mut output);
            args[2] = PcuHostArgument::read(PcuBindingRef::new(0, 2), &wrong_ghost);
            assert!(plan.call(&mut args).is_err());
            bytes::equal(&stage, &before_stage);
            bytes::equal(&output, &before_output);
        }
    }
}
#[test]
fn all_twenty_two_saved_values_survive_storage_overwrite_and_failed_preflight() {
    policy::each(|requirements| {
        policy::configure(requirements, None);
        carriers!(width, requirements);
    });
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
#[test]
fn write_only_and_portable_remain_separate_cold_refusals() {
    let backend = PcuCpuHostBackend::scalar();
    graph::with(
        pcu_facade::PcuScalarType::U8,
        47,
        false,
        PcuImplementationRequirements::default(),
        |kernel| {
            let mut bindings = kernel.bindings.to_vec();
            bindings[4].access = PcuBindingAccess::WriteOnly;
            let mut changed = *kernel;
            changed.bindings = &bindings;
            assert!(backend.prepare_host_kernel(&changed).is_err());
            let mut changed = *kernel;
            changed
                .numerical_requirements
                .numerical_options
                .reproducibility = PcuReproducibility::PortableV1;
            assert!(backend.prepare_host_kernel(&changed).is_err());
        },
    );
}

#[test]
fn cross_index_mutable_load_requires_one_lane_and_preserves_saved_bits() {
    let backend = PcuCpuHostBackend::scalar();
    for extent in [1, 47] {
        graph::with(
            pcu_facade::PcuScalarType::U8,
            extent,
            false,
            PcuImplementationRequirements::default(),
            |kernel| {
                let mut ops = kernel.ops.to_vec();
                let pcu_facade::PcuDispatchOp::Data(pcu_facade::PcuDispatchDataOp::BindingLoad {
                    index,
                    ..
                }) = &mut ops[2]
                else {
                    panic!("stage load")
                };
                *index = pcu_facade::PcuDispatchIndex::BindingElementZero;
                let mut changed = *kernel;
                changed.ops = &ops;
                let prepared = backend.prepare_host_kernel(&changed);
                if extent == 47 {
                    assert!(prepared.is_err());
                } else {
                    let mut prepared = prepared.unwrap();
                    let mut stage = [0xA5_u8; 4];
                    let mut output = [0x5A_u8; 6];
                    prepared
                        .call(&mut arguments(
                            &[0xF7_u8],
                            &0x3C_u8,
                            &mut [],
                            &mut stage,
                            &mut output,
                        ))
                        .unwrap();
                    assert_eq!(stage, [0x3C, 0xA5, 0xA5, 0xA5]);
                    assert_eq!(output, [0xF7, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A]);
                }
            },
        );
    }
}
