//! Explicit static raw native program, genuine source IR, separate graph and bit-copy oracle.
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
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuPreparedHostKernel,
    PcuScalar,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanPreparedHost as Prepared,
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
fn exercise<T: PcuScalar>(
    source: &mut Prepared,
    graph: &mut Prepared,
    mut ordinary: impl FnMut(&[T], &T, &mut [T], &mut [T]),
) {
    let mut stage = [bytes::sample::<T>(0xA5, 0); N + 3];
    let mut output = [bytes::sample::<T>(0x5A, 0); N + 5];
    let mut graph_stage = stage;
    let mut graph_output = output;
    let mut expected_stage = stage;
    let mut expected_output = output;
    for bank in [0, 0x55, 0xFF] {
        let input: [T; N + 7] = core::array::from_fn(|lane| bytes::sample(bank, lane));
        let seed = bytes::sample::<T>(bank ^ 0xC3, 19);
        bytes::native::<T, N>(&input, &seed, &mut expected_stage, &mut expected_output);
        source
            .call(&mut arguments(
                &input,
                &seed,
                &mut [],
                &mut stage,
                &mut output,
            ))
            .unwrap();
        graph
            .call(&mut arguments(
                &input,
                &seed,
                &mut [],
                &mut graph_stage,
                &mut graph_output,
            ))
            .unwrap();
        bytes::equal(&stage, &expected_stage);
        bytes::equal(&output, &expected_output);
        bytes::equal(&graph_stage, &expected_stage);
        bytes::equal(&graph_output, &expected_output);
        ordinary(&input, &seed, &mut stage, &mut output);
        bytes::equal(&stage, &expected_stage);
        bytes::equal(&output, &expected_output);
        // Failed later output preflight must preserve every earlier caller output too.
        assert!(
            source
                .call(&mut arguments(
                    &input,
                    &seed,
                    &mut [],
                    &mut stage,
                    &mut output[..N - 1]
                ))
                .is_err()
        );
        bytes::equal(&stage, &expected_stage);
        bytes::equal(&output, &expected_output);
        source
            .call(&mut arguments(
                &input,
                &seed,
                &mut [],
                &mut stage,
                &mut output,
            ))
            .unwrap();
        let wrong_ghost = [bytes::sample::<T>(0xF7, 0)];
        let mut args = arguments(&input, &seed, &mut [], &mut stage, &mut output);
        args[2] = PcuHostArgument::read(PcuBindingRef::new(0, 2), &wrong_ghost);
        assert!(source.call(&mut args).is_err());
        bytes::equal(&stage, &expected_stage);
        bytes::equal(&output, &expected_output);
    }
}
macro_rules! shape {
    ($T:ident,$backend:ident,$requirements:ident,$entry:ident,$ir:ident,$bindings:ident,$grid:expr) => {{
        let bindings = source::$bindings::<$T>();
        let mut source = source::$ir::<$T, N>(&bindings)
            .unwrap()
            .with_numerical_requirements($requirements)
            .with_ir(|kernel| $backend.prepare_host_kernel(kernel))
            .unwrap();
        let mut graph = graph::with(
            $T::TYPE,
            u32::try_from(N).unwrap(),
            $grid,
            $requirements,
            |kernel| $backend.prepare_host_kernel(kernel),
        )
        .unwrap();
        assert_eq!(
            match &source {
                Prepared::OrderedTransport(plan) => plan.profile(),
                _ => panic!("ordinary typed transport family"),
            }
            .requirements(),
            $requirements
        );
        assert_eq!(
            match &source {
                Prepared::OrderedTransport(plan) => plan.profile(),
                _ => panic!("ordinary typed transport family"),
            }
            .resources()
            .len(),
            4
        );
        assert_eq!(
            match &source {
                Prepared::OrderedTransport(plan) => plan.profile(),
                _ => panic!("ordinary typed transport family"),
            }
            .step_count(),
            6
        );
        exercise::<$T>(&mut source, &mut graph, |input, seed, stage, output| {
            source::$entry::<$T, N>(input, seed, &mut [], stage, output).unwrap()
        });
    }};
}
fn width<T: PcuScalar>(backend: &PcuVulkanBackend, requirements: PcuImplementationRequirements) {
    policy::configure(requirements, None);
    shape!(
        T,
        backend,
        requirements,
        ordered,
        ordered_ir,
        ordered_bindings,
        false
    );
    shape!(T, backend, requirements, grid, grid_ir, grid_bindings, true);
    if requirements.range_policy == pcu_facade::PcuRangePolicy::Reject {
        let local = PcuImplementationRequirements {
            range_policy: pcu_facade::PcuRangePolicy::Clamp,
            ..requirements
        };
        clamped_width::<T>(backend, local);
    }
}
fn clamped_width<T: PcuScalar>(
    backend: &PcuVulkanBackend,
    requirements: PcuImplementationRequirements,
) {
    shape!(
        T,
        backend,
        requirements,
        clamped_ordered,
        clamped_ordered_ir,
        clamped_ordered_bindings,
        false
    );
    shape!(
        T,
        backend,
        requirements,
        clamped_grid,
        clamped_grid_ir,
        clamped_grid_bindings,
        true
    );
}
#[test]
#[ignore = "requires an actual Vulkan compute device; genuine ordinary and prepared transport"]
fn all_twenty_two_saved_native_values_and_joint_host_publication() {
    let backend = PcuVulkanBackend::new().unwrap();
    policy::each(|requirements| carriers!(width, &backend, requirements));
    pcu_facade::global::clear_thread_cache().unwrap();
    pcu_facade::global::use_defaults().unwrap();
}

fn one_lane<T: PcuScalar>(backend: &PcuVulkanBackend) {
    graph::with(
        T::TYPE,
        1,
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
            let mut candidate = *kernel;
            candidate.ops = &ops;
            let mut plan = backend.prepare_host_kernel(&candidate).unwrap();
            let input = [bytes::sample::<T>(0xF7, 13)];
            let seed = bytes::sample::<T>(0x3C, 19);
            let mut stage = [bytes::sample::<T>(0xA5, 7); 4];
            let mut output = [bytes::sample::<T>(0x5A, 11); 6];
            let mut expected_stage = stage;
            let mut expected_output = output;
            bytes::native::<T, 1>(&input, &seed, &mut expected_stage, &mut expected_output);
            plan.call(&mut arguments(
                &input,
                &seed,
                &mut [],
                &mut stage,
                &mut output,
            ))
            .unwrap();
            bytes::equal(&stage, &expected_stage);
            bytes::equal(&output, &expected_output);
        },
    );
}
#[test]
#[ignore = "requires an actual Vulkan compute device; mutable element-zero is admitted only for N1"]
fn single_lane_zero_reads_current_private_wide_and_packed_words() {
    let backend = PcuVulkanBackend::new().unwrap();
    carriers!(one_lane, &backend);
}
type NativeResult<T> = Result<T, Box<dyn std::error::Error>>;
#[path = "../../benches/ordered_transport/ffi/compile/compile.rs"]
mod native_compile;
#[test]
#[ignore = "requires official GLSL compiler; creates no device or GPU submission"]
fn independent_native_control_compiles_all_seven_storage_widths() {
    for bytes in [1, 2, 4, 8, 16, 32, 64] {
        let words = native_compile::shader(47, bytes).unwrap();
        assert_eq!(words[0], 0x0723_0203);
        let mut cursor = 5;
        while cursor < words.len() {
            let op = words[cursor] & 0xFFFF;
            if op == 17 {
                assert_eq!(words[cursor + 1], 1, "native U32 Shader capability only");
            }
            cursor += usize::try_from(words[cursor] >> 16).unwrap();
        }
        if let Some(directory) = std::env::var_os("PCU_ORDERED_NATIVE_MODULE_DIR") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory).unwrap();
            let data: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            std::fs::write(directory.join(format!("native-{bytes}.spv")), data).unwrap();
        }
    }
}

fn local_flag<T: PcuScalar>() {
    let bindings = source::clamped_ordered_bindings::<T>();
    let program = source::clamped_ordered_ir::<T, N>(&bindings).unwrap();
    program.with_ir(|kernel| {
        assert_eq!(
            kernel.numerical_requirements.range_policy,
            pcu_facade::PcuRangePolicy::Clamp
        );
        let profile = fusion_pcu_spirv::validate_ordered_scalar_transport_map(kernel).unwrap();
        assert_eq!(profile.resources().len(), 4);
        assert_eq!(profile.step_count(), 6);
    });
    let bindings = source::clamped_grid_bindings::<T>();
    let program = source::clamped_grid_ir::<T, N>(&bindings).unwrap();
    program.with_ir(|kernel| {
        assert_eq!(
            kernel.numerical_requirements.range_policy,
            pcu_facade::PcuRangePolicy::Clamp
        );
        fusion_pcu_spirv::validate_ordered_scalar_transport_map(kernel).unwrap();
    });
}
#[test]
fn genuine_local_clamp_flags_and_unused_typed_rw_roles_are_inert_raw_transport() {
    carriers!(local_flag);
}

fn three_packed<T: PcuScalar>(backend: &PcuVulkanBackend) {
    assert_eq!(core::mem::size_of::<T>(), 1);
    let mut plan = graph::with(
        T::TYPE,
        3,
        false,
        PcuImplementationRequirements::default(),
        |kernel| backend.prepare_host_kernel(kernel),
    )
    .unwrap();
    let input = core::array::from_fn::<_, 3, _>(|lane| bytes::sample::<T>(0xF7, lane));
    let seed = bytes::sample::<T>(0x3C, 7);
    let mut stage = [bytes::sample::<T>(0xA5, 11); 6];
    let mut output = [bytes::sample::<T>(0x5A, 13); 8];
    let mut expected_stage = stage;
    let mut expected_output = output;
    bytes::native::<T, 3>(&input, &seed, &mut expected_stage, &mut expected_output);
    plan.call(&mut arguments(
        &input,
        &seed,
        &mut [],
        &mut stage,
        &mut output,
    ))
    .unwrap();
    bytes::equal(&stage, &expected_stage);
    bytes::equal(&output, &expected_output);
}
#[test]
#[ignore = "requires an actual Vulkan compute device; three logical bytes use padded native U32 descriptors"]
fn odd_three_byte_u8_and_fp8_prefixes_preserve_every_caller_tail() {
    let backend = PcuVulkanBackend::new().unwrap();
    three_packed::<u8>(&backend);
    three_packed::<pcu_facade::PcuF8E4M3FnBits>(&backend);
    three_packed::<pcu_facade::PcuF8E5M2Bits>(&backend);
}
#[path = "offers/offers.rs"]
mod offers;
