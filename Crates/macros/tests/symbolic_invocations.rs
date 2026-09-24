use fusion_pcu_macros::pcu_dispatch;
use pcu_alias::PcuBindingAccess;

extern crate pcu_alias as renamed_pcu;

#[pcu_dispatch(invocations = R * C, crate_path = ::renamed_pcu)]
fn matrix_map<const R: usize, const C: usize>(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] * 2.0;
}

#[pcu_dispatch(invocations = (R + 1) * C - C, crate_path = ::renamed_pcu)]
fn expression_map<const R: usize, const C: usize>(input: &[f32], output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = input[invocation] + 1.0;
}

#[pcu_dispatch(invocations = N, crate_path = ::renamed_pcu)]
fn in_place_map<const N: usize>(output: &mut [f32]) {
    let invocation = context.global_invocation_id;
    output[invocation] = output[invocation] + 1.0;
}

#[test]
fn const_generic_invocations_specialize_to_the_declared_shape() {
    let bindings = matrix_map_bindings();
    assert_eq!(bindings[0].access, PcuBindingAccess::ReadOnly);
    assert_eq!(bindings[1].access, PcuBindingAccess::ReadWrite);
    let kernel = matrix_map::<32, 64>(&bindings).expect("the map IR is valid");
    assert_eq!(kernel.ir().entry.logical_shape, [2048, 1, 1]);

    let second = matrix_map::<4, 5>(&bindings).expect("the second specialization is valid");
    assert_eq!(second.ir().entry.logical_shape, [20, 1, 1]);

    let expression_bindings = expression_map_bindings();
    let expression = expression_map::<7, 9>(&expression_bindings)
        .expect("checked arithmetic expression is valid");
    assert_eq!(expression.ir().entry.logical_shape, [63, 1, 1]);

    let mutable_binding = in_place_map_bindings();
    assert_eq!(mutable_binding[0].access, PcuBindingAccess::ReadWrite);
    let in_place = in_place_map::<16>(&mutable_binding).expect("mutable reads and writes lower");
    assert_eq!(in_place.ir().entry.logical_shape, [16, 1, 1]);
}

#[test]
fn specialized_ir_is_admitted_by_both_current_compute_lowerers() {
    let bindings = matrix_map_bindings();
    let builder = matrix_map::<32, 64>(&bindings).expect("macro builds specialized IR");
    let kernel = builder.ir();

    let hip = fusion_pcu_rocm::lower_dispatch_to_hip_source(&kernel)
        .expect("HIP lowerer admits the specialized kernel");
    assert!(hip.contains("fusion_gid >= 2048u"));

    let mut spirv = fusion_pcu_spirv::PcuSpirvFixedSink::<2048>::new();
    fusion_pcu_spirv::lower_dispatch_to_spirv(
        &kernel,
        fusion_pcu_spirv::PcuSpirvLoweringOptions::minimal_shader(),
        &mut spirv,
    )
    .expect("SPIR-V lowerer admits the specialized kernel");
    assert!(!spirv.is_empty());
}
