//! Direct syntax remains available without any runtime provider or implicit CPU fallback.
// This is the provider-free configuration. Enabling any runtime provider is
// explicit opt-in, including CPU; it is not an implicit fallback.
#![cfg(not(any(
    feature = "cpu",
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    feature = "vulkan"
)))]

#[rustfmt::skip]
use fusion_pcu::{
    global::PcuExecutionError,
    pcu,
    PcuScalar,
};

#[pcu(invocations = N)]
fn copy_scalar<T, const N: usize>(input: &[T; N], output: &mut [T; N])
where
    T: PcuScalar,
{
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[allow(non_snake_case)]
mod argument_hygiene {
    use fusion_pcu::pcu;

    #[pcu(invocations = 1)]
    pub fn collision(__PCU_SITE: &[f32; 1], __pcu_arguments: &mut [f32; 1]) {
        let id = pcu::context::global_invocation_id();
        __pcu_arguments[id] = __PCU_SITE[id];
    }
}

#[allow(non_upper_case_globals)]
mod cold_hygiene {
    use fusion_pcu::pcu;

    #[pcu(invocations: __pcu_bindings)]
    pub fn collision<const __pcu_bindings: usize, const __pcu_builder: usize>(
        input: &[u64; __pcu_bindings],
        output: &mut [u64; __pcu_bindings],
    ) {
        let id = pcu::context::global_invocation_id();
        output[id] = input[id];
    }
}

#[test]
fn default_does_not_execute_on_cpu_or_mutate_host_output() {
    let input = [1_u64, u64::MAX];
    let mut output = [7_u64, 8];
    assert!(matches!(
        copy_scalar(&input, &mut output),
        Err(PcuExecutionError::NoBackendEnabled)
    ));
    assert_eq!(output, [7, 8]);
}

#[test]
fn generated_locals_do_not_collide_with_source_parameters() {
    let mut output = [19.0];
    assert!(matches!(
        argument_hygiene::collision(&[3.0], &mut output),
        Err(PcuExecutionError::NoBackendEnabled)
    ));
    assert_eq!(output.map(f32::to_bits), [19.0_f32.to_bits()]);
}

#[test]
fn cold_locals_do_not_shadow_const_generic_parameters() {
    let mut output = [23_u64];
    assert!(matches!(
        cold_hygiene::collision::<1, 2>(&[5], &mut output),
        Err(PcuExecutionError::NoBackendEnabled)
    ));
    assert_eq!(output, [23]);
}
