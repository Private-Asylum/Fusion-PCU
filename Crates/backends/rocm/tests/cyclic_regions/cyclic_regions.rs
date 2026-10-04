//! Cold cyclic-region refusal and valid owner retry on an actual HIP device.
extern crate pcu_facade as fusion_pcu;
#[path = "../../benches/strict_matmul/selection.rs"]
mod selection;
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuDispatchOp,
    PcuHostKernelBackend,
    PcuReproducibility,
};
#[pcu(invocations = 1)]
fn identity(input: &[u32], output: &mut [u32]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}
#[test]
#[ignore = "actual ROCm device required"]
fn native_prepare_refuses_cyclic_regions_and_keeps_valid_execution() {
    static CYCLIC: [PcuDispatchOp<'static>; 1] = [PcuDispatchOp::GridStrideLoop {
        extent: 1,
        body: &CYCLIC,
    }];
    let (_discovery, backend, _runtime) = selection::selected_device();
    let bindings = identity_bindings();
    let builder = identity_ir(&bindings).unwrap();
    for reproducibility in [
        PcuReproducibility::Unspecified,
        PcuReproducibility::PortableV1,
    ] {
        let mut kernel = builder.ir();
        kernel.ops = &CYCLIC;
        kernel
            .numerical_requirements
            .numerical_options
            .reproducibility = reproducibility;
        assert!(fusion_pcu_rocm::lower_dispatch_to_hip_source(&kernel).is_err());
        assert!(backend.prepare_host_kernel(&kernel).is_err());
    }
    let mut valid = identity_prepare(backend.as_ref()).unwrap();
    let mut output = [17; 3];
    valid(&[0xfedc_ba98], &mut output).unwrap();
    assert_eq!(output, [0xfedc_ba98, 17, 17]);
}
