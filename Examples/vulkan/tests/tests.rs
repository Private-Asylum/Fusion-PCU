//! Checked arithmetic remains unsupported until SPIR-V implements its fault contract.

use fusion_pcu::PcuDispatchOpCaps;
use fusion_pcu_macros::pcu_dispatch;
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_dispatch_to_spirv,
    PcuSpirvError,
    PcuSpirvFixedSink,
    PcuSpirvLoweringOptions,
};

#[pcu_dispatch(kernel_id = 1, invocations = 250)]
fn parallel_float_kernel<const N: usize>(input_a: &[f32], input_b: &[f32], output: &mut [f32]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = ((input_a[id] + input_b[id]) * 2.0) / 2.0 - 1.0;
        id += stride;
    }
}

#[test]
fn checked_float_arithmetic_is_rejected_instead_of_lowered_unchecked() {
    let bindings = parallel_float_kernel_bindings();
    let builder = parallel_float_kernel::<2048>(&bindings).expect("valid source kernel");
    builder.with_ir(|kernel| {
        let mut sink = PcuSpirvFixedSink::<1024>::new();
        let result =
            lower_dispatch_to_spirv(kernel, PcuSpirvLoweringOptions::minimal_shader(), &mut sink);
        assert!(matches!(
            result,
            Err(PcuSpirvError::UnsupportedInstruction(caps))
                if caps == PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
        ));
    });
}
