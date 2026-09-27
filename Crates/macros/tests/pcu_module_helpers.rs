use fusion_pcu_cpu::PcuF32Reference;
use fusion_pcu_macros::pcu_module;
use pcu_alias::{
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuSynchronousHostDispatchBackend,
};
use core::num::NonZeroU32;

extern crate pcu_alias;

#[pcu_module]
mod kernels {
    #[pcu_fn]
    fn scale(value: f32) -> f32 {
        value * 2.0
    }

    #[pcu_fn]
    fn offset(value: f32) -> f32 {
        scale(value) + 1.0
    }

    #[pcu(invocations = 4, crate_path = ::pcu_alias)]
    pub fn map(input: &[f32], output: &mut [f32]) {
        let invocation = context.global_invocation_id;
        output[invocation] = offset(input[invocation]);
    }
}

#[test]
fn module_helper_calls_inline_into_bounded_map_ir_and_execute() {
    let descriptors = kernels::map_bindings();
    let builder = kernels::map(&descriptors).expect("module-owned helper lowers");
    let kernel = builder.ir();
    let data_ops = kernel
        .ops
        .iter()
        .filter(|operation| matches!(operation, PcuDispatchOp::Data(_)))
        .count();
    assert_eq!(
        data_ops, 6,
        "load, scale multiply, offset add, store plus constants"
    );
    assert!(kernel.ops.iter().any(|operation| matches!(
        operation,
        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            op: pcu_alias::PcuDispatchAluOp::Mul,
            ..
        })
    )));
    assert!(kernel.ops.iter().any(|operation| matches!(
        operation,
        PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            op: pcu_alias::PcuDispatchAluOp::Add,
            ..
        })
    )));

    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(4).expect("nonzero")),
    };
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut bindings = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&input),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::ReadWrite(&mut output),
        },
    ];
    PcuF32Reference
        .run_host(submission, &mut bindings, PcuInvocationParameters::empty())
        .expect("CPU reference executes lowered helper IR");
    assert_eq!(
        output.map(f32::to_bits),
        [3.0_f32, 5.0, 7.0, 9.0].map(f32::to_bits)
    );
}
