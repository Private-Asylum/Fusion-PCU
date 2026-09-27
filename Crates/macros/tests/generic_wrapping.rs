use fusion_pcu_cpu::PcuU8MapReference;
use fusion_pcu_macros::pcu;
use pcu_alias::{
    PcuBindingRef,
    PcuDispatchSubmission,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuWrappingInteger,
    PcuSynchronousHostDispatchBackend,
};
use core::num::NonZeroU32;

extern crate pcu_alias;

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn wrapping_add<T: PcuWrappingInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id].wrapping_add(right[id]);
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn wrapping_sub_grid<T: PcuWrappingInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id].wrapping_sub(right[id]);
        id += stride;
    }
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn wrapping_mul<T: PcuWrappingInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id].wrapping_mul(right[id]);
}

#[test]
fn generic_wrapping_maps_specialize_typed_ir_and_execute_on_cpu() {
    let left = [0_u8, 1, 128, 255, 17, 8, 4];
    let right = [1_u8, 255, 129, 1, 9, 8, 4];
    let mut output = [0_u8; 7];
    let bindings = wrapping_add_bindings::<u8>();
    let builder = wrapping_add::<u8, 7>(&bindings).expect("generic wrapping add builder");
    let kernel = builder.ir();
    assert!(kernel.ops.iter().any(|op| matches!(
        op,
        pcu_alias::PcuDispatchOp::Data(pcu_alias::PcuDispatchDataOp::Alu {
            value_type: pcu_alias::PcuValueType::Scalar(pcu_alias::PcuScalarType::U8),
            op: pcu_alias::PcuDispatchAluOp::Add,
            ..
        })
    )));
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(7).expect("nonzero")),
    };
    let mut host = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&left),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::Read(&right),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 2),
            slice: PcuHostScalarSlice::ReadWrite(&mut output),
        },
    ];
    PcuU8MapReference
        .run_host(submission, &mut host, PcuInvocationParameters::empty())
        .expect("execute generic wrapping add");
    assert_eq!(output, [1, 0, 1, 0, 26, 16, 8]);

    let mut output = [0_u8; 7];
    let bindings = wrapping_sub_grid_bindings::<u8>();
    let builder =
        wrapping_sub_grid::<u8, 7>(&bindings).expect("generic grid wrapping subtract builder");
    let kernel = builder.ir();
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(3).expect("nonzero")),
    };
    let mut host = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&left),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::Read(&right),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 2),
            slice: PcuHostScalarSlice::ReadWrite(&mut output),
        },
    ];
    PcuU8MapReference
        .run_host(submission, &mut host, PcuInvocationParameters::empty())
        .expect("execute generic grid wrapping subtract");
    assert_eq!(output, [255, 2, 255, 254, 8, 0, 0]);

    let mut output = [0_u8; 7];
    let bindings = wrapping_mul_bindings::<u8>();
    let builder = wrapping_mul::<u8, 7>(&bindings).expect("generic wrapping multiply builder");
    let kernel = builder.ir();
    assert!(kernel.ops.iter().any(|op| matches!(
        op,
        pcu_alias::PcuDispatchOp::Data(pcu_alias::PcuDispatchDataOp::Alu {
            op: pcu_alias::PcuDispatchAluOp::Mul,
            ..
        })
    )));
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(7).expect("nonzero")),
    };
    let mut host = [
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 0),
            slice: PcuHostScalarSlice::Read(&left),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 1),
            slice: PcuHostScalarSlice::Read(&right),
        },
        PcuHostScalarBinding {
            target: PcuBindingRef::new(0, 2),
            slice: PcuHostScalarSlice::ReadWrite(&mut output),
        },
    ];
    PcuU8MapReference
        .run_host(submission, &mut host, PcuInvocationParameters::empty())
        .expect("execute generic wrapping multiply");
    assert_eq!(output, [0, 255, 128, 255, 153, 64, 16]);
}
