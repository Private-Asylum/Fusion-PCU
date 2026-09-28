use fusion_pcu_cpu::PcuF32Reference;
use fusion_pcu_macros::pcu;
#[rustfmt::skip]
use pcu_alias::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchDataOp,
    PcuDispatchSubmission,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuScalar,
    PcuSynchronousHostDispatchBackend,
    PcuValueType,
};
use core::num::NonZeroU32;

extern crate pcu_alias;

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn copy<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn copy_grid<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}

#[test]
fn generic_scalar_copy_specializes_binding_type_and_executes_on_cpu() {
    let bindings = copy_bindings::<f32>();
    assert_eq!(
        bindings[0].binding_type,
        PcuBindingType::Value(PcuValueType::f32())
    );
    assert_eq!(bindings[0].access, PcuBindingAccess::ReadOnly);
    assert_eq!(
        bindings[1].binding_type,
        PcuBindingType::Value(PcuValueType::f32())
    );
    assert_eq!(bindings[1].access, PcuBindingAccess::ReadWrite);

    let builder = copy_ir::<f32, 4>(&bindings).expect("bounded generic identity builder");
    let kernel = builder.ir();
    assert!(matches!(
        kernel.ops[0],
        pcu_alias::PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. })
    ));
    assert!(matches!(
        kernel.ops[1],
        pcu_alias::PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. })
    ));
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(4).expect("nonzero")),
    };
    let input = [1.0_f32, -2.0, 3.5, 0.0];
    let mut output = [0.0_f32; 4];
    let mut host = [
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
        .run_host(submission, &mut host, PcuInvocationParameters::empty())
        .expect("CPU reference executes generic specialization");
    assert_eq!(output.map(f32::to_bits), input.map(f32::to_bits));
}

#[test]
fn generic_scalar_grid_stride_copy_specializes_extent_and_executes_on_cpu() {
    const EXTENT: usize = 11;
    const LANES: usize = 3;
    let bindings = copy_grid_bindings::<f32>();
    let builder = copy_grid_ir::<f32, EXTENT>(&bindings).expect("bounded generic grid identity");
    let kernel = builder.ir();
    let [
        pcu_alias::PcuDispatchOp::GridStrideLoop { extent, body },
        pcu_alias::PcuDispatchOp::Control(pcu_alias::PcuDispatchControlOp::Return),
    ] = kernel.ops
    else {
        panic!("generic grid identity emits one region and return")
    };
    assert_eq!(*extent, u32::try_from(EXTENT).expect("bounded extent"));
    assert!(matches!(
        body[0],
        pcu_alias::PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            index: pcu_alias::PcuDispatchIndex::GridStrideId,
            ..
        })
    ));
    assert!(matches!(
        body[1],
        pcu_alias::PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            index: pcu_alias::PcuDispatchIndex::GridStrideId,
            ..
        })
    ));

    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(
            NonZeroU32::new(u32::try_from(LANES).expect("bounded lane count"))
                .expect("nonzero lane count"),
        ),
    };
    let input: [f32; EXTENT] = core::array::from_fn(|index| {
        f32::from(u16::try_from(index).expect("bounded test index")) * -1.25
    });
    let mut output = [0_f32; EXTENT];
    let mut host = [
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
        .run_host(submission, &mut host, PcuInvocationParameters::empty())
        .expect("CPU reference executes generic grid specialization");
    assert_eq!(output.map(f32::to_bits), input.map(f32::to_bits));
}

#[test]
fn generic_scalar_copy_supports_other_sealed_scalar_specializations() {
    assert_identity_specialization::<f64, 8>();
    assert_identity_specialization::<u8, 8>();
    assert_identity_specialization::<u16, 8>();
    assert_identity_specialization::<u32, 8>();
    assert_identity_specialization::<u64, 8>();
    assert_identity_specialization::<i8, 8>();
    assert_identity_specialization::<i16, 8>();
    assert_identity_specialization::<i32, 8>();
    assert_identity_specialization::<i64, 8>();
    assert_identity_specialization::<pcu_alias::PcuF16Bits, 8>();
    assert_identity_specialization::<pcu_alias::PcuBf16Bits, 8>();
}

fn assert_identity_specialization<T: PcuScalar, const N: usize>() {
    let bindings = copy_bindings::<T>();
    let expected = PcuBindingType::Value(PcuValueType::Scalar(T::TYPE));
    assert_eq!(bindings[0].binding_type, expected);
    assert_eq!(bindings[1].binding_type, expected);
    let kernel = copy_ir::<T, N>(&bindings).expect("sealed scalar identity lowers");
    assert!(
        kernel
            .ir()
            .required_type_support()
            .contains(pcu_alias::PcuValueTypeCaps::for_scalar(T::TYPE))
    );
}
