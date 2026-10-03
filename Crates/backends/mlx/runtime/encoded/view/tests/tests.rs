//! Independent exact view bits, cached descriptors and same-session lifetime proof.
#[rustfmt::skip]
use std::rc::Rc;
use crate::{MlxRuntime, MlxError};
#[test]
#[ignore = "Requires real pinned MLX GPU view/reshape evaluation and explicit activity guard."]
fn exact_f32_view_bits_cache_shapes_and_retained_lifetimes() {
    let session = MlxRuntime::load_default().unwrap().open_gpu(0).unwrap();
    let bits = [
        0,
        0x8000_0000,
        1,
        0x007f_ffff,
        0x7fc1_2345,
        0xffc7_4321,
        0x7f80_0000,
        0xff80_0000,
    ];
    let native = session
        .upload_f32([2, 4], &bits.map(f32::from_bits))
        .unwrap();
    #[cfg(feature = "view-census")]
    crate::reset_view_call_census();
    let encoded = native.encoded_f32_view().unwrap();
    #[cfg(feature = "view-census")]
    assert_view_calls(1);
    #[cfg(feature = "view-census")]
    crate::reset_view_call_census();
    let cached = native.encoded_f32_view().unwrap();
    #[cfg(feature = "view-census")]
    assert_view_calls(0);
    assert!(Rc::ptr_eq(&encoded.native, &cached.native));
    assert_eq!(encoded.scalar_type(), fusion_pcu::PcuScalarType::F32);
    assert_eq!(encoded.element_count(), 8);
    assert_eq!(encoded.byte_len(), 32);
    for shape in [[2, 4], [4, 2], [1, 8], [8, 1]] {
        #[cfg(feature = "view-census")]
        crate::reset_view_call_census();
        let view = encoded.native_f32_view(shape).unwrap();
        #[cfg(feature = "view-census")]
        assert_view_calls(1);
        #[cfg(feature = "view-census")]
        crate::reset_view_call_census();
        let cached = encoded.native_f32_view(shape).unwrap();
        #[cfg(feature = "view-census")]
        assert_view_calls(0);
        assert!(Rc::ptr_eq(&view.array, &cached.array));
        assert!(view.session().same_session(&session));
        let roundtrip = view.encoded_f32_view().unwrap();
        let mut output = [91.0_f32; 10];
        roundtrip.read_into(&mut output).unwrap();
        assert_eq!(
            output[..8]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
        assert_eq!(output[8..], [91.0; 2]);
    }
    assert!(matches!(
        encoded.native_f32_view([0, 8]),
        Err(MlxError::InvalidExtent)
    ));
    assert!(matches!(
        encoded.native_f32_view([3, 3]),
        Err(MlxError::InvalidExtent)
    ));
    assert!(matches!(
        encoded.native_f32_view([usize::MAX, 2]),
        Err(MlxError::InvalidExtent)
    ));
    let wrong = session.upload_encoded(&[1_u32; 8]).unwrap();
    assert!(matches!(
        wrong.native_f32_view([2, 4]),
        Err(MlxError::UnsupportedScalar(fusion_pcu::PcuScalarType::U32))
    ));
    let retained = encoded.native_f32_view([4, 2]).unwrap();
    drop(native);
    drop(encoded);
    drop(cached);
    drop(session);
    let mut output = [0.0; 8];
    retained.read_into_f32(&mut output).unwrap();
    assert_eq!(output.map(f32::to_bits), bits);
}
#[cfg(feature = "view-census")]
fn assert_view_calls(expected: u64) {
    let counts = crate::view_call_census();
    assert_eq!(counts.input_clone_calls, expected);
    assert_eq!(counts.construct_calls, expected);
    assert_eq!(counts.eval_calls, expected);
    assert_eq!(counts.synchronize_calls, expected);
    assert_eq!(counts.wait_calls, expected);
    assert_eq!(counts.available_calls, expected);
    assert_eq!(counts.shared_storage_validation_calls, expected);
    assert_eq!(counts.input_release_calls, expected);
}
#[cfg(feature = "tensor")]
#[test]
#[ignore = "Requires actual same-session native MatMul composition, view retention and foreign-session rejection."]
fn encoded_native_matmul_encoded_composition() {
    use fusion_pcu::{
        PcuCompoundArithmeticPolicy, PcuPrecisionPolicy, PcuNumericalOptions, PcuScalarType,
    };
    use fusion_pcu::dialect::tensor::Graph;
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let mut graph = Graph::try_new().unwrap();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::BackendOptimized,
        ..PcuNumericalOptions::default()
    });
    let a = graph.input([2, 3], PcuScalarType::F32).unwrap();
    let b = graph.input([3, 2], PcuScalarType::F32).unwrap();
    let c = graph.matmul(a, b).unwrap();
    let prepared = session
        .prepare_matmul(&graph, graph.node(c).unwrap())
        .unwrap();
    let a = session
        .upload_encoded(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let b = session
        .upload_encoded(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0])
        .unwrap();
    let left = a.native_f32_view([2, 3]).unwrap();
    let right = b.native_f32_view([3, 2]).unwrap();
    let product = session.execute_matmul(&prepared, &left, &right).unwrap();
    let output = product.encoded_f32_view().unwrap();
    let sibling = output.clone();
    drop(a);
    drop(b);
    drop(left);
    drop(right);
    drop(product);
    let mut actual = [91.0; 6];
    output.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [58.0_f32, 64.0, 139.0, 154.0, 91.0, 91.0].map(f32::to_bits)
    );
    let foreign = runtime
        .open_gpu(0)
        .unwrap()
        .upload_encoded(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0])
        .unwrap();
    let bad = foreign.native_f32_view([3, 2]).unwrap();
    let retained_left = session
        .upload_encoded(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .native_f32_view([2, 3])
        .unwrap();
    assert!(matches!(
        session.execute_matmul(&prepared, &retained_left, &bad),
        Err(MlxError::ForeignSession)
    ));
    drop(output);
    let mut actual = [0.0; 4];
    sibling.read_into(&mut actual).unwrap();
    assert_eq!(
        actual.map(f32::to_bits),
        [58.0_f32, 64.0, 139.0, 154.0].map(f32::to_bits)
    );
}
