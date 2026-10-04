use super::*;
#[rustfmt::skip]
use crate::{
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuU512,
};
use crate::global::PcuSourceShape;
#[test]
fn typed_payload_bits_and_shape_anchor_provenance_are_exact() {
    let (mut capture, [anchor]) =
        PcuTensorGraphCapture::new::<PcuU512, 1>([PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 3,
        }])
        .unwrap();
    let expected = PcuU512::from_limbs_le([1, 2, 3, 4, 5, 6, 7, u64::MAX]);
    let options = PcuNumericalOptions {
        precision: PcuPrecisionPolicy::BackendOptimized,
        ..Default::default()
    };
    capture.numerical_options.set(options);
    let literal = capture.constant_array(&[expected, expected]).unwrap();
    let descriptor = capture.graph.node(literal.value.erase()).unwrap();
    assert_eq!(descriptor.shape, [2]);
    assert_eq!(descriptor.numerical_options, options);
    assert_eq!(descriptor.numerical_mode, None);
    assert_eq!(descriptor.float_underflow_policy, None);
    let crate::dialect::tensor::OpDescriptor::Constant(value) = descriptor.op else {
        panic!("missing constant")
    };
    assert_eq!(value.as_typed::<PcuU512>().unwrap().data(), [expected; 2]);
    let uniform = capture.uniform_like(anchor, expected).unwrap();
    let descriptor = capture.graph.node(uniform.value.erase()).unwrap();
    assert_eq!(descriptor.shape, [2, 3]);
    assert_eq!(descriptor.numerical_options, options);
    assert_eq!(descriptor.numerical_mode, None);
    assert_eq!(descriptor.float_underflow_policy, None);
    let crate::dialect::tensor::OpDescriptor::Uniform { value } = descriptor.op else {
        panic!("missing uniform")
    };
    assert_eq!(value.as_typed::<PcuU512>().unwrap(), expected);
    let (mut foreign, _) =
        PcuTensorGraphCapture::new::<PcuU512, 1>([PcuSourceShape::Slice { length: 6 }]).unwrap();
    assert!(matches!(
        foreign.uniform_like(anchor, expected),
        Err(PcuExecutionError::InvalidTensorSourcePlan)
    ));
}

#[test]
fn immutable_matrix_shape_row_order_and_exceptional_transport_are_exact() {
    let (mut capture, []) = PcuTensorGraphCapture::new::<f32, 0>([]).unwrap();
    let payload = [
        [f32::from_bits(0x7f80_0042), -0.0, f32::from_bits(1)],
        [1.0, -2.0, 3.0],
    ];
    let matrix = capture.constant_array(&payload).unwrap();
    let node = capture.graph.node(matrix.value.erase()).unwrap();
    assert_eq!(node.shape, [2, 3]);
    assert_eq!(node.numerical_mode, None);
    assert_eq!(node.float_underflow_policy, None);
    let crate::dialect::tensor::OpDescriptor::Constant(value) = node.op else {
        panic!("missing matrix constant")
    };
    let bits: Vec<_> = value
        .as_typed::<f32>()
        .unwrap()
        .data()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(
        bits,
        [
            0x7f80_0042,
            0x8000_0000,
            1,
            0x3f80_0000,
            0xc000_0000,
            0x4040_0000
        ]
    );
}
