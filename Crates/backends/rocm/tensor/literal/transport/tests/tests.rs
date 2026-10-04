//! Borrowed storage is safe only for exact initialized dense representations.
#[rustfmt::skip]
use super::{
    Payload,
    payload,
};
#[rustfmt::skip]
use super::super::super::{
    Graph,
    RocmPhysicalLayout,
    Tensor,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF16Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuI512,
    PcuScalarType,
    PcuU256,
    PcuU512,
    dialect::tensor::TensorElement,
};

fn check<T: TensorElement>(values: [T; 2], expected: &[u8]) {
    let mut graph = Graph::default();
    let constant = graph.constant_typed(Tensor::new([1, 2], values.to_vec()).unwrap());
    let node = graph.node(constant.erase()).unwrap();
    let layout = RocmPhysicalLayout::dense(u64::try_from(expected.len()).unwrap());
    let result = payload(node, layout).unwrap();
    assert!(matches!(result, Payload::Borrowed(_)));
    assert_eq!(result.bytes(), expected);
    // Verify this is the graph's own immutable initialized slice, not a copy.
    let super::super::OpDescriptor::Constant(value) = node.op else {
        panic!("constant node required");
    };
    assert_eq!(
        result.bytes().as_ptr(),
        value.as_typed::<T>().unwrap().data().as_ptr().cast()
    );
    assert!(payload(node, RocmPhysicalLayout::uniform_scalar()).is_err());
    assert!(payload(node, RocmPhysicalLayout::dense(layout.physical_bytes - 1)).is_err());
    let mut malformed = node;
    malformed.shape = &[2]; // Same byte count, wrong logical shape.
    assert!(payload(malformed, layout).is_err());
    malformed.shape = &[usize::MAX, 2];
    assert!(payload(malformed, layout).is_err());
    malformed = node;
    malformed.scalar_type = PcuScalarType::F32;
    assert!(payload(malformed, layout).is_err());

    let uniform = graph.uniform_typed([1, 2], values[1]).unwrap();
    let encoded = payload(graph.node(uniform.erase()).unwrap(), layout).unwrap();
    assert!(matches!(encoded, Payload::Encoded(_)));
    assert_eq!(encoded.bytes(), &expected[expected.len() / 2..].repeat(2));
}

#[test]
fn integer_constants_borrow_extrema_in_all_fourteen_carriers() {
    macro_rules! widths {
        ($($ty:ty),+ $(,)?) => { $(
            check([<$ty>::MIN, <$ty>::MAX],
                &[<$ty>::MIN.to_le_bytes(), <$ty>::MAX.to_le_bytes()].concat());
        )+ };
    }
    widths!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);
    check(
        [PcuU256::ZERO, PcuU256::MAX],
        &[vec![0; 32], vec![255; 32]].concat(),
    );
    check(
        [PcuU512::ZERO, PcuU512::MAX],
        &[vec![0; 64], vec![255; 64]].concat(),
    );
    let signed256 = [vec![0; 31], vec![128], vec![255; 31], vec![127]].concat();
    let signed512 = [vec![0; 63], vec![128], vec![255; 63], vec![127]].concat();
    check([PcuI256::MIN, PcuI256::MAX], &signed256);
    check([PcuI512::MIN, PcuI512::MAX], &signed512);
}

#[test]
fn floating_constants_borrow_exceptional_payloads_without_arithmetic() {
    let words64 = [0x8000_0000_0000_0000_u64, 0x7ff0_0000_0000_1234];
    check(
        words64.map(f64::from_bits),
        &words64.map(u64::to_le_bytes).concat(),
    );
    check(
        [PcuF16Bits::from_bits(0x8000), PcuF16Bits::from_bits(0x7c35)],
        &[0, 128, 0x35, 0x7c],
    );
    check(
        [PcuBf16Bits::from_bits(1), PcuBf16Bits::from_bits(0x7f93)],
        &[1, 0, 0x93, 0x7f],
    );
    check(
        [
            PcuF8E4M3FnBits::from_bits(0x80),
            PcuF8E4M3FnBits::from_bits(0xff),
        ],
        &[0x80, 0xff],
    );
    check(
        [PcuF8E5M2Bits::from_bits(1), PcuF8E5M2Bits::from_bits(0x7d)],
        &[1, 0x7d],
    );
    let words128 = [[0x1234, 0x7fff_0000_0000_0000], [0, 0x8000_0000_0000_0000]];
    check(
        words128.map(PcuF128Bits::from_limbs_le),
        &words128
            .into_iter()
            .flatten()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let words256 = [[0x5678, 0, 0, 0x7fff_f000_0000_0000], [1, 0, 0, 0]];
    check(
        words256.map(PcuF256Bits::from_limbs_le),
        &words256
            .into_iter()
            .flatten()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
    );
}
