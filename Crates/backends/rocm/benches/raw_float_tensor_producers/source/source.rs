//! Inline-const raw float encodings; the payload is never evaluated as arithmetic.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuF128Bits,
    PcuF256Bits,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuScalar,
    PcuTensor,
};
use fusion_pcu::dialect::tensor::TensorElement;
pub trait RawSource: PcuScalar + TensorElement {
    const LABEL: &'static str;
    const SENTINEL: Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn literal<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn uniform<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn capture<const N: usize>(
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        underflow: PcuFloatUnderflowPolicy,
    );
}
const fn payload128<const N: usize>() -> [PcuF128Bits; N] {
    let patterns = [
        [0x42, 0x7fff_0000_0000_0000],
        [0, 0x8000_0000_0000_0000],
        [1, 0],
        [0xdead_beef, 0xffff_8000_0000_0000],
        [u64::MAX, 0x3ffe_0123_4567_89ab],
        [0, 0x7fff_0000_0000_0000],
    ];
    let mut output = [PcuF128Bits::from_limbs_le([0; 2]); N];
    let mut index = 0;
    while index < N {
        output[index] = PcuF128Bits::from_limbs_le(patterns[index % 6]);
        index += 1;
    }
    output
}
const fn payload256<const N: usize>() -> [PcuF256Bits; N] {
    let patterns = [
        [0x42, 0, 0, 0x7fff_f000_0000_0000],
        [0, 0, 0, 0x8000_0000_0000_0000],
        [1, 0, 0, 0],
        [0xdead_beef, 1, 2, 0xffff_f800_0000_0000],
        [
            u64::MAX,
            0x1234_5678_9abc_def0,
            0xfedc_ba98_7654_3210,
            0x3fff_e012_3456_789a,
        ],
        [0, 0, 0, 0x7fff_f000_0000_0000],
    ];
    let mut output = [PcuF256Bits::from_limbs_le([0; 4]); N];
    let mut index = 0;
    while index < N {
        output[index] = PcuF256Bits::from_limbs_le(patterns[index % 6]);
        index += 1;
    }
    output
}
#[pcu]
fn literal128<const N: usize>() -> Result<PcuTensor<PcuF128Bits>, PcuExecutionError> {
    pcu::constant(const { payload128::<N>() })
}
#[pcu]
fn uniform128<const N: usize>() -> Result<PcuTensor<PcuF128Bits>, PcuExecutionError> {
    let shape = pcu::constant(const { payload128::<N>() })?;
    pcu::uniform_like(
        &shape,
        const { PcuF128Bits::from_limbs_le([0x42, 0x7fff_0000_0000_0000]) },
    )
}
#[pcu]
fn literal256<const N: usize>() -> Result<PcuTensor<PcuF256Bits>, PcuExecutionError> {
    pcu::constant(const { payload256::<N>() })
}
#[pcu]
fn uniform256<const N: usize>() -> Result<PcuTensor<PcuF256Bits>, PcuExecutionError> {
    let shape = pcu::constant(const { payload256::<N>() })?;
    pcu::uniform_like(
        &shape,
        const { PcuF256Bits::from_limbs_le([0x42, 0, 0, 0x7fff_f000_0000_0000]) },
    )
}
#[pcu]
fn matrix128() -> Result<PcuTensor<PcuF128Bits>, PcuExecutionError> {
    pcu::constant(
        const {
            [
                [
                    payload128::<6>()[0],
                    payload128::<6>()[1],
                    payload128::<6>()[2],
                ],
                [
                    payload128::<6>()[3],
                    payload128::<6>()[4],
                    payload128::<6>()[5],
                ],
            ]
        },
    )
}
#[pcu]
fn matrix_uniform128() -> Result<PcuTensor<PcuF128Bits>, PcuExecutionError> {
    let shape = matrix128()?;
    pcu::uniform_like(
        &shape,
        const { PcuF128Bits::from_limbs_le([0x42, 0x7fff_0000_0000_0000]) },
    )
}
#[pcu]
fn matrix256() -> Result<PcuTensor<PcuF256Bits>, PcuExecutionError> {
    pcu::constant(
        const {
            [
                [
                    payload256::<6>()[0],
                    payload256::<6>()[1],
                    payload256::<6>()[2],
                ],
                [
                    payload256::<6>()[3],
                    payload256::<6>()[4],
                    payload256::<6>()[5],
                ],
            ]
        },
    )
}
#[pcu]
fn matrix_uniform256() -> Result<PcuTensor<PcuF256Bits>, PcuExecutionError> {
    let shape = matrix256()?;
    pcu::uniform_like(
        &shape,
        const { PcuF256Bits::from_limbs_le([0x42, 0, 0, 0x7fff_f000_0000_0000]) },
    )
}
macro_rules! carrier {
    ($scalar:ty, $label:literal, $sentinel:expr, $literal:ident, $uniform:ident, $matrix:ident, $matrix_uniform:ident) => {
        impl RawSource for $scalar {
            const LABEL: &'static str = $label;
            const SENTINEL: Self = $sentinel;
            fn from_bytes(bytes: &[u8]) -> Self {
                Self::decode_le(bytes.try_into().unwrap())
            }
            fn literal<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError> {
                if N == 6 { $matrix() } else { $literal::<N>() }
            }
            fn uniform<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError> {
                if N == 6 {
                    $matrix_uniform()
                } else {
                    $uniform::<N>()
                }
            }
            fn capture<const N: usize>(
                mode: PcuNumericalMode,
                options: PcuNumericalOptions,
                underflow: PcuFloatUnderflowPolicy,
            ) {
                let builders = if N == 6 {
                    [
                        $matrix::__pcu_capture_entry,
                        $matrix_uniform::__pcu_capture_entry,
                    ]
                } else {
                    [
                        $literal::__pcu_capture_entry::<N>,
                        $uniform::__pcu_capture_entry::<N>,
                    ]
                };
                let expected_shape = if N == 6 { vec![2, 3] } else { vec![N] };
                for builder in builders {
                    let captured = global::__pcu_capture_tensor_program::<Self, 0, _>(
                        [],
                        underflow,
                        mode,
                        options,
                        builder,
                    )
                    .unwrap();
                    assert!(captured.argument_indices().is_empty());
                    assert!(captured.program().input_values().is_empty());
                    for &value in captured.program().selected_nodes() {
                        let node = captured.program().graph().node(value).unwrap();
                        assert_eq!(node.numerical_mode, None);
                        assert_eq!(node.float_underflow_policy, None);
                        assert_eq!(node.numerical_options, options);
                        assert_eq!(node.shape, expected_shape);
                    }
                }
            }
        }
    };
}
carrier!(
    PcuF128Bits,
    "f128",
    PcuF128Bits::from_limbs_le([0x1234, 0x3fff_0000_0000_0000]),
    literal128,
    uniform128,
    matrix128,
    matrix_uniform128
);
carrier!(
    PcuF256Bits,
    "f256",
    PcuF256Bits::from_limbs_le([0x1234, 0x5678, 0x9abc, 0x3fff_f000_0000_0000]),
    literal256,
    uniform256,
    matrix256,
    matrix_uniform256
);
#[pcu(invocations = N)]
pub fn overwrite<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let index = pcu::context::global_invocation_id();
    output[index] = input[0];
}
#[pcu]
pub fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(invocations = 6)]
pub fn overwrite_matrix<T: PcuScalar>(input: &[[T; 3]; 2], output: &mut [[T; 3]; 2]) {
    let id = pcu::context::global_invocation_id();
    let row = id / 3;
    let column = id % 3;
    output[row][column] = input[row][column];
}
