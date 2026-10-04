//! Genuine zero-argument integer producers, const-value cache identity and rank-two ownership.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuFloatUnderflowPolicy,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
};
use fusion_pcu::dialect::tensor::TensorElement;
pub trait RawSource: PcuScalar + TensorElement {
    const LABEL: &'static str;
    const SENTINEL: Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn literal<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn alternate<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn uniform<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn capture<const N: usize>(
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        uf: PcuFloatUnderflowPolicy,
    );
}
macro_rules! sources {
    ($module:ident,$ty:ty,$label:literal,$sentinel:expr,$pattern:expr,$alternate:expr) => {
        mod $module {
            use super::*;
            const PATTERN: [$ty; 3] = $pattern;
            const ALTERNATE: [$ty; 3] = $alternate;
            const fn payload<const N: usize, const VALUE: usize>() -> [$ty; N] {
                let mut values = [$sentinel; N];
                let mut index = 0;
                let pattern = if VALUE == 0 { PATTERN } else { ALTERNATE };
                while index < N {
                    values[index] = pattern[index % 3];
                    index += 1;
                }
                values
            }
            #[pcu(crate_path=::pcu_facade)]
            pub fn literal<const N: usize, const VALUE: usize>()
            -> Result<PcuTensor<$ty>, PcuExecutionError> {
                pcu::constant(const { payload::<N, VALUE>() })
            }
            #[pcu(crate_path=::pcu_facade)]
            pub fn nested<const N: usize, const VALUE: usize>()
            -> Result<PcuTensor<$ty>, PcuExecutionError> {
                literal::<N, VALUE>()
            }
            #[pcu(crate_path=::pcu_facade)]
            pub fn uniform<const N: usize>() -> Result<PcuTensor<$ty>, PcuExecutionError> {
                let shape = pcu::constant(const { payload::<N, 0>() })?;
                pcu::uniform_like(&shape, const { PATTERN[0] })
            }
            #[pcu(crate_path=::pcu_facade)]
            pub fn matrix<const VALUE: usize>() -> Result<PcuTensor<$ty>, PcuExecutionError> {
                pcu::constant(
                    const {
                        if VALUE == 0 {
                            [PATTERN, [PATTERN[2], PATTERN[1], PATTERN[0]]]
                        } else {
                            [ALTERNATE, [ALTERNATE[2], ALTERNATE[1], ALTERNATE[0]]]
                        }
                    },
                )
            }
            #[pcu(crate_path=::pcu_facade)]
            pub fn matrix_uniform() -> Result<PcuTensor<$ty>, PcuExecutionError> {
                let shape =
                    pcu::constant(const { [PATTERN, [PATTERN[2], PATTERN[1], PATTERN[0]]] })?;
                pcu::uniform_like(&shape, const { PATTERN[0] })
            }
        }
        impl RawSource for $ty {
            const LABEL: &'static str = $label;
            const SENTINEL: Self = $sentinel;
            fn from_bytes(bytes: &[u8]) -> Self {
                Self::decode_le(bytes.try_into().unwrap())
            }
            fn literal<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError> {
                if N == 6 {
                    $module::matrix::<0>()
                } else {
                    $module::nested::<N, 0>()
                }
            }
            fn alternate<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError> {
                if N == 6 {
                    $module::matrix::<1>()
                } else {
                    $module::nested::<N, 1>()
                }
            }
            fn uniform<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError> {
                if N == 6 {
                    $module::matrix_uniform()
                } else {
                    $module::uniform::<N>()
                }
            }
            fn capture<const N: usize>(
                mode: PcuNumericalMode,
                options: PcuNumericalOptions,
                uf: PcuFloatUnderflowPolicy,
            ) {
                let builders = if N == 6 {
                    [
                        $module::matrix::__pcu_capture_entry::<0>,
                        $module::matrix_uniform::__pcu_capture_entry,
                    ]
                } else {
                    [
                        $module::literal::__pcu_capture_entry::<N, 0>,
                        $module::uniform::__pcu_capture_entry::<N>,
                    ]
                };
                for builder in builders {
                    let captured = fusion_pcu::global::__pcu_capture_tensor_program::<Self, 0, _>(
                        [],
                        uf,
                        mode,
                        options,
                        builder,
                    )
                    .unwrap();
                    assert!(captured.input_values().is_empty());
                    assert!(captured.argument_indices().is_empty());
                    for &value in captured.program().selected_nodes() {
                        let node = captured.program().graph().node(value).unwrap();
                        assert_eq!(node.scalar_type, Self::TYPE);
                        assert_eq!(node.numerical_mode, None);
                        assert_eq!(node.float_underflow_policy, None);
                        assert_eq!(node.numerical_options, options);
                        assert_eq!(node.shape, if N == 6 { vec![2, 3] } else { vec![N] });
                    }
                }
            }
        }
    };
}
sources!(
    u8,
    u8,
    "u8",
    0,
    [(1_u8 << 7), u8::MAX, 7],
    [(1_u8 << 7) + 1, (u8::MAX - 1), 8]
);
sources!(i8, i8, "i8", 0, [i8::MIN, -1, 7], [i8::MIN + 1, -2, 8]);
sources!(
    u16,
    u16,
    "u16",
    0,
    [(1_u16 << 15), u16::MAX, 7],
    [(1_u16 << 15) + 1, (u16::MAX - 1), 8]
);
sources!(i16, i16, "i16", 0, [i16::MIN, -1, 7], [i16::MIN + 1, -2, 8]);
sources!(
    u32,
    u32,
    "u32",
    0,
    [(1_u32 << 31), u32::MAX, 7],
    [(1_u32 << 31) + 1, (u32::MAX - 1), 8]
);
sources!(i32, i32, "i32", 0, [i32::MIN, -1, 7], [i32::MIN + 1, -2, 8]);
sources!(
    u64,
    u64,
    "u64",
    0,
    [(1_u64 << 63), u64::MAX, 7],
    [(1_u64 << 63) + 1, (u64::MAX - 1), 8]
);
sources!(i64, i64, "i64", 0, [i64::MIN, -1, 7], [i64::MIN + 1, -2, 8]);
sources!(
    u128,
    u128,
    "u128",
    0,
    [(1_u128 << 127), u128::MAX, 7],
    [(1_u128 << 127) + 1, (u128::MAX - 1), 8]
);
sources!(
    i128,
    i128,
    "i128",
    0,
    [i128::MIN, -1, 7],
    [i128::MIN + 1, -2, 8]
);
sources!(
    u256,
    PcuU256,
    "u256",
    PcuU256::ZERO,
    [
        PcuU256::from_limbs_le([0, 0, 0, 1_u64 << 63]),
        PcuU256::from_limbs_le([u64::MAX, u64::MAX, u64::MAX, u64::MAX]),
        PcuU256::from_limbs_le([7, 0, 0, 0])
    ],
    [
        PcuU256::from_limbs_le([1, 0, 0, 1_u64 << 63]),
        PcuU256::from_limbs_le([u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX]),
        PcuU256::from_limbs_le([8, 0, 0, 0])
    ]
);
sources!(
    i256,
    PcuI256,
    "i256",
    PcuI256::ZERO,
    [
        PcuI256::from_limbs_le([0, 0, 0, 1_u64 << 63]),
        PcuI256::from_limbs_le([u64::MAX, u64::MAX, u64::MAX, u64::MAX]),
        PcuI256::from_limbs_le([7, 0, 0, 0])
    ],
    [
        PcuI256::from_limbs_le([1, 0, 0, 1_u64 << 63]),
        PcuI256::from_limbs_le([u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX]),
        PcuI256::from_limbs_le([8, 0, 0, 0])
    ]
);
sources!(
    u512,
    PcuU512,
    "u512",
    PcuU512::ZERO,
    [
        PcuU512::from_limbs_le([0, 0, 0, 0, 0, 0, 0, 1_u64 << 63]),
        PcuU512::from_limbs_le([
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX
        ]),
        PcuU512::from_limbs_le([7, 0, 0, 0, 0, 0, 0, 0])
    ],
    [
        PcuU512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 1_u64 << 63]),
        PcuU512::from_limbs_le([
            u64::MAX - 1,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX
        ]),
        PcuU512::from_limbs_le([8, 0, 0, 0, 0, 0, 0, 0])
    ]
);
sources!(
    i512,
    PcuI512,
    "i512",
    PcuI512::ZERO,
    [
        PcuI512::from_limbs_le([0, 0, 0, 0, 0, 0, 0, 1_u64 << 63]),
        PcuI512::from_limbs_le([
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX
        ]),
        PcuI512::from_limbs_le([7, 0, 0, 0, 0, 0, 0, 0])
    ],
    [
        PcuI512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 1_u64 << 63]),
        PcuI512::from_limbs_le([
            u64::MAX - 1,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX
        ]),
        PcuI512::from_limbs_le([8, 0, 0, 0, 0, 0, 0, 0])
    ]
);

#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn overwrite<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0];
}
#[pcu(crate_path=::pcu_facade,invocations=6)]
pub fn overwrite_matrix<T: PcuScalar>(input: &[[T; 3]; 2], output: &mut [[T; 3]; 2]) {
    let id = pcu::context::global_invocation_id();
    output[id / 3][id % 3] = input[id / 3][id % 3];
}
#[pcu(crate_path=::pcu_facade)]
pub fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
