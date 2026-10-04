//! Genuine immutable low-format producers with requested policies retained by math.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
use super::oracle::Format;
pub trait ProducerSource: Format {
    fn capture<const N: usize, const HALF: bool>(
        mode: pcu_facade::PcuNumericalMode,
        options: pcu_facade::PcuNumericalOptions,
        policy: pcu_facade::PcuFloatUnderflowPolicy,
    ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError>;
    fn pipeline<const N: usize, const HALF: bool>(
        input: &[Self],
    ) -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn literal<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn uniform<const N: usize, const HALF: bool>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn unused_owner<const N: usize>(
        unused: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError>;
}
macro_rules! width {
    ($module:ident,$ty:ty,$one:expr,$fraction:expr,$sign:expr) => {
        mod $module {
            use super::*;
            const fn payload<const N: usize>() -> [$ty; N] {
                let mut data = [<$ty>::from_bits(0); N];
                let mut i = 0;
                while i < N {
                    data[i] = <$ty>::from_bits(match i % 3 {
                        0 => $one,
                        1 => 1,
                        _ => $sign,
                    });
                    i += 1;
                }
                data
            }
            const fn factor<const HALF: bool>() -> $ty {
                <$ty>::from_bits(if HALF {
                    $one - (1 << $fraction)
                } else {
                    $one + (1 << $fraction)
                })
            }
            #[pcu(crate_path=::pcu_facade)]
            fn pipeline_half<const N: usize>(
                input: &[$ty],
            ) -> Result<PcuTensor<$ty>, PcuExecutionError> {
                let constant = pcu::constant(const { payload::<N>() })?;
                let uniform = pcu::uniform_like(input, const { factor::<true>() })?;
                let sum = pcu::add(input, &constant)?;
                pcu::mul(&sum, &uniform)
            }
            #[pcu(crate_path=::pcu_facade)]
            fn pipeline_double<const N: usize>(
                input: &[$ty],
            ) -> Result<PcuTensor<$ty>, PcuExecutionError> {
                let constant = pcu::constant(const { payload::<N>() })?;
                let uniform = pcu::uniform_like(input, const { factor::<false>() })?;
                let sum = pcu::add(input, &constant)?;
                pcu::mul(&sum, &uniform)
            }
            #[pcu(crate_path=::pcu_facade)]
            fn literal<const N: usize>() -> Result<PcuTensor<$ty>, PcuExecutionError> {
                pcu::constant(const { payload::<N>() })
            }
            #[pcu(crate_path=::pcu_facade)]
            fn uniform_half<const N: usize>() -> Result<PcuTensor<$ty>, PcuExecutionError> {
                let shape = pcu::constant(const { payload::<N>() })?;
                pcu::uniform_like(&shape, const { factor::<true>() })
            }
            #[pcu(crate_path=::pcu_facade)]
            fn uniform_double<const N: usize>() -> Result<PcuTensor<$ty>, PcuExecutionError> {
                let shape = pcu::constant(const { payload::<N>() })?;
                pcu::uniform_like(&shape, const { factor::<false>() })
            }
            #[pcu(crate_path=::pcu_facade)]
            fn foreign<const N: usize>(
                _unused: &PcuTensor<$ty>,
            ) -> Result<PcuTensor<$ty>, PcuExecutionError> {
                pcu::constant(const { payload::<N>() })
            }
            impl ProducerSource for $ty {
                fn capture<const N: usize, const HALF: bool>(
                    mode: pcu_facade::PcuNumericalMode,
                    options: pcu_facade::PcuNumericalOptions,
                    policy: pcu_facade::PcuFloatUnderflowPolicy,
                ) -> Result<pcu_facade::global::PcuCapturedTensorProgram, PcuExecutionError>
                {
                    pcu_facade::global::__pcu_capture_tensor_program::<Self, 1, _>(
                        [pcu_facade::global::PcuSourceShape::Slice { length: N }],
                        policy,
                        mode,
                        options,
                        if HALF {
                            pipeline_half::__pcu_capture_entry::<N>
                        } else {
                            pipeline_double::__pcu_capture_entry::<N>
                        },
                    )
                }
                fn pipeline<const N: usize, const HALF: bool>(
                    input: &[Self],
                ) -> Result<PcuTensor<Self>, PcuExecutionError> {
                    if HALF {
                        pipeline_half::<N>(input)
                    } else {
                        pipeline_double::<N>(input)
                    }
                }
                fn literal<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError> {
                    literal::<N>()
                }
                fn uniform<const N: usize, const HALF: bool>()
                -> Result<PcuTensor<Self>, PcuExecutionError> {
                    if HALF {
                        uniform_half::<N>()
                    } else {
                        uniform_double::<N>()
                    }
                }
                fn unused_owner<const N: usize>(
                    unused: &PcuTensor<Self>,
                ) -> Result<PcuTensor<Self>, PcuExecutionError> {
                    foreign::<N>(unused)
                }
            }
        }
    };
}
width!(binary16, PcuF16Bits, 0x3c00, 10, 0x8000);
width!(brain16, PcuBf16Bits, 0x3f80, 7, 0x8000);
width!(e4m3fn, PcuF8E4M3FnBits, 0x38, 3, 0x80);
width!(e5m2, PcuF8E5M2Bits, 0x3c, 2, 0x80);
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn overwrite<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0];
}
