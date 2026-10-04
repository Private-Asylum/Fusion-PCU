//! Reciprocal zero-signature raw authoring; imported fixtures retain exact provenance.
#[cfg(test)]
use fusion_pcu::pcu;
#[path = "../../../../cuda/benches/raw_low_float_producers/source/source.rs"]
#[allow(dead_code)] // Both low and wide companions export matching authoring/ownership helpers.
mod low;
#[path = "../../../../cuda/benches/raw_float_tensor_producers/source/source.rs"]
#[allow(dead_code)] // This provider adapts source provenance, not CUDA execution or arithmetic.
mod wide;
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuFloatUnderflowPolicy,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
};
use fusion_pcu::dialect::tensor::TensorElement;
pub trait Raw: PcuScalar + TensorElement {
    const LABEL: &'static str;
    const SENTINEL: Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn literal<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn uniform<const N: usize>() -> Result<PcuTensor<Self>, PcuExecutionError>;
    fn capture<const N: usize>(
        mode: PcuNumericalMode,
        options: PcuNumericalOptions,
        uf: PcuFloatUnderflowPolicy,
    );
}
macro_rules! carrier {($module:ident,$($ty:ty),+) => {$(impl Raw for $ty {
    const LABEL:&'static str=<$ty as $module::RawSource>::LABEL;
    const SENTINEL:Self=<$ty as $module::RawSource>::SENTINEL;
    fn from_bytes(bytes:&[u8])->Self{<$ty as $module::RawSource>::from_bytes(bytes)}
    fn literal<const N:usize>()->Result<PcuTensor<Self>,PcuExecutionError>{<$ty as $module::RawSource>::literal::<N>()}
    fn uniform<const N:usize>()->Result<PcuTensor<Self>,PcuExecutionError>{<$ty as $module::RawSource>::uniform::<N>()}
    fn capture<const N:usize>(mode:PcuNumericalMode,options:PcuNumericalOptions,uf:PcuFloatUnderflowPolicy){<$ty as $module::RawSource>::capture::<N>(mode,options,uf);}
})+};}
carrier!(low, PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits);
carrier!(wide, PcuF128Bits, PcuF256Bits);
pub fn overwrite<T: Raw, const N: usize>(output: &mut PcuTensor<T>) {
    if N == 6 {
        low::overwrite_matrix(&[[T::SENTINEL; 3]; 2], output).unwrap();
    } else {
        low::overwrite::<T, N>(&[T::SENTINEL], output).unwrap();
    }
}
pub fn consume<T: Raw>(owner: PcuTensor<T>) -> PcuTensor<T> {
    low::consume(owner).unwrap()
}
#[cfg(test)]
#[pcu]
pub fn identity<T: PcuScalar>(owner: &PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(owner)
}

pub trait ForeignSource: Raw {
    #[cfg(test)]
    fn unused<const N: usize>(
        owner: &PcuTensor<Self>,
    ) -> Result<PcuTensor<Self>, PcuExecutionError>;
}
macro_rules! foreign {
    ($name:ident,$ty:ty) => {
        #[cfg(test)]
        #[pcu]
        fn $name<const N: usize>(
            _owner: &PcuTensor<$ty>,
        ) -> Result<PcuTensor<$ty>, PcuExecutionError> {
            pcu::constant(const { [<$ty as Raw>::SENTINEL; N] })
        }
        impl ForeignSource for $ty {
            #[cfg(test)]
            fn unused<const N: usize>(
                owner: &PcuTensor<Self>,
            ) -> Result<PcuTensor<Self>, PcuExecutionError> {
                $name::<N>(owner)
            }
        }
    };
}
foreign!(unused_f16, PcuF16Bits);
foreign!(unused_bf16, PcuBf16Bits);
foreign!(unused_e4m3fn, PcuF8E4M3FnBits);
foreign!(unused_e5m2, PcuF8E5M2Bits);
foreign!(unused_f128, PcuF128Bits);
foreign!(unused_f256, PcuF256Bits);
