//! Independent base-256 signed-magnitude arithmetic; no core checked arithmetic is called.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedInteger,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
pub trait Wide: PcuCheckedInteger + core::fmt::Debug + Eq {
    const BYTES: usize;
    const SIGNED: bool;
    fn bytes(self) -> [u8; 64];
    fn from_bytes(bytes: [u8; 64]) -> Self;
}
macro_rules! primitive {
    ($ty:ty, $signed:expr, $n:expr) => {
        impl Wide for $ty {
            const BYTES: usize = $n;
            const SIGNED: bool = $signed;
            fn bytes(self) -> [u8; 64] {
                let mut bytes = [0; 64];
                bytes[..$n].copy_from_slice(&self.to_le_bytes());
                bytes
            }
            fn from_bytes(bytes: [u8; 64]) -> Self {
                let mut value = [0; $n];
                value.copy_from_slice(&bytes[..$n]);
                Self::from_le_bytes(value)
            }
        }
    };
}
primitive!(i8, true, 1);
primitive!(u8, false, 1);
primitive!(i16, true, 2);
primitive!(u16, false, 2);
primitive!(i32, true, 4);
primitive!(u32, false, 4);
primitive!(i64, true, 8);
primitive!(u64, false, 8);
primitive!(i128, true, 16);
primitive!(u128, false, 16);
macro_rules! carrier {
    ($ty:ty, $n:expr, $signed:expr) => {
        impl Wide for $ty {
            const BYTES: usize = $n * 8;
            const SIGNED: bool = $signed;
            fn bytes(self) -> [u8; 64] {
                let mut bytes = [0; 64];
                for (i, limb) in self.to_limbs_le().into_iter().enumerate() {
                    bytes[i * 8..i * 8 + 8].copy_from_slice(&limb.to_le_bytes());
                }
                bytes
            }
            fn from_bytes(bytes: [u8; 64]) -> Self {
                let mut limbs = [0; $n];
                for (i, limb) in limbs.iter_mut().enumerate() {
                    let mut encoded = [0; 8];
                    encoded.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
                    *limb = u64::from_le_bytes(encoded);
                }
                Self::from_limbs_le(limbs)
            }
        }
    };
}
carrier!(PcuI256, 4, true);
carrier!(PcuU256, 4, false);
carrier!(PcuI512, 8, true);
carrier!(PcuU512, 8, false);
pub fn small<T: Wide>(value: u8) -> T {
    let mut bytes = [0; 64];
    bytes[0] = value;
    T::from_bytes(bytes)
}
pub fn minimum<T: Wide>() -> T {
    let mut bytes = [0; 64];
    if T::SIGNED {
        bytes[T::BYTES - 1] = 128;
    }
    T::from_bytes(bytes)
}
pub fn maximum<T: Wide>() -> T {
    let mut bytes = [255; 64];
    if T::SIGNED {
        bytes[T::BYTES - 1] = 127;
    }
    T::from_bytes(bytes)
}
