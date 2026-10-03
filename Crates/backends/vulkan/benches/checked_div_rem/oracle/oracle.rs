//! Independent quotient/remainder arithmetic widened to i128/u128, with explicit narrow MIN/-1 detection.
#[rustfmt::skip]
use pcu_facade::PcuScalar;
pub trait Integer: PcuScalar + Default + Eq + std::fmt::Debug {
    const SIGNED: bool;
    fn raw(self) -> u64;
    fn from_raw(raw: u64) -> Self;
}
macro_rules! types{($($t:ty:$signed:literal),+)=>{$(impl Integer for $t{const SIGNED:bool=$signed;fn raw(self)->u64{let encoded=self.encode_le();let mut bytes=[0;8];bytes[..Self::HOST_SIZE].copy_from_slice(encoded.as_ref());u64::from_le_bytes(bytes)}#[allow(clippy::cast_possible_truncation,clippy::cast_possible_wrap)]fn from_raw(raw:u64)->Self{raw as Self}})+};}
types!(i8:true,u8:false,i16:true,u16:false,i32:true,u32:false,i64:true,u64:false);
pub fn expected<T: Integer>(a: T, b: T) -> (T, T, u32) {
    let a = a.raw();
    let b = b.raw();
    let bits = u32::from(T::TYPE.bit_width());
    let mask = u64::MAX >> (64 - bits);
    let min = 1_u64 << (bits - 1);
    if b == 0 {
        return (T::default(), T::default(), 4);
    }
    if T::SIGNED && a == min && b == mask {
        return (T::default(), T::default(), 5);
    }
    let (q, r) = if T::SIGNED {
        let signed = |v| {
            if v & min != 0 {
                i128::from(v) - (1_i128 << bits)
            } else {
                i128::from(v)
            }
        };
        let a = signed(a);
        let b = signed(b);
        let q = a / b;
        let r = a % b;
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let values = (q as u64, r as u64);
        values
    } else {
        (
            u64::try_from(u128::from(a) / u128::from(b)).unwrap(),
            u64::try_from(u128::from(a) % u128::from(b)).unwrap(),
        )
    };
    (T::from_raw(q), T::from_raw(r), 0)
}
