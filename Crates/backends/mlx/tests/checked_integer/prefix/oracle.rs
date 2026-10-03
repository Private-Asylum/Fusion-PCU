//! Existing core integer oracle is independent of the native full-capacity primitive.
#[rustfmt::skip]
pub use super::super::support::{
    expected,
    same as compare,
};
use super::Sample;
pub fn bank<T: Sample>(phase: usize) -> (Vec<T>, Vec<T>) {
    let mut left = vec![T::max(); 8];
    let mut right = vec![T::max(); 11];
    left[..5].fill(T::small(4));
    right[..5].fill(T::small(2));
    match phase {
        0 => {}
        1 => {
            left[0] = T::max();
            right[0] = T::small(1);
        }
        2 => {
            left[0] = T::max();
            right[0] = T::max();
        }
        3 => {
            left[0] = T::min();
            right[0] = T::small(1);
        }
        4 => {
            left[0] = T::max();
            right[0] = T::min();
        }
        5 => {
            left[0] = T::min();
            right[0] = T::small(2);
        }
        6 | 7 => {
            for (lane, (a, b)) in left[..5].iter_mut().zip(&mut right[..5]).enumerate() {
                let mut raw = vec![0; T::ENCODED_SIZE];
                for (index, byte) in raw.iter_mut().enumerate() {
                    *byte = u8::try_from((index * 37 + lane * 19 + phase * 11) % 256).unwrap();
                }
                *a = T::raw(&raw);
                raw.reverse();
                *b = T::raw(&raw);
            }
        }
        _ => unreachable!(),
    }
    (left, right)
}
