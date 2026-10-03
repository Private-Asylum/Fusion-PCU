//! Independent finite small-integer byte oracle and retained offline range goldens.
#[path = "../../integer_clamp/oracle/oracle.rs"]
#[allow(dead_code)] // The integration fixture also consumes the range goldens.
mod golden;
#[rustfmt::skip]
pub use golden::{
    Format,
    verify,
};
pub fn rows<T: Format>(operation: u32) -> Vec<(T, T, T, u32)> {
    golden::rows(operation)
}
pub fn inputs<T: Format>(count: usize, phase: usize, kind: u32) -> (Vec<T>, Vec<T>, Vec<T>) {
    let base = 3 + phase % 2;
    let left = (0..count)
        .map(|i| {
            T::small(
                u8::try_from(if kind == 3 && i == 0 {
                    base
                } else if kind == 4 && i == 0 {
                    base + 3
                } else {
                    base + (i + phase) % 4
                })
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let right = (0..count)
        .map(|i| T::small(u8::try_from((i + phase) % 3).unwrap()))
        .collect::<Vec<_>>();
    let want = (0..count)
        .map(|i| {
            let x = u8::try_from(if kind == 3 && i == 0 {
                base
            } else if kind == 4 && i == 0 {
                base + 3
            } else {
                base + (i + phase) % 4
            })
            .unwrap();
            let y = u8::try_from((i + phase) % 3).unwrap();
            let seed = u8::try_from(if kind == 4 { base + 3 } else { base }).unwrap();
            let result = match kind {
                0 => i16::from(x) * 2,
                1 => i16::from(x) * i16::from(x),
                2 => i16::from(x) - i16::from(y),
                3 => i16::from(x) - i16::from(seed),
                4 => i16::from(seed) - i16::from(x),
                _ => unreachable!(),
            };
            let mut bytes = vec![if result < 0 { 255 } else { 0 }; T::ENCODED_SIZE];
            let low = result.to_le_bytes();
            bytes[0] = low[0];
            if bytes.len() > 1 {
                bytes[1] = low[1];
            }
            T::from_bytes(&bytes)
        })
        .collect();
    (left, right, want)
}
