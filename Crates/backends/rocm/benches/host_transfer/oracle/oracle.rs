//! Input refresh and complete verification stay outside the reported timing interval.
pub const SENTINEL: u8 = 0xa7;
pub const TAIL: usize = 16;

pub fn refresh(input: &mut [u8], sequence: u64) {
    // Eight byte lanes vary independently; every complete input has a unique call tag.
    let tag = sequence.to_le_bytes();
    for (chunk_index, chunk) in input.chunks_mut(8).enumerate() {
        let salt = u8::try_from(chunk_index % 251).expect("bounded pattern salt");
        for (byte, lane) in chunk.iter_mut().zip(tag) {
            *byte = lane.wrapping_add(salt);
        }
    }
}

pub fn verify(input: &[u8], output: &[u8]) {
    assert_eq!(&output[..input.len()], input, "complete identity output");
    assert!(output[input.len()..].iter().all(|&byte| byte == SENTINEL));
}
