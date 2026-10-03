//! Real native carrier owners preserve bits, tails, session affinity and detached lifetimes.
#[path = "../scalar_transport/device/device.rs"]
mod device;
#[path = "../scalar_transport/sample/sample.rs"]
mod sample;
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuF16Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanError,
};
use sample::{same, Sample};

fn carrier<T: Sample>(backend: &PcuVulkanBackend, foreign: &PcuVulkanBackend) {
    let sentinel = T::pattern(91);
    for extent in [0, 1, 7, 65] {
        let mut input: Vec<T> = (0..extent).map(|index| T::pattern(index * 37)).collect();
        let expected = input.clone();
        let owner = backend.upload_owned(&input).unwrap();
        let mut prepared = backend.prepare_carrier_copy::<T>(extent).unwrap();
        assert_eq!(owner.len(), extent);
        assert_eq!(owner.is_empty(), extent == 0);
        assert!(backend.owns_buffer(&owner));
        assert!(!foreign.owns_buffer(&owner)); // Same physical GPU is not the same logical session.
        input.fill(sentinel);
        let copy = owner.copy_owned().unwrap();
        let sibling = owner.copy_owned().unwrap();
        assert!(owner.same_session(&copy));
        assert!(copy.same_session(&sibling));
        drop(owner);
        let mut output = vec![sentinel; extent + 3];
        copy.read_into(&mut output).unwrap();
        same(&output[..extent], &expected);
        same(&output[extent..], &[sentinel; 3]);
        if extent != 0 {
            let mut short = vec![sentinel; extent - 1];
            assert!(matches!(
                copy.read_into(&mut short),
                Err(PcuVulkanError::BufferTooSmall)
            ));
            same(&short, &vec![sentinel; extent - 1]);
        }
        drop(copy);
        sibling.read_into(&mut output).unwrap();
        same(&output[..extent], &expected);
        let copied = prepared.copy_owned(&sibling).unwrap();
        copied.read_into(&mut output).unwrap();
        same(&output[..extent], &expected);
        let foreign_owner = foreign.upload_owned(&expected).unwrap();
        assert!(matches!(
            prepared.copy_owned(&foreign_owner),
            Err(PcuVulkanError::InvalidArguments)
        ));
        if extent != 0 {
            assert!(matches!(
                prepared.copy_host(&expected[..extent - 1]),
                Err(PcuVulkanError::BufferTooSmall)
            ));
        }
        for phase in 0..64 {
            for (index, value) in input.iter_mut().enumerate() {
                *value = T::pattern(phase * 79 + index);
            }
            let escaped = prepared.copy_host(&input).unwrap();
            escaped.read_into(&mut output).unwrap();
            same(&output[..extent], &input);
            same(&output[extent..], &[sentinel; 3]);
            copied.read_into(&mut output).unwrap();
            same(&output[..extent], &expected); // Fresh output never aliases earlier escaped data.
        }
    }
}

#[test]
#[ignore = "requires an actual Vulkan compute GPU"]
fn twenty_two_native_owners_and_word_padded_transfers() {
    let (backend, _) = device::selected();
    let foreign = PcuVulkanBackend::new().unwrap();
    macro_rules! carriers { ($($ty:ty),+ $(,)?) => {$(carrier::<$ty>(&backend, &foreign);)+}; }
    carriers!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        PcuI256,
        PcuU256,
        PcuI512,
        PcuU512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
    let raw = [f64::from_bits(0x7ff0_0000_0000_0001), -0.0];
    let owner = backend.upload_owned(&raw).unwrap();
    drop(backend);
    drop(foreign);
    let copy = owner.copy_owned().unwrap();
    drop(owner);
    let mut output = [0.0; 3];
    copy.read_into(&mut output).unwrap();
    same(&output[..2], &raw);
    assert_eq!(output[2].to_bits(), 0);
}
