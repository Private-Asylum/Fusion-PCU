//! Exact logical bytes versus physical integer lanes; transport never admits arithmetic.
#[rustfmt::skip]
use fusion_pcu::PcuScalarType;
use crate::MlxError;
pub(in super::super) struct Carrier {
    pub(in super::super) logical_width: usize,
    pub(in super::super) count: usize,
    pub(in super::super) width: usize,
    pub(in super::super) dtype: i32,
    pub(in super::super) upload_format: u32,
    pub(in super::super) byte_len: usize,
}
impl Carrier {
    pub(in super::super) fn assess(scalar: PcuScalarType, count: usize) -> Result<Self, MlxError> {
        if matches!(
            scalar,
            PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4
        ) {
            return Err(MlxError::UnsupportedScalar(scalar));
        }
        let logical_width = usize::from(scalar.bit_width()) / 8;
        let (width, dtype, upload_format) = match logical_width {
            1 => (1, 1, 4),
            2 => (2, 2, 5),
            _ => (4, 3, 6),
        };
        let byte_len = count
            .checked_mul(logical_width)
            .ok_or(MlxError::InvalidExtent)?;
        let physical_count = byte_len / width;
        if count == 0
            || i32::try_from(physical_count).is_err()
            || isize::try_from(byte_len).is_err()
        {
            return Err(MlxError::InvalidExtent);
        }
        Ok(Self {
            logical_width,
            count: physical_count,
            width,
            dtype,
            upload_format,
            byte_len,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_twenty_two_exact_extents_and_physical_carrier_limits() {
        for scalar in PcuScalarType::ALL {
            if scalar.bit_width() < 8 {
                assert!(matches!(
                    Carrier::assess(scalar, 3),
                    Err(MlxError::UnsupportedScalar(_))
                ));
                continue;
            }
            let descriptor = Carrier::assess(scalar, 3).unwrap();
            assert_eq!(descriptor.byte_len, usize::from(scalar.bit_width()) / 8 * 3);
            assert_eq!(descriptor.count * descriptor.width, descriptor.byte_len);
            assert!(Carrier::assess(scalar, 0).is_err());
            assert!(Carrier::assess(scalar, usize::MAX).is_err());
            let limit =
                usize::try_from(i32::MAX).unwrap() / (descriptor.logical_width / descriptor.width);
            assert!(Carrier::assess(scalar, limit).is_ok());
            assert!(Carrier::assess(scalar, limit + 1).is_err());
        }
    }
}
