//! Complete correctness, terminal fault, rollback, tail and retry checks outside timing.
use super::{native, setup};
use native::Error;

pub fn negate<const N: usize>(mut call: impl FnMut(&[f32; N], &mut [f32]) -> Result<(), Error>) {
    let mut input = setup::array::<f32, N>(0.0);
    for (index, value) in input.iter_mut().enumerate() {
        let bits = u32::try_from(index).unwrap().wrapping_mul(0x9e37_79b9);
        *value = f32::from_bits((bits & 0x807f_ffff) | 0x3f00_0000);
    }
    input[0] = -0.0;
    input[1] = f32::from_bits(1);
    let mut expected = std::vec![42.0; N + 2];
    let mut actual = expected.clone();
    native::negate(&input, &mut expected).unwrap();
    call(&input, &mut actual).unwrap();
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    let before = actual.clone();
    input[N / 2] = f32::NAN;
    assert_eq!(
        call(&input, &mut actual),
        native::negate(&input, &mut expected)
    );
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        before
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    input[N / 2] = 2.0;
    call(&input, &mut actual).unwrap();
    native::negate(&input, &mut expected).unwrap();
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(call(&input, &mut actual[..N - 1]), Err(Error::Schema));
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}
pub fn integer<const N: usize, const MUL: bool>(
    mut call: impl FnMut(&[u64; N], &[u64; N], &mut [u64]) -> Result<(), Error>,
) {
    let mut lhs = setup::array::<u64, N>(0);
    let mut rhs = setup::array::<u64, N>(0);
    for (index, (left, right)) in lhs.iter_mut().zip(rhs.iter_mut()).enumerate() {
        *left = u64::try_from(index % 97).unwrap() + 1;
        *right = u64::try_from(index % 7).unwrap() + 1;
    }
    let mut expected = std::vec![42; N + 2];
    let mut actual = expected.clone();
    native::integer::<N, MUL>(&lhs, &rhs, &mut expected).unwrap();
    call(&lhs, &rhs, &mut actual).unwrap();
    assert_eq!(actual, expected);
    let before = actual.clone();
    lhs[N / 2] = u64::MAX;
    rhs[N / 2] = 2;
    assert_eq!(
        call(&lhs, &rhs, &mut actual),
        native::integer::<N, MUL>(&lhs, &rhs, &mut expected)
    );
    assert_eq!(actual, before);
    lhs[N / 2] = 2;
    call(&lhs, &rhs, &mut actual).unwrap();
    native::integer::<N, MUL>(&lhs, &rhs, &mut expected).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(call(&lhs, &rhs, &mut actual[..N - 1]), Err(Error::Schema));
    assert_eq!(actual, expected);
}
