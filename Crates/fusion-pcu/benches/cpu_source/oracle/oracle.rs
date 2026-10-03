//! Complete correctness, terminal fault, rollback, tail and retry checks outside timing.
use super::{native, setup};
use native::Error;

#[path = "helper_integer/helper_integer.rs"]
pub mod helper_integer;

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

pub fn composition<const N: usize, const ORDERED: bool>(
    mut call: impl FnMut(&[f32; N], &mut [f32], &mut [f32]) -> Result<(), Error>,
) {
    let mut input = setup::array::<f32, N>(1.0);
    let mut stage = vec![42.0; N + 2];
    let mut output = vec![42.0; N + 2];
    for value in [1.0, 2.0, 4.0] {
        input.fill(value);
        call(&input, &mut stage, &mut output).unwrap();
        assert_eq!(&output[..N], &vec![(value + value) * value; N]);
        assert_eq!(&output[N..], &[42.0; 2]);
        if ORDERED {
            assert_eq!(&stage[..N], &vec![value + value; N]);
        } else {
            assert_eq!(stage, vec![42.0; N + 2]);
        }
        assert_eq!(&stage[N..], &[42.0; 2]);
    }
    let before_stage = stage.clone();
    let before_output = output.clone();
    input[N / 2] = f32::INFINITY;
    let fault = call(&input, &mut stage, &mut output).unwrap_err();
    assert!(matches!(fault, Error::Fault(fault)
        if fault.kind == fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand
        && fault.invocation_id == u64::try_from(N / 2).unwrap() && !fault.recovered));
    assert_eq!(stage, before_stage);
    assert_eq!(output, before_output);
    input[N / 2] = 1.0;
    call(&input, &mut stage, &mut output).unwrap();
    let before_stage = stage.clone();
    let before_output = output.clone();
    assert_eq!(
        call(&input, &mut stage, &mut output[..N - 1]),
        Err(Error::Schema)
    );
    assert_eq!(stage, before_stage);
    assert_eq!(output, before_output);
    if ORDERED {
        assert_eq!(
            call(&input, &mut stage[..N - 1], &mut output),
            Err(Error::Schema)
        );
        assert_eq!(stage, before_stage);
        assert_eq!(output, before_output);
    }
}
