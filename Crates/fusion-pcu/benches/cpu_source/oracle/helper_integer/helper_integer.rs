//! Integer helper correctness and host fault publication outside measurement.
#[rustfmt::skip]
use super::super::{
    native::Error,
    setup,
};
use fusion_pcu::PcuExecutionFaultKind;

pub fn verify<const N: usize>(
    mut call: impl FnMut(&[u64; N], &u64, &mut [u64]) -> Result<(), Error>,
) {
    let mut input = setup::array::<u64, N>(1);
    let mut output = vec![17; N + 2];
    for phase in [0, 17, 83] {
        for (index, value) in input.iter_mut().enumerate() {
            *value = 1 + (u64::try_from(index).unwrap() + phase) % 97;
        }
        call(&input, &1, &mut output).unwrap();
        for (&actual, &value) in output[..N].iter().zip(input.iter()) {
            assert_eq!(actual, value * value);
        }
        assert_eq!(output[N..], [17; 2]);
    }
    let before = output.clone();
    input[N / 2] = u64::MAX;
    let error = call(&input, &1, &mut output).unwrap_err();
    assert!(matches!(error, Error::Fault(fault)
        if fault.kind == PcuExecutionFaultKind::ArithmeticOverflow
            && fault.invocation_id == u64::try_from(N / 2).unwrap() && !fault.recovered));
    assert_eq!(output, before);
    input[N / 2] = 2;
    call(&input, &1, &mut output).unwrap();
    let before = output.clone();
    assert_eq!(call(&input, &1, &mut output[..N - 1]), Err(Error::Schema));
    assert_eq!(output, before);
}
