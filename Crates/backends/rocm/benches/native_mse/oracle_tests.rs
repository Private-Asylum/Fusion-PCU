mod oracle;
#[rustfmt::skip]
use oracle::{
    expected,
    inputs,
    verify,
};
fn check<const N: usize>() {
    let mut previous = None;
    for phase in 0..3 {
        let (prediction, target) = inputs::<N>(phase);
        let sum: f64 = prediction
            .iter()
            .zip(target.iter())
            .map(|(left, right)| {
                let difference = f64::from(*left) - f64::from(*right);
                difference * difference
            })
            .sum();
        assert!(sum <= f64::from(1 << 24));
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        // Exact bounded sum/count.
        let reference = sum as f32 * (1.0 / N as f32);
        verify(expected::<N>(phase), &[reference]);
        assert_ne!(previous, Some(reference.to_bits()));
        previous = Some(reference.to_bits());
    }
}
#[test]
fn integer_oracle_matches_constructed_inputs_at_both_extents() {
    check::<65>();
    check::<1_048_576>();
}
