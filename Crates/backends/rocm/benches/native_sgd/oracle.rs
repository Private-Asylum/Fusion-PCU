//! Independently evaluated full-lane oracle, including a rounding witness in each fixture.
pub const RATE: f32 = 1.000_000_1_f32;

pub fn inputs<const N: usize>(phase: u32) -> (Box<[f32; N]>, Box<[f32; N]>) {
    let weights: Vec<_> = (0..N)
        .map(|lane| {
            if lane.is_multiple_of(17) {
                1.0
            } else {
                f32::from(i16::try_from(lane % 127).unwrap())
                    .mul_add(0.25, f32::from(u16::try_from(phase).unwrap()))
            }
        })
        .collect();
    let gradient: Vec<_> = (0..N)
        .map(|lane| {
            if lane.is_multiple_of(17) {
                f32::from_bits(0x3f7f_fffe)
            } else {
                f32::from(i16::try_from(lane % 31).unwrap())
                    .mul_add(0.125, -f32::from(u16::try_from(phase).unwrap()))
            }
        })
        .collect();
    (
        weights.into_boxed_slice().try_into().unwrap(),
        gradient.into_boxed_slice().try_into().unwrap(),
    )
}

pub fn expected(weights: &[f32], gradient: &[f32], contracted: bool) -> Vec<f32> {
    weights
        .iter()
        .zip(gradient)
        .map(|(&weight, &gradient)| {
            if contracted {
                (-RATE).mul_add(gradient, weight)
            } else {
                let product = RATE * gradient;
                weight - product
            }
        })
        .collect()
}

pub fn verify(expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    for (lane, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "SGD output lane {lane}"
        );
    }
}
