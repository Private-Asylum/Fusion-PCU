//! Integer/dyadic full-output oracle for the timed negative-quarter learning rate.
pub const RATE: f32 = -0.25;
pub struct Fixture<const N: usize> {
    pub weights: Box<[f32; N]>,
    pub gradient: Box<[f32; N]>,
    pub expected: Vec<f32>,
}

pub fn fixture<const N: usize>(phase: u16) -> Fixture<N> {
    assert!(N > 0);
    let mut weights = Vec::with_capacity(N);
    let mut gradient = Vec::with_capacity(N);
    let mut expected = Vec::with_capacity(N);
    for index in 0..N {
        let w = i16::try_from(index % 9).unwrap() - 4 + i16::try_from(phase % 2).unwrap();
        let g = i16::try_from(index % 7).unwrap() - 3 + i16::try_from((phase + 1) % 2).unwrap();
        weights.push(f32::from(w) / 8.0);
        gradient.push(f32::from(g) / 8.0);
        // Product -(g/8)/4 and subtraction w/8 - product are exactly representable.
        // FMA yields the same exact value; zero cancellation rounds to positive zero.
        expected.push(f32::from(4 * w + g) / 32.0);
    }
    Fixture {
        weights: weights.into_boxed_slice().try_into().unwrap(),
        gradient: gradient.into_boxed_slice().try_into().unwrap(),
        expected,
    }
}

pub fn verify(expected: &[f32], observed: &[f32]) {
    assert_eq!(expected.len(), observed.len());
    for (index, (&expected, &actual)) in expected.iter().zip(observed).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "independent SGD dyadic oracle lane{index}"
        );
    }
}
