//! Independent integer/dyadic witness; exact sums do not certify general vendor reductions.
pub struct Fixture<const N: usize> {
    pub prediction: Box<[f32; N]>,
    pub target: Box<[f32; N]>,
    pub expected: f32,
}

pub fn fixture<const N: usize>(phase: u16) -> Fixture<N> {
    assert!(N > 0 && N <= 1_048_576);
    let mut prediction = Vec::with_capacity(N);
    let mut target = Vec::with_capacity(N);
    let mut squared_units = 0_u32;
    for index in 0..N {
        let prediction_units =
            i16::try_from(index % 3).unwrap() - 1 + i16::try_from(phase % 2).unwrap();
        let target_units =
            i16::try_from(index % 2).unwrap() + i16::try_from((phase + 1) % 2).unwrap();
        let difference = i32::from(prediction_units - target_units);
        squared_units += u32::try_from(difference * difference).unwrap();
        prediction.push(f32::from(prediction_units) / 8.0);
        target.push(f32::from(target_units) / 8.0);
    }
    // Integer differences are at most three units; all partial square sums are below 2^24.
    // Dyadic division by 64 is exact. Only the specified F32 reciprocal/scale rounds.
    assert!(squared_units <= 1 << 24);
    #[allow(clippy::cast_precision_loss)] // Witness bound above proves exact integer conversion.
    let exact_sum = squared_units as f32 / 64.0;
    Fixture {
        prediction: prediction.into_boxed_slice().try_into().unwrap(),
        target: target.into_boxed_slice().try_into().unwrap(),
        expected: exact_sum * scale(N),
    }
}

#[allow(clippy::cast_precision_loss)] // Matches declared F32 count conversion, not an f64 oracle.
pub fn scale(count: usize) -> f32 {
    1.0 / count as f32
}

pub fn verify(expected: f32, observed: f32) {
    assert_eq!(
        observed.to_bits(),
        expected.to_bits(),
        "independent dyadic loss oracle"
    );
}
