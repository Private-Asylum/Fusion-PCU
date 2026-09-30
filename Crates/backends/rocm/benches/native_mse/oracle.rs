//! Exact bounded squared integers followed by the specified rounded F32 scale.
pub fn inputs<const N: usize>(phase: u32) -> (Box<[f32; N]>, Box<[f32; N]>) {
    assert!(phase <= 2);
    let prediction: Vec<_> = (0..N)
        .map(|index| {
            f32::from(u8::try_from(index % 5).unwrap()) - 2.0
                + f32::from(u8::try_from(phase).unwrap())
        })
        .collect();
    let target: Vec<_> = (0..N)
        .map(|index| f32::from(u8::try_from(index % 3).unwrap()) - 1.0)
        .collect();
    (
        prediction
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("prediction extent")),
        target
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("target extent")),
    )
}
pub fn expected<const N: usize>(phase: u32) -> f32 {
    assert!(phase <= 2);
    let sum: u32 = (0..N)
        .map(|index| {
            let difference = i32::try_from(index % 5).unwrap() - 1 + i32::try_from(phase).unwrap()
                - i32::try_from(index % 3).unwrap();
            u32::try_from(difference * difference).unwrap()
        })
        .sum();
    assert!(sum <= 1 << 24);
    assert!(N > 0 && N <= 1 << 24);
    #[allow(clippy::cast_precision_loss)]
    // Both conversions are proven exactly representable above.
    let (sum, count) = (sum as f32, N as f32);
    sum * (1.0 / count)
}
pub fn verify(expected: f32, observed: &[f32]) {
    assert_eq!(observed.len(), 1);
    assert_eq!(expected.to_bits(), observed[0].to_bits());
}
