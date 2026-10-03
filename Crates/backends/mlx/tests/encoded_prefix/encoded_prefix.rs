//! Exact raw `UInt` prefix publication preserves tails and every prior immutable owner.
#[rustfmt::skip]
use fusion_pcu_mlx::{MlxRuntime,MlxError};
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuScalarType,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits};
trait Representation: PcuScalar + Eq + core::fmt::Debug {
    const SIGN: u16;
    fn from_bits(bits: u16) -> Self;
}
macro_rules! representations {
    ($($ty:ty,$sign:expr),+) => {$(impl Representation for $ty {
        const SIGN: u16 = $sign;
        fn from_bits(bits: u16) -> Self { Self::from_bits(bits) }
    })+};
}
representations!(PcuF16Bits, 0x8000, PcuBf16Bits, 0x8000);
macro_rules! fp8_representations {
    ($($ty:ty),+) => {$(impl Representation for $ty {
        const SIGN: u16 = 0x80;
        fn from_bits(bits: u16) -> Self { Self::from_bits(u8::try_from(bits).unwrap()) }
    })+};
}
fp8_representations!(PcuF8E4M3FnBits, PcuF8E5M2Bits);
#[allow(
    clippy::cognitive_complexity,
    reason = "Keep the existing exhaustive encoding, immutable-prefix lifetime and negative preflight oracle together; Rust1.94 counts its nested assertions above the cold fixture limit."
)]
fn qualify<T: Representation>() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let other = runtime.open_gpu(0).unwrap();
    let count = usize::from(T::SIGN) * 2;
    let sentinel = T::from_bits(T::SIGN | 7);
    let mut previous = vec![sentinel; count + 3];
    previous[count + 1] = T::from_bits(T::SIGN - 1);
    previous[count + 2] = T::from_bits(0);
    let old = session.upload_encoded(&previous).unwrap();
    let sibling = old.clone();
    let plan = session
        .prepare_encoded_prefix(T::TYPE, count, count + 3)
        .unwrap();
    assert_eq!(plan.scalar_type(), T::TYPE);
    assert_eq!(plan.prefix_count(), count);
    assert_eq!(plan.output_count(), count + 3);
    let mut latest = None;
    for phase in [0, 1, 2] {
        let input: Vec<_> = (0..count)
            .map(|bits| {
                let bits = u16::try_from((bits + phase) % count).unwrap();
                T::from_bits(bits)
            })
            .collect();
        let prefix = session.upload_encoded(&input).unwrap();
        let foreign = other.upload_encoded(&input).unwrap();
        assert!(matches!(
            plan.execute(&foreign, &old),
            Err(MlxError::ForeignSession)
        ));
        let short = session.upload_encoded(&input[..1]).unwrap();
        assert!(matches!(
            plan.execute(&short, &old),
            Err(MlxError::InvalidExtent)
        ));
        let wrong_extent = session.upload_encoded(&previous[..count]).unwrap();
        assert!(matches!(
            plan.execute(&prefix, &wrong_extent),
            Err(MlxError::InvalidExtent)
        ));
        let output = plan.execute(&prefix, &old).unwrap();
        let mut actual = vec![sentinel; count + 6];
        output.read_into(&mut actual).unwrap();
        assert_eq!(&actual[..count], &input);
        assert_eq!(&actual[count..count + 3], &previous[count..]);
        assert_eq!(&actual[count + 3..], &[sentinel; 3]);
        sibling.read_into(&mut actual).unwrap();
        assert_eq!(&actual[..count + 3], &previous);
        let mut unchanged = vec![sentinel; count];
        prefix.read_into(&mut unchanged).unwrap();
        assert_eq!(unchanged, input);
        if let Some((prior, expected)) = latest.replace((output, input)) {
            prior.read_into(&mut actual).unwrap();
            assert_eq!(&actual[..count], &expected);
            assert_eq!(&actual[count..count + 3], &previous[count..]);
        }
    }
    for (prefix, output) in [(0, 1), (1, 1), (2, 1), (1, usize::MAX)] {
        assert!(matches!(
            session.prepare_encoded_prefix(T::TYPE, prefix, output),
            Err(MlxError::InvalidExtent)
        ));
    }
    assert!(matches!(
        session.prepare_encoded_prefix(PcuScalarType::Bool, 1, 2),
        Err(MlxError::UnsupportedScalar(PcuScalarType::Bool))
    ));
    let scalar_plan = session.prepare_encoded_prefix(T::TYPE, 1, 257).unwrap();
    let scalar = session
        .upload_encoded(&[T::from_bits(T::SIGN - 1)])
        .unwrap();
    let background = session.upload_encoded(&[sentinel; 257]).unwrap();
    let scalar_output = scalar_plan.execute(&scalar, &background).unwrap();
    let wrong = if T::TYPE == PcuScalarType::F16 {
        session
            .upload_encoded(&[PcuBf16Bits::from_bits(1)])
            .unwrap()
    } else {
        session.upload_encoded(&[PcuF16Bits::from_bits(1)]).unwrap()
    };
    assert!(matches!(
        scalar_plan.execute(&wrong, &background),
        Err(MlxError::UnsupportedScalar(_))
    ));
    drop(scalar_plan);
    drop(plan);
    drop(session);
    drop(runtime);
    let mut actual = [sentinel; 260];
    scalar_output.read_into(&mut actual).unwrap();
    assert_eq!(actual[0], T::from_bits(T::SIGN - 1));
    assert_eq!(&actual[1..], &[sentinel; 259]);
}
macro_rules! formats {
    ($name:ident,$ty:ty) => {
        #[test]
        #[ignore = "Requires actual MLX UInt prefix-copy and immutable owner lifetime proof."]
        fn $name() {
            qualify::<$ty>();
        }
    };
}
formats!(half_prefix, PcuF16Bits);
formats!(bfloat_prefix, PcuBf16Bits);
formats!(e4_prefix, PcuF8E4M3FnBits);
formats!(e5_prefix, PcuF8E5M2Bits);
