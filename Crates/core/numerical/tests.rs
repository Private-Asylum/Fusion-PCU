use super::*;

#[test]
fn independent_overrides_preserve_unspecified_requirements() {
    let defaults = PcuNumericalOptions::default();
    let portable = defaults.with_overrides(PcuNumericalOverrides {
        reproducibility: Some(PcuReproducibility::PortableV1),
        ..PcuNumericalOverrides::default()
    });
    assert_eq!(
        portable.compound_arithmetic,
        PcuCompoundArithmeticPolicy::Checked
    );
    assert_eq!(portable.precision, PcuPrecisionPolicy::Preserve);

    let permitted = portable.with_overrides(PcuNumericalOverrides {
        compound_arithmetic: Some(PcuCompoundArithmeticPolicy::BackendDefined),
        ..PcuNumericalOverrides::default()
    });
    assert_eq!(permitted.reproducibility, PcuReproducibility::PortableV1);
    assert_eq!(permitted.precision, PcuPrecisionPolicy::Preserve);

    let optimized = permitted.with_overrides(PcuNumericalOverrides {
        precision: Some(PcuPrecisionPolicy::BackendOptimized),
        ..PcuNumericalOverrides::default()
    });
    assert_eq!(optimized.reproducibility, PcuReproducibility::PortableV1);
    assert_eq!(
        optimized.compound_arithmetic,
        PcuCompoundArithmeticPolicy::BackendDefined
    );
    assert_eq!(
        optimized,
        optimized.with_overrides(PcuNumericalOverrides::default())
    );
}
