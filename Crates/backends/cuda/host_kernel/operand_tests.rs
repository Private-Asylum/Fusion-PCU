//! Cold source-declaration checks survive device-ABI projection.
use super::*;
#[path = "../benches/checked_float_operands/source/source.rs"]
#[allow(dead_code)] // Only the generated IR helper is used in this cold fixture.
mod source;

#[test]
fn unread_extent_is_proved_from_ir_and_declared_type_still_validates() {
    use fusion_pcu::PcuF16Bits;
    let bindings = source::mul::unused_bindings::<PcuF16Bits>();
    let builder = source::mul::unused_ir::<PcuF16Bits, 7>(&bindings).unwrap();
    let ir = builder.ir();
    let requirements = host_declaration_requirements(
        &ir,
        PcuInvocationShape::invocations(NonZeroU32::new(7).unwrap()),
    )
    .unwrap();
    assert_eq!(requirements.len(), 3);
    assert_eq!(requirements[0].min_required_bytes, 0);
    assert_eq!(requirements[0].access, PcuBindingAccess::ReadOnly);
    assert_eq!(requirements[1].min_required_bytes, 14);
    assert_eq!(requirements[2].min_required_bytes, 14);
    let empty: [PcuF16Bits; 0] = [];
    let input = [PcuF16Bits::from_bits(0x3c00); 7];
    let mut output = [PcuF16Bits::from_bits(0); 9];
    validate_host_arguments(
        &requirements,
        &[
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &empty),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), &input),
        ],
    )
    .unwrap();
    let wrong: [f32; 0] = [];
    assert!(matches!(validate_host_arguments(&requirements,&[
        PcuHostArgument::read(PcuBindingRef::new(0,0),&wrong),
        PcuHostArgument::read_write(PcuBindingRef::new(0,1),&mut output),
        PcuHostArgument::read(PcuBindingRef::new(0,2),&input),
    ]),Err(CudaHostKernelError::ScalarMismatch{binding,..}) if binding==PcuBindingRef::new(0,0)));
    assert!(matches!(validate_host_arguments(&requirements,&[
        PcuHostArgument::read_write(PcuBindingRef::new(0,1),&mut output),
        PcuHostArgument::read(PcuBindingRef::new(0,2),&input),
    ]),Err(CudaHostKernelError::MissingArgument(binding)) if binding==PcuBindingRef::new(0,0)));
}

#[path = "../benches/portable_unary/source/source.rs"]
#[allow(dead_code)] // Cold generated IR proves declaration obligations without device execution.
mod unary_source;

#[test]
fn unary_grid_unread_declaration_keeps_type_but_requires_no_storage() {
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuF16Bits,
        PcuFloatUnderflowPolicy,
        PcuImplementationRequirements,
        PcuRangePolicy,
        PcuReproducibility,
    };
    let bindings = unary_source::grid_bindings::<PcuF16Bits>();
    let mut numeric = PcuImplementationRequirements::DEFAULT;
    numeric.numerical_options.reproducibility = PcuReproducibility::PortableV1;
    let builder = unary_source::__grid_ir_with_float_underflow_policy::<PcuF16Bits, 65>(
        &bindings,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
        numeric,
    )
    .unwrap();
    builder.with_ir(|captured| {
        let mut ir = *captured;
        for reproducibility in [
            PcuReproducibility::PortableV1,
            PcuReproducibility::Unspecified,
        ] {
            ir.numerical_requirements.numerical_options.reproducibility = reproducibility;
            let obligations = host_declaration_requirements(
                &ir,
                PcuInvocationShape::invocations(NonZeroU32::new(17).unwrap()),
            )
            .unwrap();
            assert_eq!(obligations.len(), 3);
            assert_eq!(obligations[0].min_required_bytes, 0);
            assert_eq!(obligations[0].access, PcuBindingAccess::ReadOnly);
            assert_eq!(obligations[1].min_required_bytes, 130);
            assert_eq!(obligations[2].min_required_bytes, 130);
            let empty: [PcuF16Bits; 0] = [];
            let input = [PcuF16Bits::from_bits(0x3c00); 65];
            let mut output = [PcuF16Bits::from_bits(0); 67];
            validate_host_arguments(
                &obligations,
                &[
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &empty),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                    PcuHostArgument::read(PcuBindingRef::new(0, 2), &input),
                ],
            )
            .unwrap();
        }
    });
}
