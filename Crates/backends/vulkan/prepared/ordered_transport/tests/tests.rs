//! CPU-only cold schema checks; no Vulkan object is initialized by these fixtures.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuImplementationRequirements,
    PcuScalarType,
};
use fusion_pcu_spirv::validate_ordered_scalar_transport_map;
#[path = "../../../tests/ordered_transport/graph/graph.rs"]
mod graph;

#[test]
fn arbitrary_unused_declarations_stay_typed_and_allocate_no_resource_slots() {
    graph::with(
        PcuScalarType::U8,
        47,
        false,
        PcuImplementationRequirements::default(),
        |kernel| {
            let mut bindings = kernel.bindings.to_vec();
            let extra = bindings[2];
            for index in 5..37 {
                let mut b = extra;
                b.binding = index;
                bindings.push(b);
            }
            let mut candidate = *kernel;
            candidate.bindings = &bindings;
            let profile = validate_ordered_scalar_transport_map(&candidate).unwrap();
            let _schema = schema::Schema::new(&candidate, &profile).unwrap();
            assert_eq!(profile.resources().len(), 4);
            let mut bindings = candidate.bindings.to_vec();
            bindings[36].access = PcuBindingAccess::WriteOnly;
            candidate.bindings = &bindings;
            assert!(schema::Schema::new(&candidate, &profile).is_err());
        },
    );
}
#[test]
fn short_later_output_and_wrong_unused_access_refuse_complete_call_preflight() {
    graph::with(
        PcuScalarType::U8,
        47,
        true,
        PcuImplementationRequirements::default(),
        |kernel| {
            let profile = validate_ordered_scalar_transport_map(kernel).unwrap();
            let schema = schema::Schema::new(kernel, &profile).unwrap();
            let input = [0xF7_u8; 47];
            let seed = [0x3C_u8];
            let mut stage = [0xA5_u8; 50];
            let mut output = [0x5A_u8; 52];
            let mut args = [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), &seed),
                PcuHostArgument::read_write::<u8>(PcuBindingRef::new(0, 2), &mut []),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut stage),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 4), &mut output),
            ];
            assert_eq!(schema.validate(&args, &profile).unwrap(), [0, 3, 1, 4]);
            args[2] = PcuHostArgument::read(PcuBindingRef::new(0, 2), &[] as &[u8]);
            assert!(schema.validate(&args, &profile).is_err());
            args[2] = PcuHostArgument::read_write::<u8>(PcuBindingRef::new(0, 2), &mut []);
            let mut short = [0x5A_u8; 46];
            args[4] = PcuHostArgument::read_write(PcuBindingRef::new(0, 4), &mut short);
            assert!(schema.validate(&args, &profile).is_err());
            assert_eq!(stage, [0xA5; 50]);
            assert_eq!(output, [0x5A; 52]);
        },
    );
}
