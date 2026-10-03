//! Caller capacity must not weaken SSA ordering, type checks or region isolation.
#[rustfmt::skip]
use crate::{
    validate_typed_dispatch_value_flow,
    validate_typed_dispatch_value_flow_with_scratch,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp as Data,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex as Index,
    PcuDispatchKernelIr,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuKernelId,
    PcuScalarType,
    PcuTypedDispatchValidationError as Error,
    PcuValueType,
    PcuValueTypeCaps,
    model::PcuIntegerDivFlags,
};

fn binding(slot: u32, scalar: PcuScalarType, access: PcuBindingAccess) -> PcuBinding<'static> {
    PcuBinding::value(
        None,
        0,
        slot,
        PcuBindingStorageClass::Storage,
        access,
        PcuValueType::Scalar(scalar),
    )
}

fn kernel<'a>(bindings: &'a [PcuBinding<'a>], ops: &'a [Op<'a>]) -> PcuDispatchKernelIr<'a> {
    PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(1),
        entry: PcuDispatchEntryPoint {
            name: "caller_scratch",
            logical_shape: [3, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::empty(),
        feature_caps: PcuDispatchFeatureCaps::empty(),
    }
}

fn load(result: Id, slot: u32, index: Index) -> Op<'static> {
    Op::Data(Data::BindingLoad {
        result,
        binding: PcuBindingRef::new(0, slot),
        index,
    })
}

fn store(value: Id, slot: u32, index: Index) -> Op<'static> {
    Op::Data(Data::BindingStore {
        value,
        binding: PcuBindingRef::new(0, slot),
        index,
    })
}

#[test]
fn full_u16_id_space_retains_all_scalar_metadata_and_region_geometry() {
    // Test-only host allocation; the verifier receives and clears borrowed
    // metadata. Type flow does not grant packed/extended physical transport.
    let mut scratch = std::vec![None; usize::from(u16::MAX) + 1];
    for scalar in PcuScalarType::ALL {
        let bindings = [
            binding(7, scalar, PcuBindingAccess::ReadOnly),
            binding(11, scalar, PcuBindingAccess::WriteOnly),
        ];
        let direct = [
            load(Id(u16::MAX), 7, Index::InvocationId),
            store(Id(u16::MAX), 11, Index::InvocationId),
            Op::Control(PcuDispatchControlOp::Return),
        ];
        assert_eq!(
            validate_typed_dispatch_value_flow(&kernel(&bindings, &direct)),
            Err(Error::ValueOutOfRange(Id(u16::MAX))),
        );
        assert_eq!(
            validate_typed_dispatch_value_flow_with_scratch(
                &kernel(&bindings, &direct),
                &mut scratch,
            ),
            Ok(()),
        );
        assert_eq!(
            scratch[usize::from(u16::MAX)],
            Some(PcuValueType::Scalar(scalar))
        );
        let body = [
            load(Id(u16::MAX), 7, Index::GridStrideId),
            store(Id(u16::MAX), 11, Index::GridStrideId),
        ];
        let grid = [
            Op::GridStrideLoop {
                body: &body,
                extent: 17,
            },
            Op::Control(PcuDispatchControlOp::Return),
        ];
        assert_eq!(
            validate_typed_dispatch_value_flow_with_scratch(
                &kernel(&bindings, &grid),
                &mut scratch,
            ),
            Ok(()),
        );
        assert_eq!(
            validate_typed_dispatch_value_flow_with_scratch(
                &kernel(&bindings, &direct),
                &mut scratch[..usize::from(u16::MAX)],
            ),
            Err(Error::ValueOutOfRange(Id(u16::MAX))),
        );
    }
}

#[test]
fn fresh_region_clears_supplied_and_previous_failed_types() {
    let bindings = [
        binding(0, PcuScalarType::F32, PcuBindingAccess::ReadOnly),
        binding(1, PcuScalarType::F32, PcuBindingAccess::WriteOnly),
    ];
    let mut scratch = [Some(PcuValueType::f32()); 4];
    let missing = [store(Id(2), 1, Index::InvocationId)];
    assert_eq!(
        validate_typed_dispatch_value_flow_with_scratch(&kernel(&bindings, &missing), &mut scratch,),
        Err(Error::UndefinedValue(Id(2))),
    );
    let duplicate = [
        load(Id(2), 0, Index::InvocationId),
        load(Id(2), 0, Index::InvocationId),
    ];
    assert_eq!(
        validate_typed_dispatch_value_flow_with_scratch(
            &kernel(&bindings, &duplicate),
            &mut scratch,
        ),
        Err(Error::DuplicateValue(Id(2))),
    );
    assert_eq!(scratch[2], Some(PcuValueType::f32()));
    assert_eq!(
        validate_typed_dispatch_value_flow_with_scratch(&kernel(&bindings, &missing), &mut scratch,),
        Err(Error::UndefinedValue(Id(2))),
    );
    assert_eq!(scratch, [None; 4]);
}

#[test]
fn sparse_joint_outputs_keep_exact_width_order_and_checked_flags() {
    let mut scratch = std::vec![None; 4096];
    for scalar in [
        PcuScalarType::I8,
        PcuScalarType::U8,
        PcuScalarType::I16,
        PcuScalarType::U16,
        PcuScalarType::I32,
        PcuScalarType::U32,
        PcuScalarType::I64,
        PcuScalarType::U64,
        PcuScalarType::I128,
        PcuScalarType::U128,
        PcuScalarType::I256,
        PcuScalarType::U256,
        PcuScalarType::I512,
        PcuScalarType::U512,
    ] {
        let bindings = [
            binding(0, scalar, PcuBindingAccess::ReadOnly),
            binding(1, scalar, PcuBindingAccess::ReadOnly),
            binding(2, scalar, PcuBindingAccess::WriteOnly),
            binding(3, scalar, PcuBindingAccess::WriteOnly),
        ];
        let divide = |flags| Data::CheckedDivRem {
            value_type: PcuValueType::Scalar(scalar),
            flags,
            quotient: Id(2048),
            remainder: Id(4095),
            lhs: Id(512),
            rhs: Id(1024),
        };
        let mut ops = [
            load(Id(512), 0, Index::InvocationId),
            load(Id(1024), 1, Index::InvocationId),
            Op::Data(divide(PcuIntegerDivFlags::CHECKED)),
            store(Id(4095), 3, Index::InvocationId),
            store(Id(2048), 2, Index::InvocationId),
        ];
        assert_eq!(
            validate_typed_dispatch_value_flow_with_scratch(&kernel(&bindings, &ops), &mut scratch,),
            Ok(()),
        );
        assert_eq!(
            validate_typed_dispatch_value_flow_with_scratch(
                &kernel(&bindings, &ops),
                &mut scratch[..4095],
            ),
            Err(Error::ValueOutOfRange(Id(4095))),
        );
        ops[2] = Op::Data(divide(PcuIntegerDivFlags::DIV_OR_ZERO));
        assert_eq!(
            validate_typed_dispatch_value_flow_with_scratch(&kernel(&bindings, &ops), &mut scratch,),
            Err(Error::UnsupportedOperation(2)),
        );
    }
}

#[test]
fn additional_capacity_does_not_relax_types_or_allocate_for_empty_regions() {
    let bindings = [
        binding(0, PcuScalarType::F32, PcuBindingAccess::ReadOnly),
        binding(1, PcuScalarType::F64, PcuBindingAccess::WriteOnly),
    ];
    let wrong_type = [
        load(Id(300), 0, Index::InvocationId),
        store(Id(300), 1, Index::InvocationId),
    ];
    let mut scratch = [None; 301];
    assert_eq!(
        validate_typed_dispatch_value_flow_with_scratch(
            &kernel(&bindings, &wrong_type),
            &mut scratch,
        ),
        Err(Error::TypeMismatch {
            expected: PcuValueType::f64(),
            actual: PcuValueType::f32()
        }),
    );
    let return_only = [Op::Control(PcuDispatchControlOp::Return)];
    assert_eq!(
        validate_typed_dispatch_value_flow_with_scratch(&kernel(&[], &return_only), &mut [],),
        Ok(()),
    );
    assert_eq!(
        validate_typed_dispatch_value_flow_with_scratch(&kernel(&bindings, &wrong_type), &mut [],),
        Err(Error::ValueOutOfRange(Id(300))),
    );
}
