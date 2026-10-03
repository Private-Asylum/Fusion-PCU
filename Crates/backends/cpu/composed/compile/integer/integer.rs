//! Cold ten-primitive checked integer plan; existing primitive profiles remain first.
#[rustfmt::skip]
use fusion_pcu::{
    assess_checked_integer_map_resources,
    PcuCheckedInteger,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuParameterValue,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use super::{
    build,
    execution,
    slot,
    BINDINGS,
    Error,
    Plan,
    Resource,
    Step,
    STEPS,
};

pub fn prepare<T: PcuCheckedInteger>(kernel: &PcuDispatchKernelIr<'_>) -> Result<Plan, Error> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != PcuReproducibility::Unspecified
        || !(2..=BINDINGS).contains(&kernel.bindings.len())
    {
        return Err(Error::UnsupportedProfile);
    }
    let ordinal = match T::TYPE {
        PcuScalarType::I8 => 0,
        PcuScalarType::U8 => 1,
        PcuScalarType::I16 => 2,
        PcuScalarType::U16 => 3,
        PcuScalarType::I32 => 4,
        PcuScalarType::U32 => 5,
        PcuScalarType::I64 => 6,
        PcuScalarType::U64 => 7,
        PcuScalarType::I128 => 8,
        PcuScalarType::U128 => 9,
        _ => return Err(Error::UnsupportedProfile),
    };
    let descriptor = assess_checked_integer_map_resources::<BINDINGS>(
        kernel,
        PcuValueType::Scalar(T::TYPE),
        PcuValueTypeCaps::for_scalar(T::TYPE),
    )
    .map_err(Error::InvalidIntegerResources)?;
    let body = crate::host::validated_region(kernel).map_err(|_| Error::UnsupportedProfile)?;
    if body.len() > STEPS {
        return Err(Error::UnsupportedProfile);
    }
    let execute = execution::integer::select(T::TYPE).ok_or(Error::UnsupportedProfile)?;
    build(
        kernel,
        descriptor,
        body,
        (T::TYPE, 19712 + ordinal, T::HOST_SIZE, execute),
        step,
    )
}

fn step(operation: PcuDispatchOp<'_>, resources: &[Resource]) -> Result<Step, Error> {
    match operation {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            result,
            op,
            lhs,
            rhs,
            range_policy,
            ..
        }) => Ok(Step::IntegerBinary {
            result: slot(result)?,
            left: slot(lhs)?,
            right: slot(rhs)?,
            operation: op,
            range: range_policy,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
            Ok(Step::IntegerConstant {
                result: slot(result)?,
                bytes: constant(value)?,
            })
        }
        PcuDispatchOp::Data(
            PcuDispatchDataOp::BindingLoad { .. } | PcuDispatchDataOp::BindingStore { .. },
        ) => super::step(operation, resources),
        _ => Err(Error::UnsupportedProfile),
    }
}

fn constant(value: PcuParameterValue) -> Result<[u8; 16], Error> {
    let mut result = [0; 16];
    macro_rules! bytes {
        ($value:expr) => {{
            let bytes = $value.to_ne_bytes();
            result[..bytes.len()].copy_from_slice(&bytes);
        }};
    }
    match value {
        PcuParameterValue::I8(value) => bytes!(value),
        PcuParameterValue::U8(value) => bytes!(value),
        PcuParameterValue::I16(value) => bytes!(value),
        PcuParameterValue::U16(value) => bytes!(value),
        PcuParameterValue::I32(value) => bytes!(value),
        PcuParameterValue::U32(value) => bytes!(value),
        PcuParameterValue::I64(value) => bytes!(value),
        PcuParameterValue::U64(value) => bytes!(value),
        _ => return Err(Error::UnsupportedProfile),
    }
    Ok(result)
}
