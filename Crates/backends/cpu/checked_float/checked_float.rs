//! Explicit checked CPU invocation maps using only the core numerical oracles.

#[rustfmt::skip]
use fusion_pcu::{
    validate_checked_float_map_kernel,
    validate_checked_float_conversion_map_kernel,
    validate_dispatch_submission,
    validate_host_scalar_bindings,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuCheckedFloatConversion,
    PcuCheckedFloatWidening,
    PcuClampedError,
    PcuClampedFloat,
    PcuClampedFloatConversion,
    PcuDispatchCheckedFloatConversion,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchOp,
    PcuDispatchOpCaps,
    PcuDispatchSubmission,
    PcuDispatchValueId,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuInvocationParameters,
    PcuParameterValue,
    PcuRangePolicy,
    PcuSynchronousHostDispatchBackend,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use crate::{
    PcuCpuTypedBinding,
    PcuTypedConversionReferenceError,
    VALUE_SLOTS,
    typed_conversion::{
        Value,
        load_slice_value,
        store_slice_value,
        validate_bindings,
    },
};

/// Admission, storage, or terminal arithmetic failure of an explicit checked CPU map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCheckedFloatReferenceError {
    UnsupportedNumericalRequirements,
    InvalidSubmission,
    UnsupportedProfile,
    Binding(PcuTypedConversionReferenceError),
    InvalidValue(PcuDispatchValueId),
    /// Unrecovered faults make all outputs unusable. A recovered fault preserves complete output.
    Fault(PcuExecutionFault),
}

/// Opt-in bounded heterogeneous F32/F64 checked CPU oracle. No automatic fallback is provided.
///
/// SSA identifiers must fit the shared typed verifier's 256-slot limit; out-of-range identifiers
/// are rejected before execution. Host bindings remain borrowed and are never truncated.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuCheckedFloatReference;
/// Homogeneous binary32 adapter for the same checked profile.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuCheckedF32Reference;
/// Homogeneous binary64 adapter for the same checked profile.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcuCheckedF64Reference;

impl PcuCheckedFloatReference {
    /// Implemented instruction floor; raw value ALU is deliberately absent. Structural admission
    /// further restricts this floor to direct or one-dimensional grid-stride scalar maps.
    pub const SUPPORTED_INSTRUCTIONS: PcuDispatchOpCaps = PcuDispatchOpCaps::VALUE_CONSTANT
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
        .union(PcuDispatchOpCaps::CONTROL_RETURN)
        .union(PcuDispatchOpCaps::CONTROL_LOOP)
        .union(PcuDispatchOpCaps::BINDING_LOAD)
        .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
        .union(PcuDispatchOpCaps::BINDING_STORE);

    /// Executes validated checked maps over heterogeneous caller-owned typed slices.
    ///
    /// Logical invocations execute in increasing order, including grid-stride maps. Each starts
    /// with logically fresh bounded SSA. Recovered range faults continue every operation and invocation;
    /// the earliest fatal fault supersedes any recovered status. Unrecovered output must be
    /// discarded. Caller-owned buffers may contain a written prefix on fatal failure; this API
    /// does not roll back host writes or publish an owned facade result. The oracle retains no
    /// state, so a subsequent call starts cleanly.
    ///
    /// # Errors
    /// Rejects invalid profiles and bindings before touching outputs; reports terminal arithmetic
    /// faults with their lowest logical invocation. Complete clamp output remains observable.
    pub fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuCpuTypedBinding<'_>],
    ) -> Result<(), PcuCheckedFloatReferenceError> {
        if submission
            .kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            == fusion_pcu::PcuReproducibility::PortableV1
        {
            return Err(PcuCheckedFloatReferenceError::UnsupportedNumericalRequirements);
        }
        validate_profile(submission)?;
        validate_bindings(submission, bindings, false, 0)
            .map_err(PcuCheckedFloatReferenceError::Binding)?;
        execute(submission, bindings)
    }
}

fn validate_profile(
    submission: PcuDispatchSubmission<'_>,
) -> Result<(), PcuCheckedFloatReferenceError> {
    validate_dispatch_submission(submission)
        .map_err(|_| PcuCheckedFloatReferenceError::InvalidSubmission)?;
    let kernel = submission.kernel;
    let caps = PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64;
    if validate_checked_float_map_kernel(kernel, PcuValueType::f32(), caps).is_err()
        && validate_checked_float_map_kernel(kernel, PcuValueType::f64(), caps).is_err()
        && validate_checked_float_conversion_map_kernel(kernel, caps).is_err()
    {
        return Err(PcuCheckedFloatReferenceError::UnsupportedProfile);
    }
    Ok(())
}

trait Storage {
    fn load(&self, target: PcuBindingRef, element: usize) -> Value;
    fn store(
        &mut self,
        target: PcuBindingRef,
        element: usize,
        value: Value,
    ) -> Result<(), PcuCheckedFloatReferenceError>;
}
impl Storage for [PcuCpuTypedBinding<'_>] {
    fn load(&self, target: PcuBindingRef, element: usize) -> Value {
        let binding = self
            .iter()
            .find(|binding| binding.target == target)
            .expect("preflight validates binding coverage");
        load_slice_value(&binding.slice, element)
    }
    fn store(
        &mut self,
        target: PcuBindingRef,
        element: usize,
        value: Value,
    ) -> Result<(), PcuCheckedFloatReferenceError> {
        let binding = self
            .iter_mut()
            .find(|binding| binding.target == target)
            .expect("preflight validates binding coverage");
        store_slice_value(&mut binding.slice, element, value, target)
            .map_err(PcuCheckedFloatReferenceError::Binding)
    }
}

#[allow(clippy::too_many_lines)] // Keep terminal status and checked SSA execution in one auditable pass.
fn execute(
    submission: PcuDispatchSubmission<'_>,
    storage: &mut (impl Storage + ?Sized),
) -> Result<(), PcuCheckedFloatReferenceError> {
    let (body, extent) = match submission.kernel.ops {
        [PcuDispatchOp::GridStrideLoop { extent, body }, _] => (
            *body,
            usize::try_from(*extent)
                .map_err(|_| PcuCheckedFloatReferenceError::InvalidSubmission)?,
        ),
        ops => (
            ops,
            usize::try_from(submission.shape.invocation_count().get())
                .map_err(|_| PcuCheckedFloatReferenceError::InvalidSubmission)?,
        ),
    };
    let mut recovered = None;
    // The shared typed verifier proves every read is dominated by a definition in this same
    // straight-line region. There are no branches or loop-carried SSA values in either admitted
    // profile. Each used definition is overwritten before use on every logical invocation, so
    // stale unused slots are unobservable. Clear bounded scratch once per call, not per element.
    let mut values = [None; VALUE_SLOTS];
    for logical in 0..extent {
        for op in body {
            let result = match *op {
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result,
                    binding,
                    index,
                }) => {
                    let element = if index == PcuDispatchIndex::BindingElementZero {
                        0
                    } else {
                        logical
                    };
                    Some((result, Ok(storage.load(binding, element))))
                }
                PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                    let value = match value {
                        PcuParameterValue::F32(bits) => Value::F32(f32::from_bits(bits)),
                        PcuParameterValue::F64(bits) => Value::F64(f64::from_bits(bits)),
                        _ => unreachable!("preflight admits floating constants only"),
                    };
                    Some((result, Ok(value)))
                }
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    result,
                    op,
                    lhs,
                    rhs,
                    underflow_policy,
                    range_policy,
                    ..
                }) => Some((
                    result,
                    binary(
                        op,
                        get(&values, lhs)?,
                        get(&values, rhs)?,
                        underflow_policy,
                        range_policy,
                    ),
                )),
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                    result,
                    op,
                    value,
                    underflow_policy,
                    range_policy,
                    ..
                }) => Some((
                    result,
                    unary(op, get(&values, value)?, underflow_policy, range_policy),
                )),
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                    result,
                    value,
                    conversion,
                    underflow_policy,
                    range_policy,
                }) => Some((
                    result,
                    convert(
                        get(&values, value)?,
                        conversion,
                        underflow_policy,
                        range_policy,
                    ),
                )),
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) => {
                    storage.store(binding, logical, get(&values, value)?)?;
                    None
                }
                PcuDispatchOp::Control(_) => None,
                _ => unreachable!("profile admits checked scalar maps only"),
            };
            if let Some((id, result)) = result {
                let value = match result {
                    Ok(value) => value,
                    Err(EvaluationError {
                        kind,
                        recovery: Some(value),
                    }) => {
                        recovered.get_or_insert_with(|| PcuExecutionFault {
                            kind,
                            invocation_id: u64::try_from(logical).expect("logical extent fits u32"),
                            recovered: true,
                        });
                        value
                    }
                    Err(EvaluationError {
                        kind,
                        recovery: None,
                    }) => {
                        return Err(PcuCheckedFloatReferenceError::Fault(PcuExecutionFault {
                            kind,
                            invocation_id: u64::try_from(logical).expect("logical extent fits u32"),
                            recovered: false,
                        }));
                    }
                };
                values[usize::from(id.0)] = Some(value);
            }
        }
    }
    recovered.map_or(Ok(()), |fault| {
        Err(PcuCheckedFloatReferenceError::Fault(fault))
    })
}
fn get(
    values: &[Option<Value>; VALUE_SLOTS],
    id: PcuDispatchValueId,
) -> Result<Value, PcuCheckedFloatReferenceError> {
    values[usize::from(id.0)].ok_or(PcuCheckedFloatReferenceError::InvalidValue(id))
}
struct EvaluationError {
    kind: PcuExecutionFaultKind,
    recovery: Option<Value>,
}
type Evaluation = Result<Value, EvaluationError>;
fn checked<T>(
    result: Result<T, PcuExecutionFaultKind>,
    wrap: impl FnOnce(T) -> Value,
) -> Evaluation {
    result.map(wrap).map_err(|kind| EvaluationError {
        kind,
        recovery: None,
    })
}
fn clamped<T>(result: Result<T, PcuClampedError<T>>, wrap: impl FnOnce(T) -> Value) -> Evaluation {
    match result {
        Ok(value) => Ok(wrap(value)),
        Err(PcuClampedError::Range(fault)) => Err(EvaluationError {
            kind: fault.kind(),
            recovery: Some(wrap(fault.clamped_value())),
        }),
        Err(PcuClampedError::Fatal(kind)) => Err(EvaluationError {
            kind,
            recovery: None,
        }),
    }
}
fn binary_float<T: PcuClampedFloat>(
    op: PcuDispatchFloatBinaryOp,
    lhs: T,
    rhs: T,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
    wrap: impl FnOnce(T) -> Value,
) -> Evaluation {
    if range == PcuRangePolicy::Clamp {
        clamped(
            match op {
                PcuDispatchFloatBinaryOp::Add => lhs.pcu_clamped_add_with_policy(rhs, underflow),
                PcuDispatchFloatBinaryOp::Sub => lhs.pcu_clamped_sub_with_policy(rhs, underflow),
                PcuDispatchFloatBinaryOp::Mul => lhs.pcu_clamped_mul_with_policy(rhs, underflow),
                PcuDispatchFloatBinaryOp::Div => lhs.pcu_clamped_div_with_policy(rhs, underflow),
            },
            wrap,
        )
    } else {
        checked(
            match op {
                PcuDispatchFloatBinaryOp::Add => lhs.pcu_checked_add_with_policy(rhs, underflow),
                PcuDispatchFloatBinaryOp::Sub => lhs.pcu_checked_sub_with_policy(rhs, underflow),
                PcuDispatchFloatBinaryOp::Mul => lhs.pcu_checked_mul_with_policy(rhs, underflow),
                PcuDispatchFloatBinaryOp::Div => lhs.pcu_checked_div_with_policy(rhs, underflow),
            },
            wrap,
        )
    }
}
fn binary(
    op: PcuDispatchFloatBinaryOp,
    lhs: Value,
    rhs: Value,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Evaluation {
    match (lhs, rhs) {
        (Value::F32(lhs), Value::F32(rhs)) => {
            binary_float(op, lhs, rhs, underflow, range, Value::F32)
        }
        (Value::F64(lhs), Value::F64(rhs)) => {
            binary_float(op, lhs, rhs, underflow, range, Value::F64)
        }
        _ => unreachable!("typed SSA preflight validates operand widths"),
    }
}
fn unary(
    op: PcuDispatchFloatUnaryOp,
    value: Value,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Evaluation {
    fn apply<T: PcuCheckedFloat + PcuClampedFloat>(
        op: PcuDispatchFloatUnaryOp,
        value: T,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
        wrap: fn(T) -> Value,
    ) -> Evaluation {
        if range == PcuRangePolicy::Clamp {
            clamped(
                match op {
                    PcuDispatchFloatUnaryOp::Relu => value.pcu_clamped_relu_with_policy(underflow),
                    PcuDispatchFloatUnaryOp::Neg => value.pcu_clamped_neg_with_policy(underflow),
                },
                wrap,
            )
        } else {
            checked(
                match op {
                    PcuDispatchFloatUnaryOp::Relu => value.pcu_checked_relu_with_policy(underflow),
                    PcuDispatchFloatUnaryOp::Neg => value.pcu_checked_neg_with_policy(underflow),
                },
                wrap,
            )
        }
    }
    match value {
        Value::F32(value) => apply(op, value, underflow, range, Value::F32),
        Value::F64(value) => apply(op, value, underflow, range, Value::F64),
        _ => unreachable!("typed SSA preflight validates unary width"),
    }
}

fn convert(
    value: Value,
    conversion: PcuDispatchCheckedFloatConversion,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Evaluation {
    match (conversion, value) {
        (PcuDispatchCheckedFloatConversion::F32ToF64, Value::F32(value)) => {
            checked(value.pcu_checked_to_f64(), Value::F64)
        }
        (PcuDispatchCheckedFloatConversion::F64ToF32, Value::F64(value))
            if range == PcuRangePolicy::Clamp =>
        {
            clamped(value.pcu_clamped_to_f32_with_policy(underflow), Value::F32)
        }
        (PcuDispatchCheckedFloatConversion::F64ToF32, Value::F64(value)) => {
            checked(value.pcu_checked_to_f32_with_policy(underflow), Value::F32)
        }
        _ => unreachable!("typed SSA preflight validates conversion width"),
    }
}

macro_rules! homogeneous {
    ($reference:ty, $scalar:ty, $variant:ident, $value_type:expr) => {
        impl Storage for [PcuHostScalarBinding<'_, $scalar>] {
            fn load(&self, target: PcuBindingRef, element: usize) -> Value {
                let binding = self
                    .iter()
                    .find(|binding| binding.target == target)
                    .expect("host binding preflight validates coverage");
                Value::$variant(match &binding.slice {
                    PcuHostScalarSlice::Read(slice) => slice[element],
                    PcuHostScalarSlice::ReadWrite(slice) => slice[element],
                })
            }
            fn store(
                &mut self,
                target: PcuBindingRef,
                element: usize,
                value: Value,
            ) -> Result<(), PcuCheckedFloatReferenceError> {
                let Value::$variant(value) = value else {
                    unreachable!("homogeneous typed SSA")
                };
                let binding = self
                    .iter_mut()
                    .find(|binding| binding.target == target)
                    .expect("host binding preflight validates coverage");
                let PcuHostScalarSlice::ReadWrite(slice) = &mut binding.slice else {
                    unreachable!("host binding preflight validates writes")
                };
                slice[element] = value;
                Ok(())
            }
        }
        // SAFETY: Admission and host binding validation precede access; execution is synchronous
        // on exclusively borrowed host slices and retains no references on any return path.
        unsafe impl PcuSynchronousHostDispatchBackend<$scalar> for $reference {
            type Error = PcuCheckedFloatReferenceError;
            fn run_host_direct(
                &self,
                submission: PcuDispatchSubmission<'_>,
                bindings: &mut [PcuHostScalarBinding<'_, $scalar>],
                _parameters: PcuInvocationParameters<'_>,
            ) -> Result<(), Self::Error> {
                if submission
                    .kernel
                    .numerical_requirements
                    .numerical_options
                    .reproducibility
                    == fusion_pcu::PcuReproducibility::PortableV1
                {
                    return Err(PcuCheckedFloatReferenceError::UnsupportedNumericalRequirements);
                }
                validate_host_scalar_bindings::<$scalar, ()>(submission, bindings)
                    .map_err(|_| PcuCheckedFloatReferenceError::InvalidSubmission)?;
                validate_checked_float_map_kernel(
                    submission.kernel,
                    $value_type,
                    PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
                )
                .map_err(|_| PcuCheckedFloatReferenceError::UnsupportedProfile)?;
                execute(submission, bindings)
            }
        }
    };
}
homogeneous!(PcuCheckedF32Reference, f32, F32, PcuValueType::f32());
homogeneous!(PcuCheckedF64Reference, f64, F64, PcuValueType::f64());

#[cfg(test)]
mod tests;
