//! Cold, detached resource roles for one transactional checked quotient/remainder map.

#[rustfmt::skip]
use crate::{
    IntegerMapValidationError as Error,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp as Data,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex as Index,
    PcuDispatchKernelIr,
    PcuDispatchOp as Op,
    PcuDispatchValueId as Id,
    PcuRangePolicy,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Actual reads and quotient/remainder destinations, independent of declarations.
///
/// Every explicit load remains a read obligation, even when its SSA value is unused.
/// Providers must explicitly adopt this profile and freeze it during preparation.
/// Structural eligibility alone does not certify an executable implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedIntegerDivRemOperandSchema {
    inputs: [PcuBindingRef; 2],
    input_count: usize,
    indexed_inputs: [bool; 2],
    operand_inputs: [usize; 2],
    operand_indices: [Index; 2],
    outputs: [PcuBindingRef; 2],
}

impl CheckedIntegerDivRemOperandSchema {
    /// Distinct actual reads, in load order; unread declarations do not appear.
    #[must_use]
    pub fn input_bindings(&self) -> &[PcuBindingRef] {
        &self.inputs[..self.input_count]
    }

    /// Input slots for the mathematical dividend and divisor, respectively.
    #[must_use]
    pub const fn operand_inputs(&self) -> [usize; 2] {
        self.operand_inputs
    }

    /// Exact dividend/divisor load indices, including independent scalar broadcast.
    #[must_use]
    pub const fn operand_indices(&self) -> [Index; 2] {
        self.operand_indices
    }

    /// Required read spans for a prepared logical element count.
    ///
    /// A resource read both at zero and at the invocation index requires the full
    /// extent. The second slot is zero when only one distinct resource is loaded.
    #[must_use]
    pub const fn input_element_counts(&self, element_count: usize) -> [usize; 2] {
        let mut counts = [0; 2];
        let mut slot = 0;
        while slot < self.input_count {
            counts[slot] = if self.indexed_inputs[slot] {
                element_count
            } else {
                1
            };
            slot += 1;
        }
        counts
    }

    /// Distinct writable destinations, in quotient/remainder order, not store order.
    #[must_use]
    pub const fn output_bindings(&self) -> [PcuBindingRef; 2] {
        self.outputs
    }
}

/// Assesses two loads, one checked joint division, and two indexed stores.
///
/// One or two read-only declarations and exactly two writable declarations may
/// occur in any order. Loads may repeat resources; dividend/divisor may repeat or
/// reverse loaded SSA values. Store order does not determine the result roles.
/// Direct and grid-stride execution use positive one-dimensional geometry.
/// Checked division has no defined clamp recovery in this profile: zero divisor
/// and signed minimum divided by minus one remain fatal, joint rollback faults.
/// The existing distinct-input validator and backend admission remain unchanged.
///
/// # Errors
///
/// Rejects malformed interfaces, unsupported types or capability requirements,
/// invalid indices, invalid SSA, unsupported flags and ambiguous result writes.
pub fn assess_checked_integer_div_rem_operands(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    scalar_caps: PcuValueTypeCaps,
) -> Result<CheckedIntegerDivRemOperandSchema, Error> {
    validate_interface(kernel, value_type, scalar_caps)?;
    let (body, index) = map_body(kernel)?;
    let loads = [
        read_load(kernel, body[0], index, 0)?,
        read_load(kernel, body[1], index, 1)?,
    ];
    if loads[0].0 == loads[1].0 {
        return Err(Error::InvalidValue(loads[1].0));
    }
    let Data::CheckedDivRem {
        value_type: actual,
        flags,
        quotient,
        remainder,
        lhs,
        rhs,
    } = data(body[2], 2)?
    else {
        return Err(Error::UnsupportedOperation(2));
    };
    if actual != value_type || flags.bits() != 0 {
        return Err(Error::UnsupportedOperation(2));
    }
    for value in [quotient, remainder] {
        if value.0 == 0 || loads.iter().any(|load| load.0 == value) {
            return Err(Error::InvalidValue(value));
        }
    }
    if quotient == remainder {
        return Err(Error::InvalidValue(remainder));
    }
    let mut operand_inputs = [0; 2];
    let mut operand_indices = [index; 2];
    for (slot, value) in [lhs, rhs].into_iter().enumerate() {
        let load = loads
            .iter()
            .find(|load| load.0 == value)
            .ok_or(Error::InvalidValue(value))?;
        operand_inputs[slot] = usize::from(load.1 != loads[0].1);
        operand_indices[slot] = load.2;
    }
    let mut outputs = [loads[0].1; 2];
    let mut stored = [false; 2];
    for (slot, operation) in body[3..].iter().copied().enumerate() {
        let Data::BindingStore {
            binding,
            index: got,
            value,
        } = data(operation, slot + 3)?
        else {
            return Err(Error::UnsupportedOperation(slot + 3));
        };
        if got != index {
            return Err(Error::InvalidIndex(slot + 3));
        }
        if !kernel.bindings.iter().any(|declared| {
            declared.reference() == binding && declared.access != PcuBindingAccess::ReadOnly
        }) {
            return Err(Error::InvalidBinding(binding));
        }
        let result = if value == quotient {
            0
        } else if value == remainder {
            1
        } else {
            return Err(Error::InvalidValue(value));
        };
        if stored[result] {
            return Err(Error::UnsupportedOperation(slot + 3));
        }
        outputs[result] = binding;
        stored[result] = true;
    }
    if outputs[0] == outputs[1] {
        return Err(Error::InvalidBinding(outputs[1]));
    }
    let inputs = [loads[0].1, loads[1].1];
    let mut indexed_inputs = [false; 2];
    for load in loads {
        indexed_inputs[usize::from(load.1 != inputs[0])] |= load.2 != Index::BindingElementZero;
    }
    Ok(CheckedIntegerDivRemOperandSchema {
        inputs,
        input_count: if inputs[0] == inputs[1] { 1 } else { 2 },
        indexed_inputs,
        operand_inputs,
        operand_indices,
        outputs,
    })
}

fn validate_interface(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    scalar_caps: PcuValueTypeCaps,
) -> Result<(), Error> {
    if !crate::map_validation::typed_dispatch::is_supported_checked_integer(value_type)
        || !kernel.ports.is_empty()
        || !kernel.parameters.is_empty()
        || !(3..=4).contains(&kernel.bindings.len())
    {
        return Err(Error::UnsupportedInterface);
    }
    let types = scalar_caps | PcuValueTypeCaps::SCALAR_VALUES;
    let features =
        PcuDispatchFeatureCaps::MUTABLE_RESOURCES | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES;
    // Guard flat-map admission before recursive capability scanners.
    if kernel.has_nested_grid_stride_loop() {
        return Err(Error::UnsupportedOperation(0));
    }
    if kernel.required_type_support().bits() & !types.bits() != 0
        || kernel.required_feature_support().bits() & !features.bits() != 0
        || kernel.numerical_requirements.range_policy != PcuRangePolicy::Reject
    {
        return Err(Error::UnsupportedRequirements);
    }
    let mut writable = 0;
    for (slot, binding) in kernel.bindings.iter().enumerate() {
        let reference = binding.reference();
        if kernel.bindings[..slot]
            .iter()
            .any(|prior| prior.reference() == reference)
        {
            return Err(Error::DuplicateBinding(reference));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(value_type)
        {
            return Err(Error::InvalidBinding(reference));
        }
        writable += usize::from(binding.access != PcuBindingAccess::ReadOnly);
    }
    if writable != 2 {
        return Err(Error::UnsupportedInterface);
    }
    Ok(())
}

fn map_body<'a>(kernel: &PcuDispatchKernelIr<'a>) -> Result<(&'a [Op<'a>], Index), Error> {
    if kernel.entry.logical_shape[0] == 0 || kernel.entry.logical_shape[1..] != [1, 1] {
        return Err(Error::UnsupportedInterface);
    }
    let (body, index) = match kernel.ops {
        [
            Op::GridStrideLoop { extent, body },
            Op::Control(PcuDispatchControlOp::Return),
        ] => {
            if *extent == 0 {
                return Err(Error::UnsupportedOperation(0));
            }
            (*body, Index::GridStrideId)
        }
        ops if matches!(ops.last(), Some(Op::Control(PcuDispatchControlOp::Return))) => {
            (&ops[..ops.len() - 1], Index::InvocationId)
        }
        _ => return Err(Error::MissingReturn),
    };
    if body.len() != 5 {
        return Err(Error::UnsupportedOperation(body.len()));
    }
    Ok((body, index))
}

const fn data(operation: Op<'_>, slot: usize) -> Result<Data, Error> {
    if let Op::Data(data) = operation {
        Ok(data)
    } else {
        Err(Error::UnsupportedOperation(slot))
    }
}

fn read_load(
    kernel: &PcuDispatchKernelIr<'_>,
    operation: Op<'_>,
    index: Index,
    slot: usize,
) -> Result<(Id, PcuBindingRef, Index), Error> {
    let Data::BindingLoad {
        result,
        binding,
        index: got,
    } = data(operation, slot)?
    else {
        return Err(Error::UnsupportedOperation(slot));
    };
    if result.0 == 0 {
        return Err(Error::InvalidValue(result));
    }
    if got != index && got != Index::BindingElementZero {
        return Err(Error::InvalidIndex(slot));
    }
    if !kernel.bindings.iter().any(|declared| {
        declared.reference() == binding && declared.access == PcuBindingAccess::ReadOnly
    }) {
        return Err(Error::InvalidBinding(binding));
    }
    Ok((result, binding, got))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
