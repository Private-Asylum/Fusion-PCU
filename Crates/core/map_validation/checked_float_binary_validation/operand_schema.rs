//! Cold operand roles for repeated reads and declaration-order-independent float maps.

#[rustfmt::skip]
use super::{
    validate_canonical_binary_kernel,
    CheckedFloatBinaryMapValidationError as Error,
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchDataOp,
    PcuDispatchFloatBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuFloatUnderflowPolicy,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Detached input and output roles of a bounded checked floating-point binary map.
///
/// Inputs retain each distinct binding actually loaded, in load order. Rust argument
/// declarations may contain an unused read-only binding. An operand slot identifies
/// its input independently of declaration order and the other operand's index.
/// Providers must opt into this profile and freeze these roles during preparation;
/// structural admission does not certify executable support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedFloatBinaryOperandSchema {
    inputs: [PcuBindingRef; 2],
    input_count: usize,
    indexed_inputs: [bool; 2],
    operand_inputs: [usize; 2],
    operand_indices: [PcuDispatchIndex; 2],
    output: PcuBindingRef,
}

impl CheckedFloatBinaryOperandSchema {
    /// Distinct bindings actually read; a repeated input occupies just one slot.
    #[must_use]
    pub fn input_bindings(&self) -> &[PcuBindingRef] {
        &self.inputs[..self.input_count]
    }

    /// Operand input slots, in mathematical left/right order.
    #[must_use]
    pub const fn operand_inputs(&self) -> [usize; 2] {
        self.operand_inputs
    }

    /// Each operand's actual load index, including independent scalar broadcast.
    #[must_use]
    pub const fn operand_indices(&self) -> [PcuDispatchIndex; 2] {
        self.operand_indices
    }

    /// Distinct input extent requirements for the prepared logical element count.
    ///
    /// A binding read both at index zero and at an invocation index needs the full
    /// extent. No unused Rust argument needs an upload or fabricated second buffer.
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

    /// The unique writable binding receiving the checked result.
    #[must_use]
    pub const fn output_binding(&self) -> PcuBindingRef {
        self.output
    }
}

/// Assesses two typed loads, one checked binary operation and one indexed store.
///
/// One or two homogeneous read-only declarations and one writable declaration
/// may appear in any order. Loads may repeat a binding, and mathematical operands
/// may repeat or reverse the loaded SSA values. Direct and grid-stride indices,
/// including independent index-zero broadcast, retain their exact meaning.
/// This is an opt-in extension: the existing distinct-input validator stays narrow.
///
/// # Errors
///
/// Rejects malformed binding roles, unsupported capabilities or body shape,
/// invalid indices and invalid typed SSA. No backend or device work is performed.
#[allow(clippy::too_many_lines)] // One bounded cold assessment owns role normalization and SSA admission.
pub fn assess_checked_float_binary_operands(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    op: PcuDispatchFloatBinaryOp,
    underflow_policy: PcuFloatUnderflowPolicy,
    scalar_caps: PcuValueTypeCaps,
) -> Result<CheckedFloatBinaryOperandSchema, Error> {
    if !(2..=3).contains(&kernel.bindings.len()) {
        return Err(Error::UnsupportedInterface);
    }
    let mut bindings = [kernel.bindings[0]; 3];
    let mut read_count = 0;
    let mut writable = None;
    for binding in kernel.bindings {
        if binding.access == PcuBindingAccess::ReadOnly {
            if read_count == 2 {
                return Err(Error::InvalidBinding(binding.reference()));
            }
            bindings[read_count] = *binding;
            read_count += 1;
        } else if writable.replace(*binding).is_some() {
            return Err(Error::InvalidBinding(binding.reference()));
        }
    }
    let writable = writable.ok_or(Error::UnsupportedInterface)?;
    if read_count == 0 {
        return Err(Error::UnsupportedInterface);
    }
    bindings[read_count] = writable;
    let mut canonical = *kernel;
    canonical.bindings = &bindings[..=read_count];
    validate_canonical_binary_kernel(
        &canonical,
        value_type,
        op,
        underflow_policy,
        scalar_caps,
        true,
    )?;
    let body = match kernel.ops.first() {
        Some(PcuDispatchOp::GridStrideLoop { body, .. }) => *body,
        _ => kernel.ops,
    };
    let mut loads = [(
        crate::PcuDispatchValueId(0),
        writable.reference(),
        PcuDispatchIndex::InvocationId,
    ); 2];
    for (slot, load) in loads.iter_mut().enumerate() {
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index,
        }) = body[slot]
        else {
            return Err(Error::UnsupportedOperation(slot));
        };
        *load = (result, binding, index);
    }
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { lhs, rhs, .. }) = body[2]
    else {
        return Err(Error::UnsupportedOperation(2));
    };
    let input_count = if loads[0].1 == loads[1].1 { 1 } else { 2 };
    let inputs = [loads[0].1, loads[1].1];
    let mut indexed_inputs = [false; 2];
    for load in loads {
        let slot = usize::from(load.1 != inputs[0]);
        indexed_inputs[slot] |= load.2 != PcuDispatchIndex::BindingElementZero;
    }
    let mut operand_inputs = [0; 2];
    let mut operand_indices = [PcuDispatchIndex::InvocationId; 2];
    for (slot, value) in [lhs, rhs].into_iter().enumerate() {
        let load = loads
            .iter()
            .find(|load| load.0 == value)
            .ok_or(Error::InvalidValue(value))?;
        operand_inputs[slot] = usize::from(load.1 != inputs[0]);
        operand_indices[slot] = load.2;
    }
    Ok(CheckedFloatBinaryOperandSchema {
        inputs,
        input_count,
        indexed_inputs,
        operand_inputs,
        operand_indices,
        output: writable.reference(),
    })
}
