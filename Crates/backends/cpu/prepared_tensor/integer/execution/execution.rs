//! Cold selected integer runners over indexed retained tensor storage.
#[rustfmt::skip]
use alloc::vec::Vec;
#[rustfmt::skip]
use fusion_pcu::{PcuCheckedInteger,PcuExecutionFaultKind};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{NodeDescriptor,OpDescriptor,TensorElement,TensorError,ValueId};
use super::PcuCpuTensorBinding;
type Operation<T> = fn(T, T) -> Result<T, PcuExecutionFaultKind>;
pub(super) struct Step<T> {
    operation: Operation<T>,
    value: ValueId,
    out: usize,
    left: usize,
    right: usize,
    count: usize,
}
pub(super) fn compile<T: TensorElement + PcuCheckedInteger>(
    node: NodeDescriptor<'_>,
    bindings: &[PcuCpuTensorBinding],
) -> Result<Option<Step<T>>, TensorError> {
    let (left, right, operation): (ValueId, ValueId, Operation<T>) = match node.op {
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => {
            return Ok(None);
        }
        OpDescriptor::Add { left, right } => (left, right, T::pcu_checked_add),
        OpDescriptor::Sub { left, right } => (left, right, T::pcu_checked_sub),
        OpDescriptor::Mul { left, right } => (left, right, T::pcu_checked_mul),
        _ => {
            return Err(TensorError::UnsupportedScalarType {
                value: node.value,
                scalar_type: node.scalar_type,
            });
        }
    };
    let find = |value| {
        bindings
            .iter()
            .find(|b| b.value == value)
            .ok_or(TensorError::UnknownValue(value))
    };
    let output = find(node.value)?;
    let left = find(left)?.slot;
    let right = find(right)?.slot;
    // Inclusive core liveness excludes operand/output overlap. Keep this invariant cold.
    if output.slot == left || output.slot == right {
        return Err(TensorError::UnknownValue(node.value));
    }
    Ok(Some(Step {
        operation,
        value: node.value,
        out: output.slot,
        left,
        right,
        count: output.element_count,
    }))
}
impl<T: PcuCheckedInteger> Step<T> {
    #[allow(clippy::needless_range_loop)] // Indexed shared slots permit repeated operands without unsafe split borrows.
    pub(super) fn execute(&self, storage: &mut [Vec<T>]) -> Result<(), TensorError> {
        for index in 0..self.count {
            let value = (self.operation)(storage[self.left][index], storage[self.right][index])
                .map_err(|kind| TensorError::ArithmeticFault {
                    value: self.value,
                    element_index: index,
                    kind,
                })?;
            storage[self.out][index] = value;
        }
        Ok(())
    }
}
