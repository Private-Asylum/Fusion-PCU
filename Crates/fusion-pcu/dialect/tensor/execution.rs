//! Backend-neutral behavior of a prepared, bound tensor execution.

use core::num::NonZeroUsize;

#[rustfmt::skip]
use super::{
    Tensor,
    ValueId,
};

/// Executes a prepared graph using bindings owned by the selected backend.
///
/// Consumers change values and request work; resource validation and reuse belong to the
/// implementation. Content updates may reuse stable bindings. Changed bindings must be validated
/// before they can participate in execution. Device selection remains a separate caller decision.
/// Backends can expose these operations as inherent methods so ordinary use needs no trait import.
pub trait TensorExecution {
    /// Binding, execution, or readback failure reported by the selected backend.
    type Error;

    /// Updates an initial input's contents without changing its graph shape.
    ///
    /// Starting an update invalidates availability of outputs from the preceding execution,
    /// including when the update fails. An error does not promise unchanged input contents.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown input, incompatible shape, or failed transfer.
    fn update_input(&mut self, input: ValueId, value: &Tensor) -> Result<(), Self::Error>;

    /// Executes one step from the current initial inputs.
    ///
    /// # Errors
    ///
    /// Returns a binding, scheduling, or completion error. Previous outputs are unavailable once
    /// a new execution begins, and new outputs become available only after successful completion.
    fn execute(&mut self) -> Result<(), Self::Error> {
        self.execute_steps(NonZeroUsize::MIN)
    }

    /// Executes a positive number of steps using the prepared feedback relationships.
    ///
    /// Every call starts from current initial inputs. Feedback connects steps within this call;
    /// it does not silently carry the last call's outputs into a new call.
    ///
    /// # Errors
    ///
    /// Returns a binding, scheduling, or completion error. Implementations retain ownership of
    /// pending resources until completion or quarantine and reject unsafe storage reuse.
    fn execute_steps(&mut self, steps: NonZeroUsize) -> Result<(), Self::Error>;

    /// Reads a selected output of the latest successfully completed execution into host memory.
    ///
    /// This is an explicit readback; successful execution need not otherwise copy outputs to RAM.
    ///
    /// # Errors
    ///
    /// Returns an error if the output is unknown, unavailable, or cannot be transferred.
    fn read_output(&mut self, output: ValueId) -> Result<Tensor, Self::Error>;
}
