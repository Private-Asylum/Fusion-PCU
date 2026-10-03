//! Statically dispatched joint execution; resource roles remain backend metadata.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxBinaryInput,
    MlxEncodedArray,
    MlxError,
    MlxHostKernelError,
    MlxPreparedDivRemHostKernel,
    MlxPreparedDivRemRoleHostKernel,
};
#[rustfmt::skip]
use crate::{
    PcuBindingRef,
    PcuPreparedHostKernel,
    PcuScalarType,
};

// Monomorphized by the retained concrete kernel enum. No trait object, allocation
// or provider lookup is introduced into joint publication.
pub(in crate::global::provider_hosted::mlx) trait JointKernel:
    PcuPreparedHostKernel<Error = MlxHostKernelError>
{
    fn input_bindings(&self) -> &[PcuBindingRef];
    fn input_byte_lengths(&self) -> [usize; 2];
    fn output_bindings(&self) -> [PcuBindingRef; 2];
    fn output_byte_lengths(&self) -> [usize; 2];
    fn scalar_type(&self) -> PcuScalarType;
    fn execute_inputs(
        &mut self,
        inputs: &[MlxBinaryInput<'_>],
    ) -> Result<[MlxEncodedArray; 2], MlxError>;
}

impl JointKernel for MlxPreparedDivRemHostKernel {
    fn input_bindings(&self) -> &[PcuBindingRef] {
        Self::input_bindings(self)
    }
    fn input_byte_lengths(&self) -> [usize; 2] {
        Self::input_byte_lengths(self)
    }
    fn output_bindings(&self) -> [PcuBindingRef; 2] {
        *Self::output_bindings(self)
    }
    fn output_byte_lengths(&self) -> [usize; 2] {
        Self::output_byte_lengths(self)
    }
    fn scalar_type(&self) -> PcuScalarType {
        Self::scalar_type(self)
    }
    fn execute_inputs(
        &mut self,
        inputs: &[MlxBinaryInput<'_>],
    ) -> Result<[MlxEncodedArray; 2], MlxError> {
        Self::execute_inputs(self, inputs)
    }
}

impl JointKernel for MlxPreparedDivRemRoleHostKernel {
    fn input_bindings(&self) -> &[PcuBindingRef] {
        Self::input_bindings(self)
    }
    fn input_byte_lengths(&self) -> [usize; 2] {
        Self::input_byte_lengths(self)
    }
    fn output_bindings(&self) -> [PcuBindingRef; 2] {
        Self::output_bindings(self)
    }
    fn output_byte_lengths(&self) -> [usize; 2] {
        Self::output_byte_lengths(self)
    }
    fn scalar_type(&self) -> PcuScalarType {
        Self::scalar_type(self)
    }
    fn execute_inputs(
        &mut self,
        inputs: &[MlxBinaryInput<'_>],
    ) -> Result<[MlxEncodedArray; 2], MlxError> {
        Self::execute_inputs(self, inputs)
    }
}
