//! Static two-input behavior; arithmetic families retain their own exact native contracts.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxBinaryInput,
    MlxEncodedCompletion,
    MlxError,
    MlxHostKernelError,
    MlxPreparedBinaryHostKernel,
    MlxPreparedIntegerHostKernel,
};
#[rustfmt::skip]
use crate::{
    PcuBindingRef,
    PcuPreparedHostKernel,
    PcuScalarType,
};

// This trait is internal and monomorphized. The retained MlxKernel enum chooses
// the concrete family; no vtable, discovery or new warm allocation is involved.
pub(super) trait TwoInputKernel: PcuPreparedHostKernel<Error = MlxHostKernelError> {
    fn input_bindings(&self) -> &[PcuBindingRef];
    fn input_element_counts(&self) -> [usize; 2];
    fn output_binding(&self) -> PcuBindingRef;
    fn output_element_count(&self) -> usize;
    fn scalar_type(&self) -> PcuScalarType;
    fn execute_inputs(
        &mut self,
        inputs: &[MlxBinaryInput<'_>],
    ) -> Result<MlxEncodedCompletion, MlxError>;
}

macro_rules! implementation {
    ($ty:ty) => {
        impl TwoInputKernel for $ty {
            fn input_bindings(&self) -> &[PcuBindingRef] {
                Self::input_bindings(self)
            }
            fn input_element_counts(&self) -> [usize; 2] {
                Self::input_element_counts(self)
            }
            fn output_binding(&self) -> PcuBindingRef {
                Self::output_binding(self)
            }
            fn output_element_count(&self) -> usize {
                Self::output_element_count(self)
            }
            fn scalar_type(&self) -> PcuScalarType {
                Self::scalar_type(self)
            }
            fn execute_inputs(
                &mut self,
                inputs: &[MlxBinaryInput<'_>],
            ) -> Result<MlxEncodedCompletion, MlxError> {
                Self::execute_inputs(self, inputs)
            }
        }
    };
}
implementation!(MlxPreparedBinaryHostKernel);
implementation!(MlxPreparedIntegerHostKernel);
