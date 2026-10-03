//! Static behavior shared by immutable unary arithmetic and exact carrier transport.
//!
//! The caller is generic over a concrete prepared kernel. No trait object,
//! rediscovery or native descriptor construction enters the warm borrow path.

#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxEncodedArray,
    MlxEncodedCompletion,
    MlxError,
    MlxHostKernelError,
    MlxPreparedCarrierHostKernel,
    MlxPreparedHostKernel,
};
#[rustfmt::skip]
use crate::{
    PcuBindingRef,
    PcuPreparedHostKernel,
    PcuScalarType,
};

pub(super) trait SingleInputKernel:
    PcuPreparedHostKernel<Error = MlxHostKernelError>
{
    fn input_binding(&self) -> PcuBindingRef;
    fn output_binding(&self) -> PcuBindingRef;
    fn unused_bindings(&self) -> &[PcuBindingRef] {
        &[]
    }
    fn scalar_type(&self) -> PcuScalarType;
    fn input_byte_len(&self) -> usize;
    fn output_byte_len(&self) -> usize;
    fn execute_encoded_bytes(
        &mut self,
        scalar: PcuScalarType,
        bytes: &[u8],
    ) -> Result<MlxEncodedCompletion, MlxError>;
    fn execute_resident(
        &mut self,
        input: &MlxEncodedArray,
    ) -> Result<MlxEncodedCompletion, MlxError>;
}

impl SingleInputKernel for MlxPreparedHostKernel {
    fn input_binding(&self) -> PcuBindingRef {
        Self::input_binding(self)
    }
    fn output_binding(&self) -> PcuBindingRef {
        Self::output_binding(self)
    }
    fn unused_bindings(&self) -> &[PcuBindingRef] {
        Self::unused_bindings(self)
    }
    fn scalar_type(&self) -> PcuScalarType {
        Self::scalar_type(self)
    }
    fn input_byte_len(&self) -> usize {
        Self::input_byte_len(self)
    }
    fn output_byte_len(&self) -> usize {
        Self::output_byte_len(self)
    }
    fn execute_encoded_bytes(
        &mut self,
        scalar: PcuScalarType,
        bytes: &[u8],
    ) -> Result<MlxEncodedCompletion, MlxError> {
        Self::execute_encoded_bytes(self, scalar, bytes)
    }
    fn execute_resident(
        &mut self,
        input: &MlxEncodedArray,
    ) -> Result<MlxEncodedCompletion, MlxError> {
        Self::execute_resident(self, input)
    }
}

impl SingleInputKernel for MlxPreparedCarrierHostKernel {
    fn input_binding(&self) -> PcuBindingRef {
        Self::input_binding(self)
    }
    fn output_binding(&self) -> PcuBindingRef {
        Self::output_binding(self)
    }
    fn scalar_type(&self) -> PcuScalarType {
        Self::scalar_type(self)
    }
    fn input_byte_len(&self) -> usize {
        Self::input_byte_len(self)
    }
    fn output_byte_len(&self) -> usize {
        Self::output_byte_len(self)
    }
    fn execute_encoded_bytes(
        &mut self,
        scalar: PcuScalarType,
        bytes: &[u8],
    ) -> Result<MlxEncodedCompletion, MlxError> {
        Self::execute_encoded_bytes(self, scalar, bytes)
    }
    fn execute_resident(
        &mut self,
        input: &MlxEncodedArray,
    ) -> Result<MlxEncodedCompletion, MlxError> {
        Self::execute_resident(self, input)
    }
}

/// Project original source bindings into the two actual physical roles. The
/// declaration list was validated and retained cold; unused borrows contribute
/// type/access metadata only, never an upload or native resource.
pub(super) fn collect<'a, const N: usize>(
    input_binding: PcuBindingRef,
    output_binding: PcuBindingRef,
    scalar: PcuScalarType,
    unused: &[PcuBindingRef],
    arguments: [crate::global::arguments::PcuCallArgument<'a>; N],
) -> Result<
    (
        crate::global::arguments::PcuCallArgumentKind<'a>,
        crate::global::arguments::PcuCallArgumentKind<'a>,
    ),
    crate::PcuExecutionError,
> {
    use crate::global::arguments::PcuCallArgumentKind;
    if N.checked_sub(2) != Some(unused.len()) {
        return Err(crate::PcuExecutionError::InvalidTensorSourcePlan);
    }
    let mut input = None;
    let mut output = None;
    let mut seen = [output_binding; N];
    for (position, argument) in arguments.into_iter().enumerate() {
        let kind = argument.into_parts().1;
        let target = match &kind {
            PcuCallArgumentKind::Host(argument) => argument.target(),
            PcuCallArgumentKind::MlxRead(argument) => argument.target,
            PcuCallArgumentKind::MlxWrite(argument) => argument.target,
            #[cfg(any(
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                all(feature = "cpu", feature = "tensor"),
                all(feature = "vulkan", feature = "tensor")
            ))]
            _ => {
                return Err(crate::PcuExecutionError::Argument(
                    crate::global::PcuArgumentError::SessionMismatch,
                ));
            }
        };
        if seen[..position].contains(&target) {
            return Err(super::map_mlx_error(
                crate::PcuHostDispatchError::Duplicate(target),
            ));
        }
        seen[position] = target;
        if target == input_binding {
            input = Some(kind);
        } else if target == output_binding {
            output = Some(kind);
        } else if unused.contains(&target) {
            match &kind {
                PcuCallArgumentKind::Host(argument) => {
                    super::validate_host(argument, scalar, crate::PcuBindingAccess::ReadOnly, 0)?;
                }
                PcuCallArgumentKind::MlxRead(argument) => {
                    if argument.array.scalar_type() != scalar {
                        return Err(super::map_mlx_error(
                            crate::PcuHostDispatchError::TypeMismatch(target),
                        ));
                    }
                }
                PcuCallArgumentKind::MlxWrite(_) => {
                    return Err(super::map_mlx_error(
                        crate::PcuHostDispatchError::AccessMismatch(target),
                    ));
                }
                #[cfg(any(
                    feature = "rocm",
                    feature = "cuda",
                    feature = "metal",
                    all(feature = "cpu", feature = "tensor"),
                    all(feature = "vulkan", feature = "tensor")
                ))]
                _ => {
                    return Err(super::map_mlx_error(
                        crate::PcuHostDispatchError::AccessMismatch(target),
                    ));
                }
            }
        } else {
            return Err(super::map_mlx_error(
                crate::PcuHostDispatchError::Unexpected(target),
            ));
        }
    }
    let input = input
        .ok_or_else(|| super::map_mlx_error(crate::PcuHostDispatchError::Missing(input_binding)))?;
    let output = output.ok_or_else(|| {
        super::map_mlx_error(crate::PcuHostDispatchError::Missing(output_binding))
    })?;
    Ok((input, output))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
