//! Exact same-session representation views, retained by the source owner.
//!
//! A view changes the native descriptor, not the logical dtype, shape or
//! ownership. The backend verifies shared Data identity and byte extent at
//! terminal completion. No host readback or numerical conversion is used.

use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxArray,
    MlxEncodedArray,
    MlxProgramInput,
};
#[rustfmt::skip]
use crate::{
    PcuScalar,
    PcuScalarType,
    global::{
        PcuArgumentError,
        PcuExecutionError,
        arguments::{MlxSourceRoot, TensorBacking},
        tensor::{PcuTensor, PcuTensorInput, TensorInputKind},
    },
};

pub(in crate::global::tensor::opaque) fn resident_root<T: PcuScalar>(
    owner: &PcuTensor<T>,
) -> Result<&Rc<MlxSourceRoot>, PcuExecutionError> {
    owner
        .validate_initialized()
        .map_err(PcuExecutionError::Argument)?;
    let (root, same_session) = match &owner.backing {
        TensorBacking::Mlx { array, root, .. } => {
            (root, root.session.same_session(array.session()))
        }
        TensorBacking::MlxEncoded { array, root, .. } => (root, array.same_session(&root.session)),
        #[cfg(any(
            feature = "cpu",
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "vulkan"
        ))]
        _ => {
            return Err(PcuExecutionError::Argument(
                PcuArgumentError::UnsupportedResidentBorrow,
            ));
        }
    };
    if !same_session {
        return Err(PcuExecutionError::Argument(
            PcuArgumentError::SessionMismatch,
        ));
    }
    Ok(root)
}

pub(in crate::global::tensor::opaque) enum NativeInput<'a, T: PcuScalar> {
    Host(&'a [T]),
    Borrowed(&'a MlxArray),
    View(MlxArray),
}

impl<T: PcuScalar> NativeInput<'_, T> {
    pub(in crate::global::tensor::opaque) const fn as_input(&self) -> MlxProgramInput<'_, T> {
        match self {
            Self::Host(values) => MlxProgramInput::Host(values),
            Self::Borrowed(array) => MlxProgramInput::Resident(array),
            Self::View(array) => MlxProgramInput::Resident(array),
        }
    }
}

pub(in crate::global::tensor::opaque) fn bind_input<'a, T: PcuScalar>(
    input: &PcuTensorInput<'a, T>,
) -> Result<NativeInput<'a, T>, PcuExecutionError> {
    match input.kind {
        TensorInputKind::Host(values) => Ok(NativeInput::Host(values)),
        TensorInputKind::Resident(owner) => {
            resident_root(owner)?;
            match &owner.backing {
                TensorBacking::Mlx { array, .. } => Ok(NativeInput::Borrowed(array)),
                TensorBacking::MlxEncoded { array, shape, .. } => {
                    if T::TYPE != PcuScalarType::F32 {
                        return Err(PcuExecutionError::Argument(
                            PcuArgumentError::UnsupportedResidentBorrow,
                        ));
                    }
                    let [rows, columns] = shape.as_ref() else {
                        return Err(PcuExecutionError::TensorSourceRankMismatch {
                            expected: 2,
                            actual: shape.len(),
                        });
                    };
                    array
                        .native_f32_view([*rows, *columns])
                        .map(NativeInput::View)
                        .map_err(PcuExecutionError::MlxExecution)
                }
                #[cfg(any(
                    feature = "cpu",
                    feature = "rocm",
                    feature = "cuda",
                    feature = "metal",
                    feature = "vulkan"
                ))]
                _ => Err(PcuExecutionError::Argument(
                    PcuArgumentError::UnsupportedResidentBorrow,
                )),
            }
        }
    }
}

pub(in crate::global::tensor::opaque) enum EncodedInput<'a> {
    Borrowed(&'a MlxEncodedArray),
    View(MlxEncodedArray),
}

impl EncodedInput<'_> {
    pub(in crate::global::tensor::opaque) const fn array(&self) -> &MlxEncodedArray {
        match self {
            Self::Borrowed(array) => array,
            Self::View(array) => array,
        }
    }
}

pub(in crate::global::tensor::opaque) fn encoded_input<T: PcuScalar>(
    owner: &PcuTensor<T>,
) -> Result<EncodedInput<'_>, PcuExecutionError> {
    resident_root(owner)?;
    match &owner.backing {
        TensorBacking::MlxEncoded { array, .. } => Ok(EncodedInput::Borrowed(array)),
        TensorBacking::Mlx { array, .. } => {
            if T::TYPE != PcuScalarType::F32 {
                return Err(PcuExecutionError::Argument(
                    PcuArgumentError::UnsupportedResidentBorrow,
                ));
            }
            array
                .encoded_f32_view()
                .map(EncodedInput::View)
                .map_err(PcuExecutionError::MlxExecution)
        }
        #[cfg(any(
            feature = "cpu",
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "vulkan"
        ))]
        _ => Err(PcuExecutionError::Argument(
            PcuArgumentError::UnsupportedResidentBorrow,
        )),
    }
}
