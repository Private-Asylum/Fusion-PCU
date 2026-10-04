//! Immutable typed source producers; scalar storage does not imply backend admission.
#[rustfmt::skip]
use super::{
    PcuExecutionError,
    PcuTensorGraphCapture,
    PcuTensorGraphValue,
    PcuScalar,
};
#[rustfmt::skip]
use alloc::{
    vec,
    vec::Vec,
};
#[cfg(feature = "tensor")]
#[rustfmt::skip]
use crate::dialect::tensor::{
    Tensor,
    TensorElement,
};
#[cfg(feature = "tensor")]
use core::marker::PhantomData;

mod sealed {
    pub trait Payload<T> {}
    impl<T, const N: usize> Payload<T> for &[T; N] {}
    impl<T, const R: usize, const C: usize> Payload<T> for &[[T; C]; R] {}
}

/// Sealed borrowed scalar arrays accepted by the generated immutable producer.
///
/// The scalar parameter distinguishes a vector from a matrix without treating a row as
/// a device scalar. Conversion occurs only during cold source capture. This is not a
/// runtime upload/import contract or permission to evaluate exceptional floating bits.
#[doc(hidden)]
pub trait PcuImmutableTensorPayload<T: PcuScalar>: sealed::Payload<T> {
    /// Retain the declared shape and flatten initialized scalars in row-major order.
    ///
    /// # Errors
    /// Returns `InvalidTensorSourcePlan` if the declared element count overflows usize.
    fn into_tensor_parts(self) -> Result<(Vec<usize>, Vec<T>), PcuExecutionError>;
}

impl<T: PcuScalar, const N: usize> PcuImmutableTensorPayload<T> for &[T; N] {
    fn into_tensor_parts(self) -> Result<(Vec<usize>, Vec<T>), PcuExecutionError> {
        Ok((vec![N], self.to_vec()))
    }
}

impl<T: PcuScalar, const R: usize, const C: usize> PcuImmutableTensorPayload<T> for &[[T; C]; R] {
    fn into_tensor_parts(self) -> Result<(Vec<usize>, Vec<T>), PcuExecutionError> {
        matrix_parts(self)
    }
}

fn matrix_parts<T: PcuScalar, const R: usize, const C: usize>(
    payload: &[[T; C]; R],
) -> Result<(Vec<usize>, Vec<T>), PcuExecutionError> {
    let elements = R
        .checked_mul(C)
        .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
    let mut data = Vec::with_capacity(elements);
    for row in payload {
        data.extend_from_slice(row);
    }
    Ok((vec![R, C], data))
}

impl PcuTensorGraphCapture {
    /// Captures an immutable vector or matrix payload evaluated in Rust inline const.
    ///
    /// # Errors
    /// Returns shape overflow or graph-policy construction errors. Providers independently
    /// reject unsupported zero extents, scalar representations and numerical profiles.
    #[doc(hidden)]
    #[cfg(feature = "tensor")]
    pub fn constant_array<T: TensorElement, P: PcuImmutableTensorPayload<T>>(
        &mut self,
        payload: P,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        let (shape, data) = payload.into_tensor_parts()?;
        let tensor = Tensor::new(shape, data).map_err(super::super::tensor_build_error)?;
        let value = self.graph.constant_typed(tensor);
        self.graph
            .set_value_numerical_options(value.erase(), self.numerical_options.get())
            .map_err(super::super::tensor_build_error)?;
        Ok(PcuTensorGraphValue {
            value,
            capture_id: self.capture_id,
            marker: PhantomData,
        })
    }

    /// Captures an immutable scalar splat using only the anchor's logical shape.
    /// The anchor remains a shape witness; it is not added as a semantic data consumer.
    ///
    /// # Errors
    /// Returns foreign-capture, shape overflow or graph-policy construction errors.
    #[doc(hidden)]
    #[cfg(feature = "tensor")]
    pub fn uniform_like<T: TensorElement>(
        &mut self,
        anchor: PcuTensorGraphValue<T>,
        payload: T,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        self.validate_value(anchor)?;
        let shape = self
            .graph
            .shape(anchor.value.erase())
            .map_err(super::super::tensor_build_error)?
            .to_vec();
        let value = self
            .graph
            .uniform_typed(shape, payload)
            .map_err(super::super::tensor_build_error)?;
        self.graph
            .set_value_numerical_options(value.erase(), self.numerical_options.get())
            .map_err(super::super::tensor_build_error)?;
        Ok(PcuTensorGraphValue {
            value,
            capture_id: self.capture_id,
            marker: PhantomData,
        })
    }
}

#[cfg(not(feature = "tensor"))]
#[allow(clippy::missing_const_for_fn)] // Preserve the enabled non-const graph-construction API in disabled configurations.
impl PcuTensorGraphCapture {
    /// Disabled tensor configuration retains source type checking without claiming execution.
    ///
    /// # Errors
    /// Always returns TensorExecutionUnavailable.
    #[doc(hidden)]
    pub fn constant_array<T: PcuScalar, P: PcuImmutableTensorPayload<T>>(
        &mut self,
        _: P,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        Err(PcuExecutionError::TensorExecutionUnavailable)
    }

    /// Disabled tensor configuration retains source type checking without claiming execution.
    ///
    /// # Errors
    /// Always returns TensorExecutionUnavailable.
    #[doc(hidden)]
    pub fn uniform_like<T: PcuScalar>(
        &mut self,
        _: PcuTensorGraphValue<T>,
        _: T,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        Err(PcuExecutionError::TensorExecutionUnavailable)
    }
}

#[cfg(all(test, feature = "tensor"))]
#[path = "tests/tests.rs"]
mod tests;
