//! Typed owned tensor source execution. Graph preparation is cold; escaped results own storage.

use crate::PcuScalar;
#[rustfmt::skip]
use super::{
    FixedArrayShape,
    FixedMatrixShape,
    PcuArgumentError,
    PcuExecutionError,
    PcuHostCallSite,
    PcuSourceShape,
    PcuTensor,
    SliceShape,
};
#[rustfmt::skip]
use core::{
    any::TypeId,
    marker::PhantomData,
};
#[cfg(feature = "tensor")]
#[rustfmt::skip]
use core::sync::atomic::{
    AtomicUsize,
    Ordering,
};
#[cfg(feature = "tensor")]
#[rustfmt::skip]
use crate::dialect::tensor::{
    Graph,
    TensorValueId,
    ValueId,
};
#[cfg(feature = "tensor")]
use alloc::vec::Vec;

#[cfg(feature = "tensor")]
#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
static NEXT_CAPTURE_ID: AtomicUsize = AtomicUsize::new(1);

/// Opaque value token tied to one cold source graph capture.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct PcuTensorGraphValue<T: PcuScalar = f32> {
    #[cfg(feature = "tensor")]
    value: TensorValueId<T>,
    #[cfg(feature = "tensor")]
    capture_id: usize,
    marker: PhantomData<fn() -> T>,
}

/// Linear logical owner for a captured tensor value.
///
/// This token enforces ownership in generated source-call APIs. It retains graph-capture
/// provenance but does not imply runtime allocation ownership, exclusive access, or physical
/// buffer donation.
#[doc(hidden)]
pub struct PcuTensorGraphOwner<T: PcuScalar = f32> {
    value: PcuTensorGraphValue<T>,
}

impl<T: PcuScalar> PcuTensorGraphOwner<T> {
    /// Wraps a captured graph value in a non-copyable logical owner token.
    #[doc(hidden)]
    #[must_use]
    pub const fn from_graph_value(value: PcuTensorGraphValue<T>) -> Self {
        Self { value }
    }

    /// Borrows this owner as a copyable graph value for capture operations.
    #[doc(hidden)]
    #[must_use]
    pub const fn borrowed_graph_value(&self) -> PcuTensorGraphValue<T> {
        self.value
    }

    /// Consumes the logical owner and recovers its graph value token.
    #[doc(hidden)]
    #[must_use]
    pub const fn into_graph_value(self) -> PcuTensorGraphValue<T> {
        self.value
    }
}

/// Cold graph builder passed to generated owned-source companions.
#[doc(hidden)]
pub struct PcuTensorGraphCapture {
    #[cfg(feature = "tensor")]
    graph: Graph,
    #[cfg(feature = "tensor")]
    capture_id: usize,
    #[cfg(feature = "tensor")]
    active_markers: Vec<TypeId>,
    marker: PhantomData<fn() -> ()>,
}

#[cfg_attr(not(feature = "tensor"), allow(clippy::missing_const_for_fn))]
// Executable configurations mutate the graph and recursion stack; disabled stubs are constant.
impl PcuTensorGraphCapture {
    #[cfg(feature = "tensor")]
    #[cfg(test)]
    #[cfg_attr(not(feature = "rocm"), allow(dead_code))]
    fn new<T: PcuScalar, const N: usize>(
        shapes: [PcuSourceShape; N],
    ) -> Result<(Self, [PcuTensorGraphValue<T>; N]), PcuExecutionError> {
        Self::new_with_witnesses(shapes.map(PcuTensorShapeWitness::Static))
    }

    #[cfg(feature = "tensor")]
    #[cfg_attr(not(feature = "rocm"), allow(dead_code))]
    fn new_with_witnesses<T: PcuScalar, const N: usize>(
        shapes: [PcuTensorShapeWitness<'_>; N],
    ) -> Result<(Self, [PcuTensorGraphValue<T>; N]), PcuExecutionError> {
        let capture_id = NEXT_CAPTURE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)?;
        let mut graph = Graph::default();
        let mut values = Vec::with_capacity(N);
        for shape in shapes {
            let value = match shape {
                PcuTensorShapeWitness::Static(PcuSourceShape::Scalar) => graph.input_typed::<T>([]),
                PcuTensorShapeWitness::Static(
                    PcuSourceShape::Slice { length } | PcuSourceShape::FixedArray { length },
                ) => graph.input_typed::<T>([length]),
                PcuTensorShapeWitness::Static(PcuSourceShape::FixedMatrix { rows, columns }) => {
                    graph.input_typed::<T>([rows, columns])
                }
                PcuTensorShapeWitness::Dynamic(dimensions) => graph.input_typed::<T>(dimensions),
            }
            .map_err(super::tensor_build_error)?;
            values.push(PcuTensorGraphValue {
                value,
                capture_id,
                marker: PhantomData,
            });
        }
        let values = values
            .try_into()
            .map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)?;
        Ok((
            Self {
                graph,
                capture_id,
                active_markers: Vec::new(),
                marker: PhantomData,
            },
            values,
        ))
    }

    /// Registers a typed graph input without staging or uploading any tensor data.
    #[doc(hidden)]
    pub fn input<T: PcuScalar>(
        &mut self,
        shape: PcuSourceShape,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            let dimensions = match shape {
                PcuSourceShape::Scalar => alloc::vec::Vec::new(),
                PcuSourceShape::Slice { length } | PcuSourceShape::FixedArray { length } => {
                    alloc::vec![length]
                }
                PcuSourceShape::FixedMatrix { rows, columns } => alloc::vec![rows, columns],
            };
            let value = self
                .graph
                .input_typed::<T>(dimensions)
                .map_err(super::tensor_build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = shape;
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Enters a generated source/helper companion, rejecting recursion and excessive nesting.
    #[doc(hidden)]
    pub fn enter(&mut self, marker: TypeId) -> Result<(), PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            if self.active_markers.contains(&marker) {
                return Err(PcuExecutionError::RecursiveTensorSource);
            }
            if self.active_markers.len() >= 64 {
                return Err(PcuExecutionError::TensorSourceNestingLimit);
            }
            self.active_markers.push(marker);
            Ok(())
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = marker;
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Leaves the most recently entered generated companion.
    #[doc(hidden)]
    pub fn leave(&mut self) {
        #[cfg(feature = "tensor")]
        {
            let _ = self.active_markers.pop();
        }
    }

    /// Captures an identity edge without adding a graph operation.
    #[doc(hidden)]
    pub fn identity<T: PcuScalar>(
        &mut self,
        value: PcuTensorGraphValue<T>,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(value)?;
            Ok(value)
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = value;
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Captures one elementwise ReLU node from an existing value in this graph.
    #[doc(hidden)]
    pub fn relu<T: PcuScalar>(
        &mut self,
        value: PcuTensorGraphValue<T>,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(value)?;
            let value = self
                .graph
                .relu_typed(value.value)
                .map_err(super::tensor_build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = value;
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Captures one elementwise addition after validating both operands belong to this graph.
    #[doc(hidden)]
    pub fn add<T: PcuScalar>(
        &mut self,
        lhs: PcuTensorGraphValue<T>,
        rhs: PcuTensorGraphValue<T>,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(lhs)?;
            self.validate_value(rhs)?;
            let value = self
                .graph
                .add_typed(lhs.value, rhs.value)
                .map_err(super::tensor_build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (lhs, rhs);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Captures one elementwise subtraction after validating both operands belong to this graph.
    #[doc(hidden)]
    pub fn sub<T: PcuScalar>(
        &mut self,
        lhs: PcuTensorGraphValue<T>,
        rhs: PcuTensorGraphValue<T>,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(lhs)?;
            self.validate_value(rhs)?;
            let value = self
                .graph
                .sub_typed(lhs.value, rhs.value)
                .map_err(super::tensor_build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (lhs, rhs);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Captures one elementwise multiplication after validating both operands belong to this graph.
    #[doc(hidden)]
    pub fn mul<T: PcuScalar>(
        &mut self,
        lhs: PcuTensorGraphValue<T>,
        rhs: PcuTensorGraphValue<T>,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(lhs)?;
            self.validate_value(rhs)?;
            let value = self
                .graph
                .mul_typed(lhs.value, rhs.value)
                .map_err(super::tensor_build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (lhs, rhs);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Captures a matrix product after validating both operands belong to this graph.
    #[doc(hidden)]
    pub fn matmul<T: PcuScalar>(
        &mut self,
        lhs: PcuTensorGraphValue<T>,
        rhs: PcuTensorGraphValue<T>,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(lhs)?;
            self.validate_value(rhs)?;
            let value = self
                .graph
                .matmul_typed(lhs.value, rhs.value)
                .map_err(super::tensor_build_error)?;
            Ok(PcuTensorGraphValue {
                value,
                capture_id: self.capture_id,
                marker: PhantomData,
            })
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (lhs, rhs);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Validates that a captured graph value has exactly the declared source rank and extents.
    #[doc(hidden)]
    pub fn require_shape<T: PcuScalar>(
        &self,
        value: PcuTensorGraphValue<T>,
        expected: PcuSourceShape,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(value)?;
            let actual = self
                .graph
                .shape(value.value.erase())
                .map_err(super::tensor_build_error)?;
            let matches = match expected {
                PcuSourceShape::Scalar => actual.is_empty(),
                PcuSourceShape::Slice { length } | PcuSourceShape::FixedArray { length } => {
                    actual == [length]
                }
                PcuSourceShape::FixedMatrix { rows, columns } => actual == [rows, columns],
            };
            if !matches {
                return Err(PcuExecutionError::TensorSourceShapeMismatch {
                    expected,
                    actual: actual.to_vec(),
                });
            }
            Ok(value)
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (value, expected);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    /// Validates a dynamic source parameter's rank without constraining its runtime extents.
    #[doc(hidden)]
    pub fn require_rank<T: PcuScalar>(
        &self,
        value: PcuTensorGraphValue<T>,
        rank: usize,
    ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError> {
        #[cfg(feature = "tensor")]
        {
            self.validate_value(value)?;
            let actual = self
                .graph
                .shape(value.value.erase())
                .map_err(super::tensor_build_error)?;
            if actual.len() != rank {
                return Err(PcuExecutionError::TensorSourceRankMismatch {
                    expected: rank,
                    actual: actual.len(),
                });
            }
            Ok(value)
        }
        #[cfg(not(feature = "tensor"))]
        {
            let _ = (value, rank);
            Err(PcuExecutionError::TensorExecutionUnavailable)
        }
    }

    #[cfg(feature = "tensor")]
    fn validate_value<T: PcuScalar>(
        &self,
        value: PcuTensorGraphValue<T>,
    ) -> Result<(), PcuExecutionError> {
        if value.capture_id != self.capture_id {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        self.graph
            .shape(value.value.erase())
            .map_err(super::tensor_build_error)?;
        Ok(())
    }

    #[cfg(feature = "tensor")]
    #[cfg_attr(not(feature = "rocm"), allow(dead_code))]
    fn finish<T: PcuScalar>(
        self,
        value: PcuTensorGraphValue<T>,
    ) -> Result<(Graph, ValueId), PcuExecutionError> {
        self.validate_value(value)?;
        if !self.active_markers.is_empty() {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        Ok((self.graph, value.value.erase()))
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct PcuTensorInput<'a, T: PcuScalar> {
    #[cfg_attr(not(all(feature = "rocm", feature = "tensor")), allow(dead_code))]
    kind: TensorInputKind<'a, T>,
    #[cfg_attr(not(all(feature = "rocm", feature = "tensor")), allow(dead_code))]
    shape: PcuTensorShapeWitness<'a>,
}

/// Stack carrier for a mixed owned composition with borrowed sources and consumed owners.
#[doc(hidden)]
pub enum PcuTensorCallInput<'a, T: PcuScalar> {
    /// A source descriptor that remains borrowed for the duration of the call.
    Borrowed(PcuTensorInput<'a, T>),
    /// A by-value owner that may transfer only if the selected graph uses it.
    Consumed(PcuTensor<T>),
}

#[derive(Clone, Copy)]
enum PcuTensorShapeWitness<'a> {
    #[cfg_attr(not(feature = "tensor"), allow(dead_code))]
    Static(PcuSourceShape),
    // The provider-off source adapter keeps the same descriptor type but cannot construct it.
    #[cfg_attr(not(all(feature = "rocm", feature = "tensor")), allow(dead_code))]
    #[cfg_attr(not(feature = "tensor"), allow(dead_code))]
    Dynamic(&'a [usize]),
}

/// Shape marker for resident tensors whose rank and extents come from the owner at runtime.
#[doc(hidden)]
pub struct DynamicResidentShape;

#[cfg_attr(not(all(feature = "rocm", feature = "tensor")), allow(dead_code))]
#[derive(Clone, Copy)]
enum TensorInputKind<'a, T: PcuScalar> {
    Host(&'a [T]),
    Resident(&'a PcuTensor<T>),
}

impl<'a, T: PcuScalar> PcuTensorInput<'a, T> {
    const fn host(values: &'a [T], shape: PcuSourceShape) -> Self {
        Self {
            kind: TensorInputKind::Host(values),
            shape: PcuTensorShapeWitness::Static(shape),
        }
    }
    #[cfg(feature = "rocm")]
    const fn resident(owner: &'a PcuTensor<T>, shape: PcuSourceShape) -> Self {
        Self {
            kind: TensorInputKind::Resident(owner),
            shape: PcuTensorShapeWitness::Static(shape),
        }
    }
    #[cfg(feature = "rocm")]
    fn resident_dynamic(owner: &'a PcuTensor<T>) -> Self {
        Self {
            kind: TensorInputKind::Resident(owner),
            shape: PcuTensorShapeWitness::Dynamic(owner.shape()),
        }
    }
}

mod sealed {
    pub trait TensorSource<T, Shape> {}
}

/// Storage borrowed by generated owned-result entries, retaining fixed rank and extents.
#[doc(hidden)]
pub trait PcuTensorSource<T: PcuScalar, Shape = SliceShape>:
    sealed::TensorSource<T, Shape>
{
    /// # Errors
    /// Rejects resident values whose shape, initialization or completion does not permit reads.
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError>;
}

impl<T: PcuScalar> sealed::TensorSource<T, SliceShape> for [T] {}
impl<T: PcuScalar> PcuTensorSource<T, SliceShape> for [T] {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Ok(PcuTensorInput::host(
            self,
            PcuSourceShape::Slice { length: self.len() },
        ))
    }
}
impl<T: PcuScalar, const N: usize> sealed::TensorSource<T, SliceShape> for [T; N] {}
impl<T: PcuScalar, const N: usize> PcuTensorSource<T, SliceShape> for [T; N] {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Ok(PcuTensorInput::host(
            self,
            PcuSourceShape::Slice { length: N },
        ))
    }
}
impl<T: PcuScalar, const N: usize> sealed::TensorSource<T, FixedArrayShape<N>> for [T; N] {}
impl<T: PcuScalar, const N: usize> PcuTensorSource<T, FixedArrayShape<N>> for [T; N] {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Ok(PcuTensorInput::host(
            self,
            PcuSourceShape::FixedArray { length: N },
        ))
    }
}
impl<T: PcuScalar> sealed::TensorSource<T, SliceShape> for alloc::vec::Vec<T> {}
impl<T: PcuScalar> PcuTensorSource<T, SliceShape> for alloc::vec::Vec<T> {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Ok(PcuTensorInput::host(
            self.as_slice(),
            PcuSourceShape::Slice { length: self.len() },
        ))
    }
}

// Closed wrapper implementations preserve native borrows without a blanket Deref bound.
macro_rules! host_container_source {
    ($($container:ident)::+) => {
        impl<T: PcuScalar> sealed::TensorSource<T, SliceShape> for $($container)::+<[T]> {}
        impl<T: PcuScalar> PcuTensorSource<T, SliceShape> for $($container)::+<[T]> {
            fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
                Ok(PcuTensorInput::host(self, PcuSourceShape::Slice { length: self.len() }))
            }
        }
        impl<T: PcuScalar, const N: usize> sealed::TensorSource<T, SliceShape> for $($container)::+<[T; N]> {}
        impl<T: PcuScalar, const N: usize> PcuTensorSource<T, SliceShape> for $($container)::+<[T; N]> {
            fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
                Ok(PcuTensorInput::host(self.as_ref().as_slice(), PcuSourceShape::Slice { length: N }))
            }
        }
        impl<T: PcuScalar, const N: usize> sealed::TensorSource<T, FixedArrayShape<N>> for $($container)::+<[T; N]> {}
        impl<T: PcuScalar, const N: usize> PcuTensorSource<T, FixedArrayShape<N>> for $($container)::+<[T; N]> {
            fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
                Ok(PcuTensorInput::host(self.as_ref().as_slice(), PcuSourceShape::FixedArray { length: N }))
            }
        }
    };
}
host_container_source!(alloc::boxed::Box);
host_container_source!(alloc::rc::Rc);
host_container_source!(alloc::sync::Arc);

macro_rules! host_matrix_source {
    ($($container:ident)::+) => {
        impl<T: PcuScalar, const R: usize, const C: usize>
            sealed::TensorSource<T, FixedMatrixShape<R, C>> for $($container)::+<[[T; C]; R]> {}
        impl<T: PcuScalar, const R: usize, const C: usize>
            PcuTensorSource<T, FixedMatrixShape<R, C>> for $($container)::+<[[T; C]; R]>
        {
            fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
                Ok(PcuTensorInput::host(
                    self.as_ref().as_flattened(),
                    PcuSourceShape::FixedMatrix { rows: R, columns: C },
                ))
            }
        }
    };
}
host_matrix_source!(alloc::boxed::Box);
host_matrix_source!(alloc::rc::Rc);
host_matrix_source!(alloc::sync::Arc);

impl<T: PcuScalar, const R: usize, const C: usize> sealed::TensorSource<T, FixedMatrixShape<R, C>>
    for [[T; C]; R]
{
}
impl<T: PcuScalar, const R: usize, const C: usize> PcuTensorSource<T, FixedMatrixShape<R, C>>
    for [[T; C]; R]
{
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Ok(PcuTensorInput::host(
            self.as_flattened(),
            PcuSourceShape::FixedMatrix {
                rows: R,
                columns: C,
            },
        ))
    }
}

#[cfg(feature = "rocm")]
impl<T: PcuScalar> sealed::TensorSource<T, SliceShape> for PcuTensor<T> {}
#[cfg(feature = "rocm")]
impl<T: PcuScalar> PcuTensorSource<T, SliceShape> for PcuTensor<T> {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        let length = self.device_tensor().buffer().len();
        let shape = PcuSourceShape::Slice { length };
        Ok(PcuTensorInput::resident(self, shape))
    }
}
#[cfg(not(feature = "rocm"))]
impl<T: PcuScalar> sealed::TensorSource<T, SliceShape> for PcuTensor<T> {}
#[cfg(not(feature = "rocm"))]
impl<T: PcuScalar> PcuTensorSource<T, SliceShape> for PcuTensor<T> {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Err(PcuArgumentError::ProviderUnavailable)
    }
}
#[cfg(feature = "rocm")]
impl<T: PcuScalar, const N: usize> sealed::TensorSource<T, FixedArrayShape<N>> for PcuTensor<T> {}
#[cfg(feature = "rocm")]
impl<T: PcuScalar, const N: usize> PcuTensorSource<T, FixedArrayShape<N>> for PcuTensor<T> {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        let shape = PcuSourceShape::FixedArray { length: N };
        Ok(PcuTensorInput::resident(self, shape))
    }
}
#[cfg(not(feature = "rocm"))]
impl<T: PcuScalar, const N: usize> sealed::TensorSource<T, FixedArrayShape<N>> for PcuTensor<T> {}
#[cfg(not(feature = "rocm"))]
impl<T: PcuScalar, const N: usize> PcuTensorSource<T, FixedArrayShape<N>> for PcuTensor<T> {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Err(PcuArgumentError::ProviderUnavailable)
    }
}
#[cfg(feature = "rocm")]
impl<T: PcuScalar, const R: usize, const C: usize> sealed::TensorSource<T, FixedMatrixShape<R, C>>
    for PcuTensor<T>
{
}
#[cfg(not(feature = "rocm"))]
impl<T: PcuScalar, const R: usize, const C: usize> sealed::TensorSource<T, FixedMatrixShape<R, C>>
    for PcuTensor<T>
{
}
#[cfg(not(feature = "rocm"))]
impl<T: PcuScalar, const R: usize, const C: usize> PcuTensorSource<T, FixedMatrixShape<R, C>>
    for PcuTensor<T>
{
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Err(PcuArgumentError::ProviderUnavailable)
    }
}

#[cfg(feature = "rocm")]
impl<T: PcuScalar> sealed::TensorSource<T, DynamicResidentShape> for PcuTensor<T> {}
#[cfg(feature = "rocm")]
impl<T: PcuScalar> PcuTensorSource<T, DynamicResidentShape> for PcuTensor<T> {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Ok(PcuTensorInput::resident_dynamic(self))
    }
}
#[cfg(not(feature = "rocm"))]
impl<T: PcuScalar> sealed::TensorSource<T, DynamicResidentShape> for PcuTensor<T> {}
#[cfg(not(feature = "rocm"))]
impl<T: PcuScalar> PcuTensorSource<T, DynamicResidentShape> for PcuTensor<T> {
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        Err(PcuArgumentError::ProviderUnavailable)
    }
}
#[cfg(feature = "rocm")]
impl<T: PcuScalar, const R: usize, const C: usize> PcuTensorSource<T, FixedMatrixShape<R, C>>
    for PcuTensor<T>
{
    fn as_tensor_source(&self) -> Result<PcuTensorInput<'_, T>, PcuArgumentError> {
        let shape = PcuSourceShape::FixedMatrix {
            rows: R,
            columns: C,
        };
        Ok(PcuTensorInput::resident(self, shape))
    }
}

impl<T: PcuScalar> core::fmt::Debug for PcuTensor<T> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PcuTensor")
            .field("scalar", &T::TYPE)
            .field("shape", &self.shape())
            .finish_non_exhaustive()
    }
}

// Provider-disabled builds retain the same instance API, although no owner can be minted.
#[cfg_attr(not(feature = "rocm"), allow(clippy::unused_self))]
impl<T: PcuScalar> PcuTensor<T> {
    /// Dense logical dimensions; no device transfer occurs.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Provider configurations borrow dynamic shape metadata.
    pub fn shape(&self) -> &[usize] {
        #[cfg(feature = "rocm")]
        {
            self.device_tensor().shape()
        }
        #[cfg(not(feature = "rocm"))]
        {
            &[]
        } // No safe constructor exists when no execution provider is compiled.
    }

    /// Logical scalar element count; no device transfer occurs.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Provider metadata comes through its typed storage view.
    pub fn len(&self) -> usize {
        #[cfg(feature = "rocm")]
        {
            self.device_tensor().buffer().len()
        }
        #[cfg(not(feature = "rocm"))]
        {
            0
        } // The disabled-provider owner is not constructible by consumers.
    }

    /// Whether the logical tensor contains no scalar elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Copies this initialized logical tensor into caller-owned RAM without consuming its owner.
    ///
    /// The destination must contain exactly the tensor's flattened element count. Device storage
    /// stays owned by this value until ordinary Drop or consumption; readback does not free it.
    ///
    /// # Errors
    /// Rejects unavailable backends, invalid initialization, uncertain completion, length mismatch,
    /// or backend transfer failure. No implicit CPU computation or backend migration occurs.
    #[allow(clippy::missing_const_for_fn)] // Executable configurations perform validated device IO.
    pub fn read_into(&self, destination: &mut [T]) -> Result<(), PcuExecutionError> {
        #[cfg(feature = "rocm")]
        {
            self.validate_initialized().map_err(super::argument_error)?;
            let pool = crate::PcuMemoryResource::pool(self.device_tensor().buffer().resource());
            self.session()
                .backend()
                .download_buffer(pool, self.device_tensor().buffer(), destination)
                .map_err(PcuExecutionError::DeviceExecution)
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = destination;
            Err(PcuExecutionError::NoBackendEnabled)
        }
    }
}

/// Generated typed owned-result entry; its graph factory runs only on a cold cache miss.
///
/// # Errors
/// Returns unsupported feature, selection, preparation, allocation or device execution errors.
#[doc(hidden)]
#[allow(clippy::needless_pass_by_value)] // Generated calls transfer the validated source carrier into execution.
#[cfg_attr(
    not(all(feature = "rocm", feature = "tensor")),
    allow(clippy::missing_const_for_fn)
)] // Enabled providers execute runtime work through the same API.
pub fn call_owned_tensor_capture<T: PcuScalar, const N: usize, F>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    inputs: [PcuTensorInput<'_, T>; N],
    build: F,
) -> Result<PcuTensor<T>, PcuExecutionError>
where
    F: FnOnce(
            &mut PcuTensorGraphCapture,
            [PcuTensorGraphValue<T>; N],
        ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
        + 'static,
{
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    {
        execution::call(site, specialization, &inputs, build)
    }
    #[cfg(not(all(feature = "rocm", feature = "tensor")))]
    {
        let _ = (site, specialization, inputs, build);
        Err(PcuExecutionError::TensorExecutionUnavailable)
    }
}

/// Generated consuming entry. Ownership is retained through terminal execution.
///
/// Identity may transfer unchanged backing without requiring exclusivity. Destructive reuse
/// requires both graph legality and physical exclusivity; other operations use the ordinary
/// fresh-output scheduler. Transferring the logical owner is not, on its own, permission to
/// overwrite aliased physical storage.
///
/// # Errors
/// Rejects invalid resident rank/readiness, unavailable providers, preparation or execution
/// failure. Failed execution publishes no new owner; uncertain completion retains its leases.
#[doc(hidden)]
#[allow(clippy::needless_pass_by_value)] // The generated signature deliberately consumes the owner.
#[cfg_attr(
    not(all(feature = "rocm", feature = "tensor")),
    allow(clippy::missing_const_for_fn)
)] // Executable configurations validate and execute through the retained provider.
pub fn call_consumed_tensor_capture<T: PcuScalar, F>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    input: PcuTensor<T>,
    build: F,
) -> Result<PcuTensor<T>, PcuExecutionError>
where
    F: FnOnce(
            &mut PcuTensorGraphCapture,
            [PcuTensorGraphValue<T>; 1],
        ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
        + 'static,
{
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    {
        execution::call_consumed(site, specialization, input, build)
    }
    #[cfg(not(all(feature = "rocm", feature = "tensor")))]
    {
        let _ = (site, specialization, input, build);
        Err(PcuExecutionError::TensorExecutionUnavailable)
    }
}

/// Generated consuming entry for a two-input composition with one by-value resident owner.
///
/// The donor remains borrowed while the selected graph and its input affinity are resolved. If
/// the graph prunes it, the ordinary borrowed-input route runs and the owner is dropped normally.
/// Otherwise the carrier moves into the backend only after source borrows have ended.
///
/// # Errors
/// Returns source validation, selection, preparation, provider, or execution errors. Failed
/// execution publishes no new owner; uncertain completion retains its leases.
#[doc(hidden)]
#[allow(clippy::needless_pass_by_value)] // Generated calls transfer the validated source carrier into execution.
#[cfg_attr(
    not(all(feature = "rocm", feature = "tensor")),
    allow(clippy::missing_const_for_fn)
)] // Enabled providers execute runtime work through the same API.
pub fn call_consumed_pair_tensor_capture<T: PcuScalar, F>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    donor: PcuTensor<T>,
    other: PcuTensorInput<'_, T>,
    donor_index: usize,
    build: F,
) -> Result<PcuTensor<T>, PcuExecutionError>
where
    F: FnOnce(
            &mut PcuTensorGraphCapture,
            [PcuTensorGraphValue<T>; 2],
        ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
        + 'static,
{
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    {
        execution::call_consumed_pair(site, specialization, donor, other, donor_index, build)
    }
    #[cfg(not(all(feature = "rocm", feature = "tensor")))]
    {
        let _ = (site, specialization, donor, other, donor_index, build);
        Err(PcuExecutionError::TensorExecutionUnavailable)
    }
}

/// Generated consuming entry for a composition with two or more by-value resident owners.
///
/// All owners remain borrowed until the captured graph and selected inputs are resolved. Pruned
/// owners are dropped normally. One or two selected owners move to the backend afterward; larger
/// selected sets use the ordinary fresh-output path while their owners remain live.
///
/// # Errors
/// Returns source validation, selection, preparation, provider, or execution errors. Failed
/// execution publishes no new owner; uncertain completion retains its leases.
#[doc(hidden)]
#[allow(clippy::needless_pass_by_value)] // Generated calls transfer the validated owner array.
#[cfg_attr(
    not(all(feature = "rocm", feature = "tensor")),
    allow(clippy::missing_const_for_fn)
)] // Enabled providers execute runtime work through the same API.
pub fn call_consumed_owners_tensor_capture<T: PcuScalar, const N: usize, F>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    inputs: [PcuTensor<T>; N],
    build: F,
) -> Result<PcuTensor<T>, PcuExecutionError>
where
    F: FnOnce(
            &mut PcuTensorGraphCapture,
            [PcuTensorGraphValue<T>; N],
        ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
        + 'static,
{
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    {
        execution::call_consumed_owners::<T, N, F>(site, specialization, inputs, build)
    }
    #[cfg(not(all(feature = "rocm", feature = "tensor")))]
    {
        let _ = (site, specialization, inputs, build);
        Err(PcuExecutionError::TensorExecutionUnavailable)
    }
}

/// Generated entry for larger mixed signatures containing consumed owners.
#[doc(hidden)]
#[allow(clippy::needless_pass_by_value)] // The carrier array owns consumed arguments.
#[cfg_attr(
    not(all(feature = "rocm", feature = "tensor")),
    allow(clippy::missing_const_for_fn)
)] // Enabled providers execute runtime work through the same API.
pub fn call_mixed_consumed_tensor_capture<T: PcuScalar, const N: usize, F>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    inputs: [PcuTensorCallInput<'_, T>; N],
    build: F,
) -> Result<PcuTensor<T>, PcuExecutionError>
where
    F: FnOnce(
            &mut PcuTensorGraphCapture,
            [PcuTensorGraphValue<T>; N],
        ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
        + 'static,
{
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    {
        execution::call_mixed_consumed::<T, N, F>(site, specialization, inputs, build)
    }
    #[cfg(not(all(feature = "rocm", feature = "tensor")))]
    {
        let _ = (site, specialization, inputs, build);
        Err(PcuExecutionError::TensorExecutionUnavailable)
    }
}

#[cfg(all(feature = "rocm", feature = "tensor"))]
mod execution {
    #[rustfmt::skip]
    use super::{
        TypeId,
        PcuArgumentError,
        PcuHostCallSite,
        PcuTensorInput,
        PcuTensorCallInput,
        PcuTensorShapeWitness,
        PcuTensor,
        PcuExecutionError,
        TensorInputKind,
        PcuTensorGraphCapture,
        PcuTensorGraphValue,
        PcuScalar,
    };
    #[rustfmt::skip]
    use crate::{
        PcuSourceShape,
        PcuHostArgument,
        PcuBindingRef,
        PcuMemoryPoolId,
        PcuMemoryProvider,
        PcuOwnedDispatchBackend,
    };
    #[rustfmt::skip]
    use crate::dialect::tensor::{
        TensorArithmeticCapability,
        TensorArithmeticRewritePolicy,
        TensorPointwiseGroupingPolicy,
        ValueId,
    };
    #[rustfmt::skip]
    use fusion_pcu_rocm::{
        RocmMemoryProvider,
        RocmMemoryResource,
        RocmOwnedPreparedTensorGraph,
        RocmTensorAssessor,
    };
    #[rustfmt::skip]
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
        sync::Arc,
        sync::atomic::Ordering,
    };
    use crate::global::session::RocmSession;

    fn tensor_element_count(shape: PcuSourceShape) -> Result<usize, PcuExecutionError> {
        match shape {
            PcuSourceShape::Scalar => Ok(1),
            PcuSourceShape::Slice { length } | PcuSourceShape::FixedArray { length } => Ok(length),
            PcuSourceShape::FixedMatrix { rows, columns } => rows
                .checked_mul(columns)
                .ok_or(PcuExecutionError::InvalidTensorSourcePlan),
        }
    }

    // Only called when a host staging tensor is first allocated on a cold call.
    fn tensor_dimensions(shape: PcuSourceShape) -> Vec<usize> {
        match shape {
            PcuSourceShape::Scalar => Vec::new(),
            PcuSourceShape::Slice { length } | PcuSourceShape::FixedArray { length } => {
                vec![length]
            }
            PcuSourceShape::FixedMatrix { rows, columns } => vec![rows, columns],
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum InputShapeRole {
        Static(PcuSourceShape),
        DynamicResident,
    }

    struct InputShapeKey {
        role: InputShapeRole,
        dimensions: Vec<usize>,
    }

    impl InputShapeKey {
        fn from_witness(witness: PcuTensorShapeWitness<'_>) -> Self {
            match witness {
                PcuTensorShapeWitness::Static(shape) => Self {
                    role: InputShapeRole::Static(shape),
                    dimensions: tensor_dimensions(shape),
                },
                PcuTensorShapeWitness::Dynamic(dimensions) => Self {
                    role: InputShapeRole::DynamicResident,
                    dimensions: dimensions.to_vec(),
                },
            }
        }

        fn matches(&self, witness: PcuTensorShapeWitness<'_>) -> bool {
            match witness {
                PcuTensorShapeWitness::Static(shape) => self.role == InputShapeRole::Static(shape),
                PcuTensorShapeWitness::Dynamic(dimensions) => {
                    self.role == InputShapeRole::DynamicResident && self.dimensions == dimensions
                }
            }
        }
    }

    fn entry_matches_shapes<const N: usize>(
        entry: &Entry,
        shapes: &[PcuTensorShapeWitness<'_>; N],
    ) -> bool {
        entry.shape_keys.len() == N
            && entry
                .shape_keys
                .iter()
                .zip(shapes)
                .all(|(key, witness)| key.matches(*witness))
    }

    struct Entry {
        specialization: TypeId,
        factory: TypeId,
        scalar_type: crate::core::PcuScalarType,
        shape_keys: Vec<InputShapeKey>,
        resident_affinity: bool,
        session: Rc<RocmSession>,
        prepared: RocmOwnedPreparedTensorGraph,
        input_ids: Vec<ValueId>,
        input_indices: Vec<usize>,
        pool: PcuMemoryPoolId,
        memory: RocmMemoryProvider,
        host_inputs: Vec<Option<RocmMemoryResource>>,
    }
    struct BuiltProgram {
        program: Arc<crate::dialect::tensor::TensorOwnedSelectedProgram>,
        input_ids: Vec<ValueId>,
        input_indices: Vec<usize>,
    }
    #[derive(Default)]
    struct State {
        generation: u64,
        entries: Vec<Entry>,
    }
    std::thread_local! {
        static STATE: RefCell<State> = RefCell::new(State::default());
        static LAST_GENERATION: Cell<u64> = const { Cell::new(0) };
    }

    fn synchronize_generation() -> Result<u64, PcuExecutionError> {
        let generation = crate::global::hosted::current_generation();
        LAST_GENERATION
            .try_with(|last| {
                if last.get() != generation {
                    STATE
                        .try_with(|state| {
                            let mut state = state
                                .try_borrow_mut()
                                .map_err(|_| PcuExecutionError::ReentrantCall)?;
                            state.entries.clear();
                            state.generation = generation;
                            Ok(())
                        })
                        .map_err(|_| PcuExecutionError::ThreadUnavailable)??;
                    last.set(generation);
                }
                Ok(generation)
            })
            .map_err(|_| PcuExecutionError::ThreadUnavailable)?
    }

    fn with_state<R>(
        operation: impl FnOnce(&mut State) -> Result<R, PcuExecutionError>,
    ) -> Result<R, PcuExecutionError> {
        STATE
            .try_with(|state| {
                let mut state = state
                    .try_borrow_mut()
                    .map_err(|_| PcuExecutionError::ReentrantCall)?;
                operation(&mut state)
            })
            .map_err(|_| PcuExecutionError::ThreadUnavailable)?
    }

    fn build_program<T: PcuScalar, const N: usize, F>(
        shapes: [PcuTensorShapeWitness<'_>; N],
        build: F,
    ) -> Result<BuiltProgram, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue<T>; N],
            ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
            + 'static,
    {
        let (mut capture, input_values) =
            PcuTensorGraphCapture::new_with_witnesses::<T, N>(shapes)?;
        let captured_input_ids = input_values.map(|value| value.value.erase());
        let output_value = build(&mut capture, input_values)?;
        let (graph, output_id) = capture.finish(output_value)?;
        let program = Arc::new(
            graph
                .into_selected_program(
                    &[output_id],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .map_err(crate::global::tensor_build_error)?,
        );
        let mut input_ids = Vec::with_capacity(program.input_values().len());
        let mut input_indices = Vec::with_capacity(program.input_values().len());
        for &selected_id in program.input_values() {
            let Some(argument_index) = captured_input_ids
                .iter()
                .position(|captured_id| *captured_id == selected_id)
            else {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            };
            input_ids.push(selected_id);
            input_indices.push(argument_index);
        }
        if input_ids.is_empty() {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        Ok(BuiltProgram {
            program,
            input_ids,
            input_indices,
        })
    }

    fn prepare_entry<T: PcuScalar, const N: usize>(
        affinity: Option<&Rc<RocmSession>>,
        specialization: TypeId,
        factory: TypeId,
        shapes: [PcuTensorShapeWitness<'_>; N],
        built: BuiltProgram,
    ) -> Result<(Entry, u64, usize), PcuExecutionError> {
        let (session, prepared, generation, capacity) =
            crate::global::hosted::prepare_tensor(affinity, |session| {
                let prepared = session
                    .tensor_assessor()
                    .map_err(PcuExecutionError::TensorInitialization)?
                    .prepare_shared_owned_program(Arc::clone(&built.program))
                    .map_err(PcuExecutionError::TensorExecution)?;
                Ok(prepared)
            })?;
        let pool = PcuMemoryPoolId(session.backend().device_identity().device_id());
        let memory = session.backend().memory_provider(pool);
        let entry = Entry {
            specialization,
            factory,
            scalar_type: T::TYPE,
            shape_keys: shapes
                .iter()
                .copied()
                .map(InputShapeKey::from_witness)
                .collect(),
            resident_affinity: affinity.is_some(),
            session,
            prepared,
            input_ids: built.input_ids,
            input_indices: built.input_indices,
            pool,
            memory,
            host_inputs: (0..N).map(|_| None).collect(),
        };
        Ok((entry, generation, capacity))
    }

    fn stage_host_input<T: PcuScalar>(
        entry: &mut Entry,
        index: usize,
        values: &[T],
    ) -> Result<(), PcuExecutionError> {
        if let Some(staging) = entry.host_inputs[index].as_mut() {
            let argument = PcuHostArgument::read(PcuBindingRef::new(0, 0), values);
            if let Err(error) = entry.memory.transfer_to(staging, 0, argument.bytes()) {
                // A quarantined allocation is retained by its physical access lease. Do not reuse
                // that failed staging handle on the next call or pin it through this cache entry.
                entry.host_inputs[index] = None;
                return Err(PcuExecutionError::Memory(error));
            }
        } else {
            let buffer = entry
                .session
                .backend()
                .upload_buffer(entry.pool, values)
                .map_err(PcuExecutionError::DeviceExecution)?;
            entry.host_inputs[index] = Some(buffer.into_resource());
        }
        Ok(())
    }

    fn preflight_shapes<'a, T: PcuScalar, const N: usize>(
        inputs: &'a [PcuTensorInput<'a, T>; N],
    ) -> Result<[PcuTensorShapeWitness<'a>; N], PcuExecutionError> {
        if N == 0 {
            return Err(PcuExecutionError::EmptyTensorInput);
        }
        let shapes = core::array::from_fn(|index| inputs[index].shape);
        for (index, shape) in shapes.iter().copied().enumerate() {
            let elements = match shape {
                PcuTensorShapeWitness::Static(shape) => tensor_element_count(shape)?,
                PcuTensorShapeWitness::Dynamic(dimensions) => dimensions
                    .iter()
                    .try_fold(1_usize, |elements, dimension| {
                        elements.checked_mul(*dimension)
                    })
                    .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?,
            };
            if elements == 0 {
                return Err(PcuExecutionError::EmptyTensorInput);
            }
            if let TensorInputKind::Host(values) = &inputs[index].kind
                && values.len() != elements
            {
                return Err(PcuExecutionError::Argument(
                    PcuArgumentError::SourceShapeMismatch {
                        expected: match shape {
                            PcuTensorShapeWitness::Static(shape) => shape,
                            PcuTensorShapeWitness::Dynamic(_) => {
                                return Err(PcuExecutionError::InvalidTensorSourcePlan);
                            }
                        },
                        actual: PcuSourceShape::Slice {
                            length: values.len(),
                        },
                    },
                ));
            }
        }

        Ok(shapes)
    }

    fn preflight_selected_inputs<'a, T: PcuScalar, const N: usize>(
        inputs: &'a [PcuTensorInput<'_, T>; N],
        selected_indices: &[usize],
    ) -> Result<Option<&'a Rc<RocmSession>>, PcuExecutionError> {
        let mut affinity = None;
        for &index in selected_indices {
            let input = inputs
                .get(index)
                .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
            if let TensorInputKind::Resident(owner) = &input.kind {
                let owner_session = owner.session();
                if affinity.is_some_and(|existing| !Rc::ptr_eq(existing, owner_session)) {
                    return Err(PcuExecutionError::Argument(
                        PcuArgumentError::SessionMismatch,
                    ));
                }
                match input.shape {
                    PcuTensorShapeWitness::Static(shape) => owner
                        .validate_read(shape)
                        .map_err(PcuExecutionError::Argument)?,
                    PcuTensorShapeWitness::Dynamic(_) => owner
                        .validate_initialized()
                        .map_err(PcuExecutionError::Argument)?,
                }
                owner
                    .device_tensor()
                    .buffer()
                    .resource()
                    .validate_access_available()
                    .map_err(|_| {
                        PcuExecutionError::Argument(PcuArgumentError::ResidentCompletionUncertain)
                    })?;
                affinity = Some(owner_session);
            }
        }
        Ok(affinity)
    }

    fn execute_entry<T: PcuScalar, const N: usize>(
        entry: &mut Entry,
        inputs: &[PcuTensorInput<'_, T>; N],
    ) -> Result<PcuTensor<T>, PcuExecutionError> {
        // Resident affinity is checked before staging. Retained shape slices keep borrowed
        // descriptors stack-only on warm calls.
        for selected_index in 0..entry.input_indices.len() {
            let index = entry.input_indices[selected_index];
            if let TensorInputKind::Host(values) = &inputs[index].kind {
                stage_host_input(entry, index, values)?;
            }
        }
        let assessor = entry
            .session
            .tensor_assessor()
            .map_err(PcuExecutionError::TensorInitialization)?;
        let mut sources: [Option<fusion_pcu_rocm::RocmTensorInputRef<'_>>; N] =
            core::array::from_fn(|_| None);
        for selected_index in 0..entry.input_indices.len() {
            let index = entry.input_indices[selected_index];
            let input = &inputs[index];
            let source = match &input.kind {
                TensorInputKind::Host(_) => assessor.borrow_resource_input_ref(
                    entry.host_inputs[index]
                        .as_ref()
                        .expect("host input staged before bindings"),
                    &entry.shape_keys[index].dimensions,
                    T::TYPE,
                    entry.pool,
                ),
                TensorInputKind::Resident(owner) => {
                    assessor.borrow_device_input_ref(owner.device_tensor(), entry.pool)
                }
            }
            .map_err(PcuExecutionError::TensorExecution)?;
            sources[index] = Some(source);
        }
        let first_index = *entry
            .input_indices
            .first()
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        let first_binding = (
            entry.input_ids[0],
            sources[first_index]
                .as_ref()
                .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?,
        );
        let mut bindings = [first_binding; N];
        for (binding_index, (&input_id, &source_index)) in
            entry.input_ids.iter().zip(&entry.input_indices).enumerate()
        {
            bindings[binding_index] = (
                input_id,
                sources[source_index]
                    .as_ref()
                    .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?,
            );
        }
        let outputs = assessor
            .execute_owned_program_outputs_from_inputs(
                &entry.prepared,
                &bindings[..entry.input_ids.len()],
                entry.pool,
                &mut entry.memory,
            )
            .map_err(PcuExecutionError::TensorExecution)?;
        let mut outputs = outputs.into_iter();
        let (_, tensor) = outputs
            .next()
            .ok_or(PcuExecutionError::PreparationDidNotProduceKernel)?;
        let extra_output = outputs.next();
        debug_assert!(extra_output.is_none(), "source capture has one output");
        Ok(PcuTensor::from_successful_output(
            tensor,
            Rc::clone(&entry.session),
        ))
    }

    pub(super) fn call<T: PcuScalar, const N: usize, F>(
        site: &PcuHostCallSite,
        specialization: TypeId,
        inputs: &[PcuTensorInput<'_, T>; N],
        build: F,
    ) -> Result<PcuTensor<T>, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue<T>; N],
            ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
            + 'static,
    {
        let shapes = preflight_shapes(inputs)?;
        let generation = synchronize_generation()?;
        with_state(|state| {
            let slot = resolve_entry_slot::<T, N, F>(
                state,
                generation,
                site,
                specialization,
                shapes,
                inputs,
                build,
            )?;
            execute_entry(&mut state.entries[slot], inputs)
        })
    }

    pub(super) fn call_consumed<T: PcuScalar, F>(
        site: &PcuHostCallSite,
        specialization: TypeId,
        input: PcuTensor<T>,
        build: F,
    ) -> Result<PcuTensor<T>, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue<T>; 1],
            ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
            + 'static,
    {
        // The shape witness borrows the owner only while resolving and preflighting the entry.
        // Its lexical scope ends before the physical carrier is moved into the consuming
        // backend API; exclusivity and graph legality remain backend-checked.
        let generation = synchronize_generation()?;
        with_state(|state| {
            let slot = {
                let source = <PcuTensor<T> as super::PcuTensorSource<
                    T,
                    super::DynamicResidentShape,
                >>::as_tensor_source(&input)
                .map_err(crate::global::argument_error)?;
                let inputs = [source];
                let shapes = preflight_shapes(&inputs)?;
                resolve_entry_slot::<T, 1, F>(
                    state,
                    generation,
                    site,
                    specialization,
                    shapes,
                    &inputs,
                    build,
                )?
            };
            let (tensor, retained_session) = input.into_device_parts();
            let entry = &mut state.entries[slot];
            if !Rc::ptr_eq(&retained_session, &entry.session) {
                return Err(PcuExecutionError::Argument(
                    PcuArgumentError::SessionMismatch,
                ));
            }
            let assessor = entry
                .session
                .tensor_assessor()
                .map_err(PcuExecutionError::TensorInitialization)?;
            let output = assessor
                .execute_owned_program_consuming_input(
                    &entry.prepared,
                    tensor,
                    entry.pool,
                    &mut entry.memory,
                )
                .map_err(PcuExecutionError::TensorExecution)?;
            Ok(PcuTensor::from_successful_output(output, retained_session))
        })
    }

    pub(super) fn call_consumed_pair<T: PcuScalar, F>(
        site: &PcuHostCallSite,
        specialization: TypeId,
        donor: PcuTensor<T>,
        other: PcuTensorInput<'_, T>,
        donor_index: usize,
        build: F,
    ) -> Result<PcuTensor<T>, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue<T>; 2],
            ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
            + 'static,
    {
        if donor_index > 1 {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let generation = synchronize_generation()?;
        with_state(|state| {
            let request = EntryRequest {
                generation,
                site,
                specialization,
            };
            let selected = match resolve_consumed_pair::<T, F>(
                state,
                request,
                &donor,
                other,
                donor_index,
                build,
            )? {
                ConsumedPairResolution::Pruned(output) => return Ok(output),
                ConsumedPairResolution::Selected(selected) => selected,
            };

            let (tensor, retained_session) = donor.into_device_parts();
            let entry = &mut state.entries[selected.slot];
            if !Rc::ptr_eq(&retained_session, &entry.session) {
                return Err(PcuExecutionError::Argument(
                    PcuArgumentError::SessionMismatch,
                ));
            }
            let assessor = entry
                .session
                .tensor_assessor()
                .map_err(PcuExecutionError::TensorInitialization)?;
            let output = if let Some(other_value) = selected.other_value {
                let other_index = 1 - donor_index;
                let other_ref = match &other.kind {
                    TensorInputKind::Host(_) => assessor.borrow_resource_input_ref(
                        entry.host_inputs[other_index]
                            .as_ref()
                            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?,
                        &entry.shape_keys[other_index].dimensions,
                        T::TYPE,
                        entry.pool,
                    ),
                    TensorInputKind::Resident(owner) => {
                        assessor.borrow_device_input_ref(owner.device_tensor(), entry.pool)
                    }
                }
                .map_err(PcuExecutionError::TensorExecution)?;
                assessor.execute_owned_program_consuming_binary_input(
                    &entry.prepared,
                    selected.donor_value,
                    tensor,
                    &[(other_value, &other_ref)],
                    entry.pool,
                    &mut entry.memory,
                )
            } else {
                assessor.execute_owned_program_consuming_input(
                    &entry.prepared,
                    tensor,
                    entry.pool,
                    &mut entry.memory,
                )
            }
            .map_err(PcuExecutionError::TensorExecution)?;
            Ok(PcuTensor::from_successful_output(output, retained_session))
        })
    }

    pub(super) fn call_consumed_owners<T: PcuScalar, const N: usize, F>(
        site: &PcuHostCallSite,
        specialization: TypeId,
        inputs: [PcuTensor<T>; N],
        build: F,
    ) -> Result<PcuTensor<T>, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue<T>; N],
            ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
            + 'static,
    {
        if N < 2 {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let generation = synchronize_generation()?;
        with_state(|state| {
            let request = EntryRequest {
                generation,
                site,
                specialization,
            };
            let selected = match resolve_consumed_owners::<T, N, F>(state, request, &inputs, build)?
            {
                ConsumedOwnersResolution::Pruned(output) => return Ok(output),
                ConsumedOwnersResolution::Selected(selected) => selected,
            };
            if selected.count > 2 {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            }
            let selected_owners = take_selected_owner_inputs(inputs, selected)?;
            let entry = &mut state.entries[selected.slot];
            let Entry {
                session,
                prepared,
                pool,
                memory,
                shape_keys,
                host_inputs,
                ..
            } = entry;
            let assessor = session
                .tensor_assessor()
                .map_err(PcuExecutionError::TensorInitialization)?;
            let context = SelectedOwnerExecution {
                session,
                prepared,
                pool: *pool,
                memory,
                shape_keys,
                host_inputs,
            };
            let output = if selected.count == 1 {
                execute_single_selected_owner(&assessor, context, selected_owners)?
            } else if selected.count == 2 {
                execute_both_selected_owners(&assessor, context, selected_owners, selected)?
            } else {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            };
            Ok(output)
        })
    }

    pub(super) fn call_mixed_consumed<T: PcuScalar, const N: usize, F>(
        site: &PcuHostCallSite,
        specialization: TypeId,
        inputs: [PcuTensorCallInput<'_, T>; N],
        build: F,
    ) -> Result<PcuTensor<T>, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue<T>; N],
            ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
            + 'static,
    {
        if N <= 2 {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let generation = synchronize_generation()?;
        with_state(|state| {
            let request = EntryRequest {
                generation,
                site,
                specialization,
            };
            let selected = match resolve_mixed_consumed::<T, N, F>(state, request, &inputs, build)?
            {
                ConsumedMixedResolution::Fresh(output) => return Ok(output),
                ConsumedMixedResolution::Selected(selected) => selected,
            };
            let selected_inputs = take_mixed_selected_inputs(inputs, selected)?;
            let entry = &mut state.entries[selected.slot];
            let Entry {
                session,
                prepared,
                pool,
                memory,
                shape_keys,
                host_inputs,
                ..
            } = entry;
            let assessor = session
                .tensor_assessor()
                .map_err(PcuExecutionError::TensorInitialization)?;
            let context = SelectedOwnerExecution {
                session,
                prepared,
                pool: *pool,
                memory,
                shape_keys,
                host_inputs,
            };
            match (selected.count, selected.owner_count) {
                (1, 1) => {
                    let output =
                        execute_single_selected_owner(&assessor, context, selected_inputs.owners)?;
                    Ok(output)
                }
                (2, 2) => execute_both_selected_owners(
                    &assessor,
                    context,
                    selected_inputs.owners,
                    selected,
                ),
                (2, 1) => execute_owner_with_borrowed_peer(
                    &assessor,
                    context,
                    selected_inputs.owners,
                    selected_inputs.borrowed,
                    selected,
                ),
                _ => Err(PcuExecutionError::InvalidTensorSourcePlan),
            }
        })
    }

    #[derive(Clone, Copy)]
    struct ConsumedOwnersSelection {
        slot: usize,
        input_indices: [usize; 2],
        input_ids: [Option<ValueId>; 2],
        count: usize,
        owner_count: usize,
    }

    enum ConsumedOwnersResolution<T: PcuScalar> {
        Pruned(PcuTensor<T>),
        Selected(ConsumedOwnersSelection),
    }

    enum ConsumedMixedResolution<T: PcuScalar> {
        Fresh(PcuTensor<T>),
        Selected(ConsumedOwnersSelection),
    }

    struct SelectedMixedInputs<'a, T: PcuScalar> {
        owners: [Option<PcuTensor<T>>; 2],
        borrowed: [Option<PcuTensorInput<'a, T>>; 2],
    }

    struct SelectedOwnerExecution<'a> {
        session: &'a Rc<RocmSession>,
        prepared: &'a RocmOwnedPreparedTensorGraph,
        pool: PcuMemoryPoolId,
        memory: &'a mut RocmMemoryProvider,
        shape_keys: &'a [InputShapeKey],
        host_inputs: &'a [Option<RocmMemoryResource>],
    }

    fn resolve_consumed_owners<T: PcuScalar, const N: usize, F>(
        state: &mut State,
        request: EntryRequest<'_>,
        owners: &[PcuTensor<T>; N],
        build: F,
    ) -> Result<ConsumedOwnersResolution<T>, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue<T>; N],
            ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
            + 'static,
    {
        let inputs = owners.each_ref().map(PcuTensorInput::resident_dynamic);
        let shapes = preflight_shapes(&inputs)?;
        let slot = resolve_entry_slot::<T, N, F>(
            state,
            request.generation,
            request.site,
            request.specialization,
            shapes,
            &inputs,
            build,
        )?;
        let entry = &mut state.entries[slot];
        let count = entry.input_indices.len();
        if count == 0 || count > 2 {
            return execute_entry(entry, &inputs).map(ConsumedOwnersResolution::Pruned);
        }
        if entry.input_ids.len() != count {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let mut input_indices = [0; 2];
        let mut input_ids = [None; 2];
        for position in 0..count {
            input_indices[position] = entry.input_indices[position];
            input_ids[position] = Some(entry.input_ids[position]);
        }
        Ok(ConsumedOwnersResolution::Selected(
            ConsumedOwnersSelection {
                slot,
                input_indices,
                input_ids,
                count,
                owner_count: count,
            },
        ))
    }

    fn resolve_mixed_consumed<T: PcuScalar, const N: usize, F>(
        state: &mut State,
        request: EntryRequest<'_>,
        inputs: &[PcuTensorCallInput<'_, T>; N],
        build: F,
    ) -> Result<ConsumedMixedResolution<T>, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue<T>; N],
            ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
            + 'static,
    {
        let sources = inputs.each_ref().map(|input| match input {
            PcuTensorCallInput::Borrowed(source) => *source,
            PcuTensorCallInput::Consumed(owner) => PcuTensorInput::resident_dynamic(owner),
        });
        let shapes = preflight_shapes(&sources)?;
        let slot = resolve_entry_slot::<T, N, F>(
            state,
            request.generation,
            request.site,
            request.specialization,
            shapes,
            &sources,
            build,
        )?;
        let entry = &mut state.entries[slot];
        let count = entry.input_indices.len();
        if count == 0 || count > 2 || entry.input_ids.len() != count {
            return execute_entry(entry, &sources).map(ConsumedMixedResolution::Fresh);
        }
        let mut selected = ConsumedOwnersSelection {
            slot,
            input_indices: [0; 2],
            input_ids: [None; 2],
            count,
            owner_count: 0,
        };
        for position in 0..count {
            let formal_index = entry.input_indices[position];
            let input = inputs
                .get(formal_index)
                .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
            selected.input_indices[position] = formal_index;
            selected.input_ids[position] = Some(entry.input_ids[position]);
            if matches!(input, PcuTensorCallInput::Consumed(_)) {
                selected.owner_count += 1;
            }
        }
        if selected.owner_count == 0 {
            return execute_entry(entry, &sources).map(ConsumedMixedResolution::Fresh);
        }
        for position in 0..count {
            let formal_index = selected.input_indices[position];
            let PcuTensorCallInput::Borrowed(source) = &inputs[formal_index] else {
                continue;
            };
            if let TensorInputKind::Host(values) = &source.kind {
                stage_host_input(entry, formal_index, values)?;
            }
        }
        Ok(ConsumedMixedResolution::Selected(selected))
    }

    fn take_selected_owner_inputs<T: PcuScalar, const N: usize>(
        owners: [PcuTensor<T>; N],
        selected: ConsumedOwnersSelection,
    ) -> Result<[Option<PcuTensor<T>>; 2], PcuExecutionError> {
        let mut selected_owners = [None, None];
        for (formal_index, owner) in owners.into_iter().enumerate() {
            if formal_index == selected.input_indices[0] {
                selected_owners[0] = Some(owner);
            } else if selected.count == 2 && formal_index == selected.input_indices[1] {
                selected_owners[1] = Some(owner);
            } else {
                drop(owner);
            }
        }
        if selected_owners[0].is_none() || (selected.count == 2 && selected_owners[1].is_none()) {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        Ok(selected_owners)
    }

    fn take_mixed_selected_inputs<T: PcuScalar, const N: usize>(
        inputs: [PcuTensorCallInput<'_, T>; N],
        selected: ConsumedOwnersSelection,
    ) -> Result<SelectedMixedInputs<'_, T>, PcuExecutionError> {
        let mut owners = [None, None];
        let mut borrowed = [None, None];
        for (formal_index, input) in inputs.into_iter().enumerate() {
            let position = if formal_index == selected.input_indices[0] {
                Some(0)
            } else if selected.count == 2 && formal_index == selected.input_indices[1] {
                Some(1)
            } else {
                None
            };
            match (position, input) {
                (Some(position), PcuTensorCallInput::Consumed(owner)) => {
                    owners[position] = Some(owner);
                }
                (Some(position), PcuTensorCallInput::Borrowed(source)) => {
                    borrowed[position] = Some(source);
                }
                (None, PcuTensorCallInput::Consumed(owner)) => drop(owner),
                (None, PcuTensorCallInput::Borrowed(_)) => {}
            }
        }
        if owners.iter().filter(|owner| owner.is_some()).count() != selected.owner_count {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        Ok(SelectedMixedInputs { owners, borrowed })
    }

    fn execute_single_selected_owner<T: PcuScalar>(
        assessor: &RocmTensorAssessor<'_>,
        context: SelectedOwnerExecution<'_>,
        owner_inputs: [Option<PcuTensor<T>>; 2],
    ) -> Result<PcuTensor<T>, PcuExecutionError> {
        let SelectedOwnerExecution {
            session,
            prepared,
            pool,
            memory,
            ..
        } = context;
        if owner_inputs[1].is_some() {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let [selected_owner, _] = owner_inputs;
        let (tensor, retained_session) = selected_owner
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?
            .into_device_parts();
        if !Rc::ptr_eq(&retained_session, session) {
            return Err(PcuExecutionError::Argument(
                PcuArgumentError::SessionMismatch,
            ));
        }
        let output = assessor
            .execute_owned_program_consuming_input(prepared, tensor, pool, memory)
            .map_err(PcuExecutionError::TensorExecution)?;
        Ok(PcuTensor::from_successful_output(output, retained_session))
    }

    fn execute_both_selected_owners<T: PcuScalar>(
        assessor: &RocmTensorAssessor<'_>,
        context: SelectedOwnerExecution<'_>,
        owner_inputs: [Option<PcuTensor<T>>; 2],
        selected: ConsumedOwnersSelection,
    ) -> Result<PcuTensor<T>, PcuExecutionError> {
        let SelectedOwnerExecution {
            session,
            prepared,
            pool,
            memory,
            ..
        } = context;
        if selected.input_indices[0] == selected.input_indices[1] {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let [Some(lhs), Some(rhs)] = owner_inputs else {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        };
        let (tensor0, retained_session) = lhs.into_device_parts();
        let (tensor1, other_session) = rhs.into_device_parts();
        if !Rc::ptr_eq(&retained_session, &other_session) || !Rc::ptr_eq(&retained_session, session)
        {
            return Err(PcuExecutionError::Argument(
                PcuArgumentError::SessionMismatch,
            ));
        }
        let ordered_inputs = [
            (
                selected.input_ids[0].ok_or(PcuExecutionError::InvalidTensorSourcePlan)?,
                tensor0,
            ),
            (
                selected.input_ids[1].ok_or(PcuExecutionError::InvalidTensorSourcePlan)?,
                tensor1,
            ),
        ];
        let output = assessor
            .execute_owned_program_consuming_binary_pair(prepared, ordered_inputs, pool, memory)
            .map_err(PcuExecutionError::TensorExecution)?;
        drop(other_session);
        Ok(PcuTensor::from_successful_output(output, retained_session))
    }

    fn execute_owner_with_borrowed_peer<T: PcuScalar>(
        assessor: &RocmTensorAssessor<'_>,
        context: SelectedOwnerExecution<'_>,
        owners: [Option<PcuTensor<T>>; 2],
        borrowed: [Option<PcuTensorInput<'_, T>>; 2],
        selected: ConsumedOwnersSelection,
    ) -> Result<PcuTensor<T>, PcuExecutionError> {
        let SelectedOwnerExecution {
            session,
            prepared,
            pool,
            memory,
            shape_keys,
            host_inputs,
        } = context;
        let owner_position = match (owners[0].is_some(), owners[1].is_some()) {
            (true, false) => 0,
            (false, true) => 1,
            _ => return Err(PcuExecutionError::InvalidTensorSourcePlan),
        };
        let peer_position = 1 - owner_position;
        if owners[peer_position].is_some() {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let [first_owner, second_owner] = owners;
        let owner = if owner_position == 0 {
            first_owner
        } else {
            second_owner
        }
        .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        let (tensor, retained_session) = owner.into_device_parts();
        if !Rc::ptr_eq(&retained_session, session) {
            return Err(PcuExecutionError::Argument(
                PcuArgumentError::SessionMismatch,
            ));
        }
        let peer_input =
            borrowed[peer_position].ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        let peer_index = selected.input_indices[peer_position];
        let peer_value =
            selected.input_ids[peer_position].ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        let donor_value =
            selected.input_ids[owner_position].ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        let peer_ref = match peer_input.kind {
            TensorInputKind::Host(_) => assessor.borrow_resource_input_ref(
                host_inputs
                    .get(peer_index)
                    .and_then(Option::as_ref)
                    .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?,
                &shape_keys
                    .get(peer_index)
                    .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?
                    .dimensions,
                T::TYPE,
                pool,
            ),
            TensorInputKind::Resident(owner) => {
                assessor.borrow_device_input_ref(owner.device_tensor(), pool)
            }
        }
        .map_err(PcuExecutionError::TensorExecution)?;
        let output = assessor
            .execute_owned_program_consuming_binary_input(
                prepared,
                donor_value,
                tensor,
                &[(peer_value, &peer_ref)],
                pool,
                memory,
            )
            .map_err(PcuExecutionError::TensorExecution)?;
        Ok(PcuTensor::from_successful_output(output, retained_session))
    }

    struct ConsumedPairSelection {
        slot: usize,
        donor_value: ValueId,
        other_value: Option<ValueId>,
    }

    enum ConsumedPairResolution<T: PcuScalar> {
        Pruned(PcuTensor<T>),
        Selected(ConsumedPairSelection),
    }

    #[derive(Clone, Copy)]
    struct EntryRequest<'a> {
        generation: u64,
        site: &'a PcuHostCallSite,
        specialization: TypeId,
    }

    fn resolve_consumed_pair<T: PcuScalar, F>(
        state: &mut State,
        request: EntryRequest<'_>,
        donor: &PcuTensor<T>,
        other: PcuTensorInput<'_, T>,
        donor_index: usize,
        build: F,
    ) -> Result<ConsumedPairResolution<T>, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue<T>; 2],
            ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
            + 'static,
    {
        let donor_source = <PcuTensor<T> as super::PcuTensorSource<
            T,
            super::DynamicResidentShape,
        >>::as_tensor_source(donor)
        .map_err(crate::global::argument_error)?;
        let inputs = if donor_index == 0 {
            [donor_source, other]
        } else {
            [other, donor_source]
        };
        let shapes = preflight_shapes(&inputs)?;
        let slot = resolve_entry_slot::<T, 2, F>(
            state,
            request.generation,
            request.site,
            request.specialization,
            shapes,
            &inputs,
            build,
        )?;
        let entry = &mut state.entries[slot];
        if !entry.input_indices.contains(&donor_index) {
            // A consumed but pruned parameter stays an ordinary Rust owner and is not selected
            // for session affinity or staging.
            return execute_entry(entry, &inputs).map(ConsumedPairResolution::Pruned);
        }

        let donor_position = entry
            .input_indices
            .iter()
            .position(|&index| index == donor_index)
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        let donor_value = *entry
            .input_ids
            .get(donor_position)
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        let other_index = 1 - donor_index;
        let other_value = if let Some(other_position) = entry
            .input_indices
            .iter()
            .position(|&index| index == other_index)
        {
            if let TensorInputKind::Host(values) = &inputs[other_index].kind {
                stage_host_input(entry, other_index, values)?;
            }
            Some(
                *entry
                    .input_ids
                    .get(other_position)
                    .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?,
            )
        } else {
            None
        };
        Ok(ConsumedPairResolution::Selected(ConsumedPairSelection {
            slot,
            donor_value,
            other_value,
        }))
    }

    fn resolve_entry_slot<T: PcuScalar, const N: usize, F>(
        state: &mut State,
        generation: u64,
        site: &PcuHostCallSite,
        specialization: TypeId,
        shapes: [PcuTensorShapeWitness<'_>; N],
        inputs: &[PcuTensorInput<'_, T>; N],
        build: F,
    ) -> Result<usize, PcuExecutionError>
    where
        F: FnOnce(
                &mut PcuTensorGraphCapture,
                [PcuTensorGraphValue<T>; N],
            ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
            + 'static,
    {
        let factory = TypeId::of::<F>();
        if state.generation != generation {
            state.entries.clear();
            state.generation = generation;
        }
        let matches_base = |entry: &Entry| {
            entry.specialization == specialization
                && entry.factory == factory
                && entry.scalar_type == T::TYPE
                && entry_matches_shapes(entry, &shapes)
        };
        let hint = site.slot.load(Ordering::Relaxed);
        let candidate = state
            .entries
            .get(hint)
            .filter(|entry| matches_base(entry))
            .map(|_| hint)
            .or_else(|| state.entries.iter().position(matches_base));
        if let Some(candidate) = candidate {
            let affinity =
                preflight_selected_inputs(inputs, &state.entries[candidate].input_indices)?;
            let matches_affinity = |entry: &Entry| {
                affinity.map_or(!entry.resident_affinity, |root| {
                    entry.resident_affinity && Rc::ptr_eq(root, &entry.session)
                })
            };
            if matches_affinity(&state.entries[candidate]) {
                site.slot.store(candidate, Ordering::Relaxed);
                return Ok(candidate);
            }
            if let Some(slot) = state
                .entries
                .iter()
                .position(|entry| matches_base(entry) && matches_affinity(entry))
            {
                site.slot.store(slot, Ordering::Relaxed);
                return Ok(slot);
            }
        }

        let selected_program = build_program::<T, N, F>(shapes, build)?;
        let affinity = preflight_selected_inputs(inputs, &selected_program.input_indices)?;
        let (entry, prepared_generation, capacity) =
            prepare_entry::<T, N>(affinity, specialization, factory, shapes, selected_program)?;
        if prepared_generation != generation {
            state.entries.clear();
            state.generation = prepared_generation;
        }
        let slot = if state.entries.len() == capacity {
            let victim = hint.min(state.entries.len() - 1);
            state.entries[victim] = entry;
            victim
        } else {
            state.entries.push(entry);
            state.entries.len() - 1
        };
        site.slot.store(slot, Ordering::Relaxed);
        Ok(slot)
    }

    pub(super) fn clear_cache() -> Result<(), PcuExecutionError> {
        with_state(|state| {
            state.entries.clear();
            Ok(())
        })
    }

    #[cfg(test)]
    mod cache_tests {
        use super::with_state;

        #[test]
        fn scalar_neutral_cache_state_keeps_generation() {
            with_state(|state| {
                state.generation = 0x32;
                state.entries.clear();
                Ok(())
            })
            .unwrap();

            with_state(|state| {
                assert_eq!(state.generation, 0x32);
                Ok(())
            })
            .unwrap();
        }

        #[test]
        fn generation_synchronization_invalidates_neutral_cache() {
            with_state(|state| {
                state.generation = u64::MAX;
                state.entries.clear();
                Ok(())
            })
            .unwrap();

            let expected = crate::global::hosted::current_generation();
            super::LAST_GENERATION.with(|last| last.set(expected.wrapping_sub(1)));
            super::synchronize_generation().unwrap();

            with_state(|state| {
                assert_eq!(state.generation, expected);
                Ok(())
            })
            .unwrap();
        }
    }
}

#[cfg(all(test, feature = "tensor"))]
mod capture_tests {
    #[rustfmt::skip]
    use super::{
        PcuExecutionError,
        PcuSourceShape,
        PcuTensorGraphCapture,
        PcuTensorGraphOwner,
    };
    use core::any::TypeId;

    #[test]
    fn graph_owner_borrow_and_consume_preserve_capture_provenance() {
        let shape = PcuSourceShape::Slice { length: 4 };
        let (capture, [value]) = PcuTensorGraphCapture::new::<f32, 1>([shape]).unwrap();
        let owner = PcuTensorGraphOwner::from_graph_value(value);
        let borrowed = owner.borrowed_graph_value();
        capture.require_rank(borrowed, 1).unwrap();
        let (other, _) = PcuTensorGraphCapture::new::<f32, 1>([shape]).unwrap();
        assert!(other.validate_value(borrowed).is_err());
        let moved = owner.into_graph_value();
        assert!(other.validate_value(moved).is_err());
        let (graph, output) = capture.finish(moved).unwrap();
        assert_eq!(graph.shape(output).unwrap(), [4]);
    }

    #[test]
    fn graph_values_cannot_cross_capture_boundaries() {
        let shape = PcuSourceShape::Slice { length: 4 };
        let (mut first, [value]) = PcuTensorGraphCapture::new::<f32, 1>([shape]).unwrap();
        let (second, _) = PcuTensorGraphCapture::new::<f32, 1>([shape]).unwrap();
        let foreign = first.identity(value).unwrap();
        assert!(second.validate_value(foreign).is_err());
        first.finish(value).unwrap();
    }

    #[test]
    fn one_capture_can_register_independent_f32_and_f64_inputs() {
        let slice = PcuSourceShape::Slice { length: 2 };
        let (mut capture, [f32_input]) = PcuTensorGraphCapture::new::<f32, 1>([slice]).unwrap();
        let f64_input = capture.input::<f64>(slice).unwrap();
        let f64_output = capture.relu(f64_input).unwrap();
        capture.require_shape(f64_output, slice).unwrap();

        let (graph, output) = capture.finish(f64_output).unwrap();
        assert_eq!(
            graph.node(f32_input.value.erase()).unwrap().scalar_type,
            crate::core::PcuScalarType::F32
        );
        assert_eq!(
            graph.node(f64_input.value.erase()).unwrap().scalar_type,
            crate::core::PcuScalarType::F64
        );
        assert_eq!(
            graph.node(output).unwrap().scalar_type,
            crate::core::PcuScalarType::F64
        );
    }

    #[test]
    fn cross_capture_validation_rejects_foreign_f64_tokens() {
        let shape = PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 3,
        };
        let (mut first, [foreign]) = PcuTensorGraphCapture::new::<f64, 1>([shape]).unwrap();
        let (second, _) = PcuTensorGraphCapture::new::<f64, 0>([]).unwrap();
        let foreign = first.identity(foreign).unwrap();
        assert!(second.validate_value(foreign).is_err());
        first.finish(foreign).unwrap();
    }

    #[test]
    fn recursion_and_nesting_guards_recover_after_leave() {
        let (mut capture, [value]) =
            PcuTensorGraphCapture::new::<f32, 1>([PcuSourceShape::Slice { length: 1 }]).unwrap();
        let marker = TypeId::of::<u8>();
        capture.enter(marker).unwrap();
        assert!(matches!(
            capture.enter(marker),
            Err(PcuExecutionError::RecursiveTensorSource)
        ));
        capture.leave();
        capture.enter(marker).unwrap();
        capture.leave();
        capture.active_markers.resize(64, marker);
        assert!(matches!(
            capture.enter(TypeId::of::<u16>()),
            Err(PcuExecutionError::TensorSourceNestingLimit)
        ));
        capture.active_markers.clear();
        capture.finish(value).unwrap();
    }

    #[test]
    fn graph_finish_rejects_unbalanced_companion_scope() {
        let (mut capture, [value]) =
            PcuTensorGraphCapture::new::<f32, 1>([PcuSourceShape::Slice { length: 2 }]).unwrap();
        capture.enter(TypeId::of::<u8>()).unwrap();
        assert!(matches!(
            capture.finish(value),
            Err(PcuExecutionError::InvalidTensorSourcePlan)
        ));
    }

    #[test]
    fn addition_checks_capture_provenance_and_shape() {
        let slice = PcuSourceShape::Slice { length: 4 };
        let (mut capture, [lhs, rhs]) =
            PcuTensorGraphCapture::new::<f32, 2>([slice, slice]).unwrap();
        let output = capture.add(lhs, rhs).unwrap();
        let (graph, output_id) = capture.finish(output).unwrap();
        assert_eq!(graph.shape(output_id).unwrap(), [4]);

        let (mut mismatched, [lhs, rhs]) = PcuTensorGraphCapture::new::<f32, 2>([
            PcuSourceShape::Slice { length: 4 },
            PcuSourceShape::Slice { length: 5 },
        ])
        .unwrap();
        assert!(matches!(
            mismatched.add(lhs, rhs),
            Err(PcuExecutionError::TensorBuild(
                crate::dialect::tensor::TensorError::ShapeMismatch { left, right }
            )) if left == [4] && right == [5]
        ));
        let (_other, [foreign]) = PcuTensorGraphCapture::new::<f32, 1>([slice]).unwrap();
        assert!(mismatched.add(lhs, foreign).is_err());
    }

    #[test]
    fn subtraction_and_multiplication_preserve_tensor_shape() {
        let slice = PcuSourceShape::Slice { length: 3 };
        let (mut capture, [lhs, rhs]) =
            PcuTensorGraphCapture::new::<f32, 2>([slice, slice]).unwrap();
        let difference = capture.sub(lhs, rhs).unwrap();
        let product = capture.mul(difference, rhs).unwrap();
        let (graph, output) = capture.finish(product).unwrap();
        assert_eq!(graph.shape(output).unwrap(), [3]);
    }

    #[test]
    fn matrix_capture_preserves_extents_and_validates_helper_shapes() {
        let lhs_shape = PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 3,
        };
        let rhs_shape = PcuSourceShape::FixedMatrix {
            rows: 3,
            columns: 2,
        };
        let (mut capture, [lhs, rhs]) =
            PcuTensorGraphCapture::new::<f32, 2>([lhs_shape, rhs_shape]).unwrap();
        capture.require_shape(lhs, lhs_shape).unwrap();
        capture.require_rank(lhs, 2).unwrap();
        assert!(matches!(
            capture.require_rank(lhs, 1),
            Err(PcuExecutionError::TensorSourceRankMismatch {
                expected: 1,
                actual: 2,
            })
        ));
        assert!(matches!(
            capture.require_shape(
                lhs,
                PcuSourceShape::FixedMatrix {
                    rows: 3,
                    columns: 2,
                }
            ),
            Err(PcuExecutionError::TensorSourceShapeMismatch {
                expected: PcuSourceShape::FixedMatrix {
                    rows: 3,
                    columns: 2,
                },
                actual,
            }) if actual == [2, 3]
        ));
        let output = capture.matmul(lhs, rhs).unwrap();
        capture
            .require_shape(
                output,
                PcuSourceShape::FixedMatrix {
                    rows: 2,
                    columns: 2,
                },
            )
            .unwrap();
        let (graph, output_id) = capture.finish(output).unwrap();
        assert_eq!(graph.shape(output_id).unwrap(), [2, 2]);
    }
}

#[cfg(all(feature = "rocm", feature = "tensor"))]
pub(super) fn clear_cache() -> Result<(), PcuExecutionError> {
    execution::clear_cache()
}
