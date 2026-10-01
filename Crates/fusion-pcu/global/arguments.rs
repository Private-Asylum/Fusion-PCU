//! Typed host/resident source carriers used by generated same-name direct entries.
//!
//! Host references stage through the selected execution provider, while resident owners retain
//! their session and storage across calls. No public raw resident constructor or fake host slice
//! view is exposed; mutable resident borrows track submission and completion outcomes.

#[rustfmt::skip]
use crate::{
    PcuBindingRef,
    PcuHostArgument,
    PcuScalar,
};
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
#[rustfmt::skip]
use super::resident::{
    DeviceArgument,
    DeviceTensor,
    Session,
};
use core::marker::PhantomData;

/// Semantic source shape, retaining rank and source role instead of only a flattened element
/// count. The descriptor is stack-only and fixed arrays need no heap-backed shape vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcuSourceShape {
    Scalar,
    Slice { length: usize },
    FixedArray { length: usize },
    FixedMatrix { rows: usize, columns: usize },
}

impl PcuSourceShape {
    /// Whether `actual` is a valid resident shape for this source-level parameter.
    #[must_use]
    pub fn accepts_resident_shape(self, actual: &[usize]) -> bool {
        match self {
            Self::Scalar => actual.is_empty(),
            Self::Slice { length } | Self::FixedArray { length } => actual == [length],
            Self::FixedMatrix { rows, columns } => actual == [rows, columns],
        }
    }

    /// Slice parameters accept fixed arrays of the same rank-one extent through normal Rust
    /// coercion. Every other source-role pairing stays explicit.
    #[must_use]
    pub const fn accepts_source_role(self, actual: Self) -> bool {
        match (self, actual) {
            (
                Self::Slice { length: expected },
                Self::Slice { length: found } | Self::FixedArray { length: found },
            )
            | (Self::FixedArray { length: expected }, Self::FixedArray { length: found }) => {
                expected == found
            }
            (
                Self::FixedMatrix {
                    rows: expected_rows,
                    columns: expected_columns,
                },
                Self::FixedMatrix {
                    rows: actual_rows,
                    columns: actual_columns,
                },
            ) => expected_rows == actual_rows && expected_columns == actual_columns,
            (Self::Scalar, Self::Scalar) => true,
            _ => false,
        }
    }
}

/// Argument construction can fail before any operation is submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcuArgumentError {
    SourceShapeMismatch {
        expected: PcuSourceShape,
        actual: PcuSourceShape,
    },
    ResidentShapeMismatch {
        expected: PcuSourceShape,
    },
    SessionMismatch,
    ResidentCompletionUncertain,
    /// A logical resident tensor is unavailable when the facade has no device provider.
    ProviderUnavailable,
}

/// Opaque per-argument carrier passed from generated source signatures to the hosted dispatcher.
#[doc(hidden)]
#[cfg_attr(
    not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu"
    )),
    allow(dead_code)
)] // The disabled facade never inspects mixed arguments.
pub struct PcuCallArgument<'a> {
    shape: PcuSourceShape,
    kind: PcuCallArgumentKind<'a>,
}

#[doc(hidden)]
#[cfg_attr(
    not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu"
    )),
    allow(dead_code)
)] // Resident/host discrimination runs only with a provider.
pub(super) enum PcuCallArgumentKind<'a> {
    Host(PcuHostArgument<'a>),
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    ResidentRead(PcuResidentReadArgument<'a>),
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    ResidentWrite(PcuResidentWriteArgument<'a>),
}

#[cfg_attr(
    not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu"
    )),
    allow(dead_code)
)] // Used by the provider-backed stack splitter.
impl<'a> PcuCallArgument<'a> {
    #[allow(clippy::missing_const_for_fn)] // Resident variants own drop guards and backend borrows.
    pub(super) fn into_parts(self) -> (PcuSourceShape, PcuCallArgumentKind<'a>) {
        (self.shape, self.kind)
    }

    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    pub(super) const fn kind(&self) -> &PcuCallArgumentKind<'a> {
        &self.kind
    }
}

impl<'a> PcuCallArgument<'a> {
    const fn host(argument: PcuHostArgument<'a>, shape: PcuSourceShape) -> Self {
        Self {
            shape,
            kind: PcuCallArgumentKind::Host(argument),
        }
    }
}

#[allow(clippy::unnecessary_wraps)] // Host and resident conversions share one fallible source contract.
const fn host_call_argument(
    argument: PcuHostArgument<'_>,
    shape: PcuSourceShape,
) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
    Ok(PcuCallArgument::host(argument, shape))
}

mod sealed {
    pub trait Sealed<T, Shape> {}
    pub trait DeviceBufferOwner {}
}

pub struct ScalarShape;
pub struct SliceShape;
pub struct FixedArrayShape<const N: usize>;
pub struct FixedMatrixShape<const R: usize, const C: usize>;

/// Sealed read-only storage accepted by generated direct source entries.
#[doc(hidden)]
pub trait PcuReadStorage<T: PcuScalar, Shape>: sealed::Sealed<T, Shape> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError>;
}

/// Sealed exclusively borrowed storage accepted by generated destination arguments.
#[doc(hidden)]
pub trait PcuWriteStorage<T: PcuScalar, Shape>: sealed::Sealed<T, Shape> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError>;
}

impl<T: PcuScalar> sealed::Sealed<T, ScalarShape> for T {}
impl<T: PcuScalar> PcuReadStorage<T, ScalarShape> for T {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read_scalar(target, self),
            PcuSourceShape::Scalar,
        )
    }
}
impl<T: PcuScalar> PcuWriteStorage<T, ScalarShape> for T {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read_write_scalar(target, self),
            PcuSourceShape::Scalar,
        )
    }
}

impl<T: PcuScalar> sealed::Sealed<T, ScalarShape> for alloc::boxed::Box<T> {}
impl<T: PcuScalar> PcuReadStorage<T, ScalarShape> for alloc::boxed::Box<T> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read_scalar(target, self.as_ref()),
            PcuSourceShape::Scalar,
        )
    }
}
impl<T: PcuScalar> PcuWriteStorage<T, ScalarShape> for alloc::boxed::Box<T> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read_write_scalar(target, self.as_mut()),
            PcuSourceShape::Scalar,
        )
    }
}

impl<T: PcuScalar> sealed::Sealed<T, SliceShape> for [T] {}
impl<T: PcuScalar> PcuReadStorage<T, SliceShape> for [T] {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read(target, self),
            PcuSourceShape::Slice { length: self.len() },
        )
    }
}
impl<T: PcuScalar> PcuWriteStorage<T, SliceShape> for [T] {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        let length = self.len();
        host_call_argument(
            PcuHostArgument::read_write(target, self),
            PcuSourceShape::Slice { length },
        )
    }
}

impl<T: PcuScalar, const N: usize> sealed::Sealed<T, SliceShape> for [T; N] {}
impl<T: PcuScalar, const N: usize> PcuReadStorage<T, SliceShape> for [T; N] {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read(target, self.as_slice()),
            PcuSourceShape::FixedArray { length: N },
        )
    }
}
impl<T: PcuScalar, const N: usize> PcuWriteStorage<T, SliceShape> for [T; N] {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read_write(target, self.as_mut_slice()),
            PcuSourceShape::FixedArray { length: N },
        )
    }
}
impl<T: PcuScalar, const N: usize> sealed::Sealed<T, FixedArrayShape<N>> for [T; N] {}
impl<T: PcuScalar, const N: usize> PcuReadStorage<T, FixedArrayShape<N>> for [T; N] {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read(target, self.as_slice()),
            PcuSourceShape::FixedArray { length: N },
        )
    }
}
impl<T: PcuScalar, const N: usize> PcuWriteStorage<T, FixedArrayShape<N>> for [T; N] {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read_write(target, self.as_mut_slice()),
            PcuSourceShape::FixedArray { length: N },
        )
    }
}

impl<T: PcuScalar, const R: usize, const C: usize> sealed::Sealed<T, FixedMatrixShape<R, C>>
    for [[T; C]; R]
{
}
impl<T: PcuScalar, const R: usize, const C: usize> PcuReadStorage<T, FixedMatrixShape<R, C>>
    for [[T; C]; R]
{
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read(target, self.as_flattened()),
            PcuSourceShape::FixedMatrix {
                rows: R,
                columns: C,
            },
        )
    }
}
impl<T: PcuScalar, const R: usize, const C: usize> PcuWriteStorage<T, FixedMatrixShape<R, C>>
    for [[T; C]; R]
{
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read_write(target, self.as_flattened_mut()),
            PcuSourceShape::FixedMatrix {
                rows: R,
                columns: C,
            },
        )
    }
}

impl<T: PcuScalar, const R: usize, const C: usize> sealed::Sealed<T, FixedMatrixShape<R, C>>
    for alloc::boxed::Box<[[T; C]; R]>
{
}
impl<T: PcuScalar, const R: usize, const C: usize> PcuReadStorage<T, FixedMatrixShape<R, C>>
    for alloc::boxed::Box<[[T; C]; R]>
{
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read(target, self.as_ref().as_flattened()),
            PcuSourceShape::FixedMatrix {
                rows: R,
                columns: C,
            },
        )
    }
}
impl<T: PcuScalar, const R: usize, const C: usize> PcuWriteStorage<T, FixedMatrixShape<R, C>>
    for alloc::boxed::Box<[[T; C]; R]>
{
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        let values = self.as_mut().as_flattened_mut();
        host_call_argument(
            PcuHostArgument::read_write(target, values),
            PcuSourceShape::FixedMatrix {
                rows: R,
                columns: C,
            },
        )
    }
}

impl<T: PcuScalar> sealed::Sealed<T, SliceShape> for alloc::vec::Vec<T> {}
impl<T: PcuScalar> PcuReadStorage<T, SliceShape> for alloc::vec::Vec<T> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read(target, self.as_slice()),
            PcuSourceShape::Slice { length: self.len() },
        )
    }
}
impl<T: PcuScalar> PcuWriteStorage<T, SliceShape> for alloc::vec::Vec<T> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        let values = self.as_mut_slice();
        let length = values.len();
        host_call_argument(
            PcuHostArgument::read_write(target, values),
            PcuSourceShape::Slice { length },
        )
    }
}

impl<T: PcuScalar> sealed::Sealed<T, SliceShape> for alloc::boxed::Box<[T]> {}
impl<T: PcuScalar> PcuReadStorage<T, SliceShape> for alloc::boxed::Box<[T]> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read(target, self),
            PcuSourceShape::Slice { length: self.len() },
        )
    }
}
impl<T: PcuScalar> PcuWriteStorage<T, SliceShape> for alloc::boxed::Box<[T]> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        let values: &mut [T] = self;
        let length = values.len();
        host_call_argument(
            PcuHostArgument::read_write(target, values),
            PcuSourceShape::Slice { length },
        )
    }
}
impl<T: PcuScalar, const N: usize> sealed::Sealed<T, SliceShape> for alloc::boxed::Box<[T; N]> {}
impl<T: PcuScalar, const N: usize> PcuReadStorage<T, SliceShape> for alloc::boxed::Box<[T; N]> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read(target, self.as_ref().as_slice()),
            PcuSourceShape::FixedArray { length: N },
        )
    }
}
impl<T: PcuScalar, const N: usize> PcuWriteStorage<T, SliceShape> for alloc::boxed::Box<[T; N]> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        let values = self.as_mut().as_mut_slice();
        host_call_argument(
            PcuHostArgument::read_write(target, values),
            PcuSourceShape::FixedArray { length: N },
        )
    }
}
impl<T: PcuScalar, const N: usize> sealed::Sealed<T, FixedArrayShape<N>>
    for alloc::boxed::Box<[T; N]>
{
}
impl<T: PcuScalar, const N: usize> PcuReadStorage<T, FixedArrayShape<N>>
    for alloc::boxed::Box<[T; N]>
{
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        host_call_argument(
            PcuHostArgument::read(target, self.as_ref().as_slice()),
            PcuSourceShape::FixedArray { length: N },
        )
    }
}
impl<T: PcuScalar, const N: usize> PcuWriteStorage<T, FixedArrayShape<N>>
    for alloc::boxed::Box<[T; N]>
{
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        let values = self.as_mut().as_mut_slice();
        host_call_argument(
            PcuHostArgument::read_write(target, values),
            PcuSourceShape::FixedArray { length: N },
        )
    }
}

macro_rules! shared_slice_storage {
    ($($container:ident)::+) => {
        impl<T: PcuScalar> sealed::Sealed<T, SliceShape> for $($container)::+<[T]> {}
        impl<T: PcuScalar> PcuReadStorage<T, SliceShape> for $($container)::+<[T]> {
            fn as_pcu_call_argument(&self, target: PcuBindingRef) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
                let values: &[T] = &**self;
                host_call_argument(PcuHostArgument::read(target, values), PcuSourceShape::Slice { length: values.len() })
            }
        }
    };
}
shared_slice_storage!(alloc::rc::Rc);
shared_slice_storage!(alloc::sync::Arc);

macro_rules! shared_scalar_storage {
    ($($container:ident)::+) => {
        impl<T: PcuScalar> sealed::Sealed<T, ScalarShape> for $($container)::+<T> {}
        impl<T: PcuScalar> PcuReadStorage<T, ScalarShape> for $($container)::+<T> {
            fn as_pcu_call_argument(&self, target: PcuBindingRef) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
                host_call_argument(PcuHostArgument::read_scalar(target, self.as_ref()), PcuSourceShape::Scalar)
            }
        }
    };
}
shared_scalar_storage!(alloc::rc::Rc);
shared_scalar_storage!(alloc::sync::Arc);

macro_rules! shared_matrix_storage {
    ($($container:ident)::+) => {
        impl<T: PcuScalar, const R: usize, const C: usize> sealed::Sealed<T, FixedMatrixShape<R, C>>
            for $($container)::+<[[T; C]; R]> {}
        impl<T: PcuScalar, const R: usize, const C: usize> PcuReadStorage<T, FixedMatrixShape<R, C>>
            for $($container)::+<[[T; C]; R]> {
            fn as_pcu_call_argument(&self, target: PcuBindingRef) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
                host_call_argument(
                    PcuHostArgument::read(target, self.as_ref().as_flattened()),
                    PcuSourceShape::FixedMatrix { rows: R, columns: C },
                )
            }
        }
    };
}
shared_matrix_storage!(alloc::rc::Rc);
shared_matrix_storage!(alloc::sync::Arc);

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidentValidity {
    Ready,
    Uncertain,
}

/// A logical tensor whose backing may be retained by an execution provider.
///
/// Consumers cannot construct this owner directly. It is returned by successful owned-result
/// kernels and keeps its provider resources alive independently of the thread cache.
/// Borrowing retains the logical value; moving transfers it. A move permits a provider to
/// consider storage reuse, but does not establish physical backing exclusivity by itself.
/// Ordinary `Drop` releases this owner's claim; readback does not consume it.
///
/// A live borrow prevents transferring the owner, including while that borrow is used by a
/// later device operation:
///
/// ```compile_fail,E0505
/// use fusion_pcu::PcuTensor;
/// fn transfer(value: PcuTensor<f32>) -> PcuTensor<f32> { value }
/// fn invalid(value: PcuTensor<f32>) {
///     let borrowed = &value;
///     let moved = transfer(value);
///     let _ = borrowed.len();
///     drop(moved);
/// }
/// ```
pub struct PcuTensor<T: PcuScalar> {
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    tensor: DeviceTensor<T>,
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    session: std::rc::Rc<Session>,
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    validity: ResidentValidity,
    marker: PhantomData<fn() -> T>,
}

impl<T: PcuScalar> PcuTensor<T> {
    /// Consumes explicitly initialized device storage into a logical resident owner.
    ///
    /// This advanced ownership seam performs no transfers or numerical work. The statically
    /// selected backend verifies physical session, scalar type, actual extent, completion and
    /// shape. Provider availability and tensor arithmetic remain separate admission requirements.
    ///
    /// Each current Metal import creates a distinct facade affinity root. A single imported owner
    /// composes with RAM arguments; separately imported owners cannot share one source call, even
    /// when their storage originated on the same physical device. A shared import-session seam
    /// is required before multiple imported owners can compose; physical identity checks remain
    /// stricter than device ordinals alone.
    ///
    /// # Errors
    /// Rejects incompatible storage or shape without submitting device work.
    pub fn from_device_buffer<B: PcuResidentBufferOwner<T>>(
        backend: B,
        buffer: crate::PcuDeviceBuffer<T, B::Resource>,
        dimensions: &[usize],
    ) -> Result<Self, super::PcuExecutionError> {
        backend.into_resident_owner(buffer, dimensions)
    }
    #[cfg(any(
        feature = "metal",
        all(any(feature = "rocm", feature = "cuda"), feature = "tensor")
    ))]
    pub(super) fn from_successful_output(
        tensor: DeviceTensor<T>,
        session: std::rc::Rc<Session>,
    ) -> Self {
        Self {
            tensor,
            session,
            validity: ResidentValidity::Ready,
            marker: PhantomData,
        }
    }

    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    pub(super) const fn device_tensor(&self) -> &DeviceTensor<T> {
        &self.tensor
    }

    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    pub(super) const fn session(&self) -> &std::rc::Rc<Session> {
        &self.session
    }

    #[cfg(all(any(feature = "rocm", feature = "cuda"), feature = "tensor"))]
    pub(super) fn into_device_parts(self) -> (DeviceTensor<T>, std::rc::Rc<Session>) {
        (self.tensor, self.session)
    }

    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    pub(super) const fn validate_initialized(&self) -> Result<(), PcuArgumentError> {
        match self.validity {
            ResidentValidity::Ready => Ok(()),
            ResidentValidity::Uncertain => Err(PcuArgumentError::ResidentCompletionUncertain),
        }
    }

    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    pub(super) fn validate_read(
        &self,
        expected_shape: PcuSourceShape,
    ) -> Result<(), PcuArgumentError> {
        if !expected_shape.accepts_resident_shape(self.tensor.shape()) {
            return Err(PcuArgumentError::ResidentShapeMismatch {
                expected: expected_shape,
            });
        }
        self.validate_initialized()?;
        self.tensor.validate_access_available()
    }

    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    fn read_argument(
        &self,
        target: PcuBindingRef,
        expected_shape: PcuSourceShape,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.validate_read(expected_shape)?;
        Ok(PcuResidentReadArgument {
            argument: self.tensor.read_argument(target),
            shape: expected_shape,
            session: &self.session,
        }
        .into_call_argument())
    }

    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    fn write_argument(
        &mut self,
        target: PcuBindingRef,
        expected_shape: PcuSourceShape,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.validate_read(expected_shape)?;
        let Self {
            tensor,
            session,
            validity,
            ..
        } = self;
        let guard = ResidentWriteGuard::new(validity);
        Ok(PcuResidentWriteArgument {
            argument: tensor.read_write_argument(target),
            shape: expected_shape,
            session,
            guard,
        }
        .into_call_argument())
    }
}

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<T: PcuScalar, Shape> sealed::Sealed<T, Shape> for PcuTensor<T> {}
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<T: PcuScalar> PcuReadStorage<T, ScalarShape> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.read_argument(target, PcuSourceShape::Scalar)
    }
}
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<T: PcuScalar> PcuReadStorage<T, SliceShape> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        let length = self.tensor.len();
        self.read_argument(target, PcuSourceShape::Slice { length })
    }
}
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<T: PcuScalar, const N: usize> PcuReadStorage<T, FixedArrayShape<N>> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.read_argument(target, PcuSourceShape::FixedArray { length: N })
    }
}
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<T: PcuScalar, const R: usize, const C: usize> PcuReadStorage<T, FixedMatrixShape<R, C>>
    for PcuTensor<T>
{
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.read_argument(
            target,
            PcuSourceShape::FixedMatrix {
                rows: R,
                columns: C,
            },
        )
    }
}
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<T: PcuScalar> PcuWriteStorage<T, ScalarShape> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.write_argument(target, PcuSourceShape::Scalar)
    }
}
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<T: PcuScalar> PcuWriteStorage<T, SliceShape> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        let length = self.tensor.len();
        self.write_argument(target, PcuSourceShape::Slice { length })
    }
}
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<T: PcuScalar, const N: usize> PcuWriteStorage<T, FixedArrayShape<N>> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.write_argument(target, PcuSourceShape::FixedArray { length: N })
    }
}
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<T: PcuScalar, const R: usize, const C: usize> PcuWriteStorage<T, FixedMatrixShape<R, C>>
    for PcuTensor<T>
{
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.write_argument(
            target,
            PcuSourceShape::FixedMatrix {
                rows: R,
                columns: C,
            },
        )
    }
}

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
pub(super) struct PcuResidentReadArgument<'a> {
    pub(super) argument: DeviceArgument<'a>,
    pub(super) shape: PcuSourceShape,
    pub(super) session: &'a std::rc::Rc<Session>,
}

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<'a> PcuResidentReadArgument<'a> {
    const fn into_call_argument(self) -> PcuCallArgument<'a> {
        PcuCallArgument {
            shape: self.shape,
            kind: PcuCallArgumentKind::ResidentRead(self),
        }
    }
}

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
pub(super) struct PcuResidentWriteArgument<'a> {
    pub(super) argument: DeviceArgument<'a>,
    pub(super) shape: PcuSourceShape,
    pub(super) session: &'a std::rc::Rc<Session>,
    pub(super) guard: ResidentWriteGuard<'a>,
}

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<'a> PcuResidentWriteArgument<'a> {
    const fn into_call_argument(self) -> PcuCallArgument<'a> {
        PcuCallArgument {
            shape: self.shape,
            kind: PcuCallArgumentKind::ResidentWrite(self),
        }
    }
}

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidentWriteDisposition {
    NotSubmitted,
    MayHaveWritten,
    KnownPartial,
    Complete,
}

/// Tracks whether a mutable resident value remains readable across failure. The caller marks
/// possible submission only after preflight; Drop restores the prior state for prelaunch errors,
/// preserves initialized storage after a quiescent partial error, and blocks uncertain completion.
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
pub(super) struct ResidentWriteGuard<'a> {
    validity: &'a mut ResidentValidity,
    prior: ResidentValidity,
    disposition: ResidentWriteDisposition,
}

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl<'a> ResidentWriteGuard<'a> {
    const fn new(validity: &'a mut ResidentValidity) -> Self {
        let prior = *validity;
        Self {
            validity,
            prior,
            disposition: ResidentWriteDisposition::NotSubmitted,
        }
    }

    pub(super) const fn mark_may_have_written(&mut self) {
        self.disposition = ResidentWriteDisposition::MayHaveWritten;
    }

    pub(super) const fn mark_known_partial(&mut self) {
        self.disposition = ResidentWriteDisposition::KnownPartial;
    }

    pub(super) const fn mark_complete(&mut self) {
        self.disposition = ResidentWriteDisposition::Complete;
    }
}

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
impl Drop for ResidentWriteGuard<'_> {
    fn drop(&mut self) {
        *self.validity = match self.disposition {
            ResidentWriteDisposition::NotSubmitted => self.prior,
            ResidentWriteDisposition::MayHaveWritten => ResidentValidity::Uncertain,
            ResidentWriteDisposition::KnownPartial => match self.prior {
                // A quiescent operation error may leave a changed prefix, but every element of a
                // previously initialized scalar buffer still has a valid Rust representation.
                ResidentValidity::Ready => ResidentValidity::Ready,
                ResidentValidity::Uncertain => ResidentValidity::Uncertain,
            },
            ResidentWriteDisposition::Complete => ResidentValidity::Ready,
        };
    }
}

/// Sealed static ownership adapter for advanced device-buffer interoperation.
///
/// Implementations must prove initialized payload independently of the generic typed allocation
/// contract before marking the returned logical owner Ready.
#[doc(hidden)]
pub trait PcuResidentBufferOwner<T: PcuScalar>: sealed::DeviceBufferOwner {
    type Resource;
    /// # Errors
    /// Rejects inconsistent device, scalar, extent, completion or shape contracts.
    fn into_resident_owner(
        self,
        buffer: crate::PcuDeviceBuffer<T, Self::Resource>,
        dimensions: &[usize],
    ) -> Result<PcuTensor<T>, super::PcuExecutionError>;
}
#[cfg(feature = "metal")]
impl sealed::DeviceBufferOwner for fusion_pcu_metal::MetalOwnedDispatchBackend {}
#[cfg(feature = "metal")]
impl<T: PcuScalar> PcuResidentBufferOwner<T> for fusion_pcu_metal::MetalOwnedDispatchBackend {
    type Resource = fusion_pcu_metal::MetalMemoryResource;
    fn into_resident_owner(
        self,
        buffer: crate::PcuDeviceBuffer<T, Self::Resource>,
        dimensions: &[usize],
    ) -> Result<PcuTensor<T>, super::PcuExecutionError> {
        super::resident::from_metal_buffer(self, buffer, dimensions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_shapes_preserve_rank_and_parameter_role() {
        let fixed_matrix = PcuSourceShape::FixedMatrix {
            rows: 2,
            columns: 3,
        };
        assert!(fixed_matrix.accepts_resident_shape(&[2, 3]));
        assert!(!fixed_matrix.accepts_resident_shape(&[3, 2]));
        assert!(!fixed_matrix.accepts_resident_shape(&[6]));
        assert!(
            PcuSourceShape::Slice { length: 6 }
                .accepts_source_role(PcuSourceShape::FixedArray { length: 6 })
        );
        assert!(
            !PcuSourceShape::FixedArray { length: 6 }
                .accepts_source_role(PcuSourceShape::Slice { length: 6 })
        );
    }

    #[test]
    fn host_carriers_retain_scalar_and_mutable_access_contracts() {
        let scalar = 4.0_f32;
        let argument = PcuReadStorage::<f32, ScalarShape>::as_pcu_call_argument(
            &scalar,
            PcuBindingRef::new(0, 0),
        )
        .expect("host scalar conversion succeeds");
        assert_eq!(argument.shape, PcuSourceShape::Scalar);
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        let PcuCallArgumentKind::Host(host_argument) = argument.kind else {
            panic!("host scalar conversion remains host-backed");
        };
        #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "metal")))]
        let PcuCallArgumentKind::Host(host_argument) = argument.kind;
        assert_eq!(host_argument.access(), crate::PcuBindingAccess::ReadOnly);

        let mut matrix = [[1_u32, 2, 3], [4, 5, 6]];
        let argument = PcuWriteStorage::<u32, FixedMatrixShape<2, 3>>::as_pcu_call_argument(
            &mut matrix,
            PcuBindingRef::new(0, 1),
        )
        .expect("host matrix conversion succeeds");
        assert_eq!(
            argument.shape,
            PcuSourceShape::FixedMatrix {
                rows: 2,
                columns: 3
            }
        );
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        let PcuCallArgumentKind::Host(host_argument) = argument.kind else {
            panic!("host matrix conversion remains host-backed");
        };
        #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "metal")))]
        let PcuCallArgumentKind::Host(host_argument) = argument.kind;
        assert_eq!(host_argument.access(), crate::PcuBindingAccess::ReadWrite);
        assert_eq!(host_argument.bytes().len(), 6 * core::mem::size_of::<u32>());
    }

    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    #[test]
    fn resident_write_guard_preserves_initialized_values_but_blocks_uncertain_completion() {
        let mut validity = ResidentValidity::Ready;
        {
            let _guard = ResidentWriteGuard::new(&mut validity);
        }
        assert_eq!(validity, ResidentValidity::Ready);
        {
            let mut guard = ResidentWriteGuard::new(&mut validity);
            guard.mark_known_partial();
        }
        assert_eq!(validity, ResidentValidity::Ready);
        {
            let mut guard = ResidentWriteGuard::new(&mut validity);
            guard.mark_may_have_written();
        }
        assert_eq!(validity, ResidentValidity::Uncertain);
        {
            let mut guard = ResidentWriteGuard::new(&mut validity);
            guard.mark_complete();
        }
        assert_eq!(validity, ResidentValidity::Ready);
    }
}
