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
#[cfg(any(
    feature = "mlx",
    not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        all(feature = "cpu", feature = "tensor"),
        all(feature = "vulkan", feature = "tensor")
    ))
))]
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
    /// A completed unsuccessful operation discarded this logical resident value.
    /// Its backing remains owned until drop, but it cannot be read or borrowed again.
    ResidentValueDiscarded,
    /// A logical resident tensor is unavailable when the facade has no device provider.
    ProviderUnavailable,
    /// This owner has no supported generic device-buffer invocation view.
    UnsupportedResidentBorrow,
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
    #[cfg(all(feature = "vulkan", feature = "tensor"))]
    VulkanRead(PcuVulkanReadArgument<'a>),
    #[cfg(all(feature = "vulkan", feature = "tensor"))]
    VulkanWrite(PcuVulkanWriteArgument<'a>),
    #[cfg(all(feature = "cpu", feature = "tensor"))]
    CpuOwner(PcuHostArgument<'a>),
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    ResidentRead(PcuResidentReadArgument<'a>),
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    ResidentWrite(PcuResidentWriteArgument<'a>),
    #[cfg(feature = "mlx")]
    MlxRead(PcuMlxReadArgument<'a>),
    #[cfg(feature = "mlx")]
    MlxWrite(PcuMlxWriteArgument<'a>),
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

    #[cfg(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        all(feature = "cpu", feature = "tensor"),
        all(feature = "vulkan", feature = "tensor")
    ))]
    pub(super) const fn kind(&self) -> &PcuCallArgumentKind<'a> {
        &self.kind
    }
}

impl<'a> PcuCallArgument<'a> {
    /// Metadata for an unread source declaration, established by validated lowering.
    ///
    /// No consumer storage is inspected or borrowed. The empty typed declaration
    /// retains access/type validation but contributes no residency affinity, upload,
    /// owner lease or device allocation. Providers must still assess the actual IR;
    /// this is not an initialized device buffer or permission to skip a real load.
    #[doc(hidden)]
    #[must_use]
    pub const fn unused_read<T: PcuScalar>(target: PcuBindingRef) -> Self {
        Self::host(
            PcuHostArgument::read::<T>(target, &[]),
            PcuSourceShape::Slice { length: 0 },
        )
    }

    /// Metadata for a mutable declaration with no reads or writes in validated lowering.
    ///
    /// The generated function still takes an exclusive Rust borrow. This empty
    /// read/write declaration preserves access and type checks, but never borrows
    /// consumer backing, inspects its session or contributes residency affinity.
    /// Providers must reject any actual IR access that this empty view cannot cover.
    #[doc(hidden)]
    #[must_use]
    pub const fn unused_read_write<T: PcuScalar>(target: PcuBindingRef) -> Self {
        Self::host(
            PcuHostArgument::read_write::<T>(target, &mut []),
            PcuSourceShape::Slice { length: 0 },
        )
    }

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

#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    all(feature = "vulkan", feature = "tensor")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResidentValidity {
    Ready,
    Uncertain,
    Discarded,
}

/// A logical tensor whose backing may be retained by an execution provider.
///
/// Consumers cannot construct this owner directly. It is returned by successful owned-result
/// kernels and keeps its provider resources alive independently of the thread cache.
/// Borrowing retains the logical value; moving transfers it. A move permits a provider to
/// consider storage reuse, but does not establish physical backing exclusivity by itself.
/// Ordinary `Drop` releases this owner's claim; readback does not consume it.
/// Current live backings retain thread-local `Rc` metadata or provider sessions. This owner is
/// neither `Send` nor `Sync`, including with CPU-only execution; cross-thread transfer is not
/// currently guaranteed.
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
    pub(super) backing: TensorBacking<T>,
}

pub(super) enum TensorBacking<T: PcuScalar> {
    #[cfg(all(feature = "vulkan", feature = "tensor"))]
    Vulkan {
        buffer: fusion_pcu_vulkan::PcuVulkanOwnedBuffer<T>,
        shape: std::rc::Rc<[usize]>,
        root: std::rc::Rc<fusion_pcu_vulkan::PcuVulkanBackend>,
        validity: ResidentValidity,
    },
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    #[cfg_attr(not(any(feature = "metal", feature = "tensor")), allow(dead_code))]
    // These provider-only builds have no initialized owner constructor.
    Device {
        tensor: DeviceTensor<T>,
        session: std::rc::Rc<Session>,
        validity: ResidentValidity,
    },
    #[cfg(feature = "mlx")]
    Mlx {
        array: fusion_pcu_mlx::MlxArray,
        shape: [usize; 2],
        root: std::rc::Rc<MlxSourceRoot>,
        marker: PhantomData<fn() -> T>,
    },
    #[cfg(feature = "mlx")]
    MlxEncoded {
        array: fusion_pcu_mlx::MlxEncodedArray,
        shape: std::rc::Rc<[usize]>,
        root: std::rc::Rc<MlxSourceRoot>,
        marker: PhantomData<fn() -> T>,
    },
    #[cfg(all(feature = "cpu", feature = "tensor"))]
    Cpu {
        values: Vec<T>,
        shape: std::rc::Rc<[usize]>,
    },
    #[cfg(not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        all(feature = "cpu", feature = "tensor"),
        all(feature = "vulkan", feature = "tensor")
    )))]
    #[allow(dead_code)]
    // Provider-off builds retain the public owner API without a constructor.
    Unavailable(PhantomData<fn() -> T>),
}

#[cfg(feature = "mlx")]
pub(super) struct MlxSourceRoot {
    pub(super) discovery: fusion_pcu_mlx::MlxDiscovery,
    pub(super) session: fusion_pcu_mlx::MlxSession,
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
    pub(super) const fn from_successful_output(
        tensor: DeviceTensor<T>,
        session: std::rc::Rc<Session>,
    ) -> Self {
        Self {
            backing: TensorBacking::Device {
                tensor,
                session,
                validity: ResidentValidity::Ready,
            },
        }
    }

    #[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
    #[cfg_attr(
        not(any(
            feature = "mlx",
            all(feature = "cpu", feature = "tensor"),
            all(feature = "vulkan", feature = "tensor")
        )),
        allow(clippy::unnecessary_wraps)
    )] // The common checked accessor rejects opaque backings when compiled.
    pub(super) const fn device_tensor(&self) -> Result<&DeviceTensor<T>, PcuArgumentError> {
        match &self.backing {
            TensorBacking::Device { tensor, .. } => Ok(tensor),
            #[cfg(any(
                feature = "mlx",
                all(feature = "cpu", feature = "tensor"),
                all(feature = "vulkan", feature = "tensor")
            ))]
            _ => Err(PcuArgumentError::UnsupportedResidentBorrow),
        }
    }

    #[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
    #[cfg_attr(
        not(any(
            feature = "mlx",
            all(feature = "cpu", feature = "tensor"),
            all(feature = "vulkan", feature = "tensor")
        )),
        allow(clippy::unnecessary_wraps)
    )] // The common checked accessor rejects opaque backings when compiled.
    pub(super) const fn session(&self) -> Result<&std::rc::Rc<Session>, PcuArgumentError> {
        match &self.backing {
            TensorBacking::Device { session, .. } => Ok(session),
            #[cfg(any(
                feature = "mlx",
                all(feature = "cpu", feature = "tensor"),
                all(feature = "vulkan", feature = "tensor")
            ))]
            _ => Err(PcuArgumentError::UnsupportedResidentBorrow),
        }
    }

    #[cfg(all(any(feature = "rocm", feature = "cuda"), feature = "tensor"))]
    #[cfg_attr(
        not(any(
            feature = "mlx",
            all(feature = "cpu", feature = "tensor"),
            all(feature = "vulkan", feature = "tensor")
        )),
        allow(clippy::unnecessary_wraps)
    )] // Opaque/host tensor variants must retain the fallible consuming boundary.
    pub(super) fn into_device_parts(
        self,
    ) -> Result<(DeviceTensor<T>, std::rc::Rc<Session>), super::PcuExecutionError> {
        match self.backing {
            TensorBacking::Device {
                tensor, session, ..
            } => Ok((tensor, session)),
            #[cfg(any(
                feature = "mlx",
                all(feature = "cpu", feature = "tensor"),
                all(feature = "vulkan", feature = "tensor")
            ))]
            _ => Err(super::PcuExecutionError::Argument(
                PcuArgumentError::UnsupportedResidentBorrow,
            )),
        }
    }

    #[allow(clippy::missing_const_for_fn)] // MLX checks live native access state.
    #[cfg_attr(
        all(
            feature = "cpu",
            feature = "tensor",
            not(any(feature = "rocm", feature = "cuda", feature = "metal", feature = "mlx"))
        ),
        allow(clippy::unnecessary_wraps)
    )] // Shared API also validates live device and opaque completion.
    pub(super) fn validate_initialized(&self) -> Result<(), PcuArgumentError> {
        match &self.backing {
            #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
            TensorBacking::Device { validity, .. } => match validity {
                ResidentValidity::Ready => Ok(()),
                ResidentValidity::Uncertain => Err(PcuArgumentError::ResidentCompletionUncertain),
                ResidentValidity::Discarded => Err(PcuArgumentError::ResidentValueDiscarded),
            },
            #[cfg(feature = "mlx")]
            TensorBacking::Mlx { array, .. } => array
                .validate_access_available()
                .map_err(|_| PcuArgumentError::ResidentCompletionUncertain),
            #[cfg(feature = "mlx")]
            TensorBacking::MlxEncoded { array, .. } => array
                .validate_access_available()
                .map_err(|_| PcuArgumentError::ResidentCompletionUncertain),
            #[cfg(all(feature = "cpu", feature = "tensor"))]
            TensorBacking::Cpu { .. } => Ok(()),
            #[cfg(all(feature = "vulkan", feature = "tensor"))]
            TensorBacking::Vulkan {
                buffer, validity, ..
            } => {
                match validity {
                    ResidentValidity::Uncertain => {
                        return Err(PcuArgumentError::ResidentCompletionUncertain);
                    }
                    ResidentValidity::Discarded => {
                        return Err(PcuArgumentError::ResidentValueDiscarded);
                    }
                    ResidentValidity::Ready => {}
                }
                buffer
                    .validate_access_available()
                    .map_err(|_| PcuArgumentError::ResidentCompletionUncertain)
            }
            #[cfg(not(any(
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                feature = "mlx",
                all(feature = "cpu", feature = "tensor"),
                all(feature = "vulkan", feature = "tensor")
            )))]
            TensorBacking::Unavailable(_) => Err(PcuArgumentError::ProviderUnavailable),
        }
    }

    #[cfg(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        all(feature = "cpu", feature = "tensor"),
        all(feature = "vulkan", feature = "tensor")
    ))]
    pub(super) fn validate_read(
        &self,
        expected_shape: PcuSourceShape,
    ) -> Result<(), PcuArgumentError> {
        if !expected_shape.accepts_resident_shape(self.shape()) {
            return Err(PcuArgumentError::ResidentShapeMismatch {
                expected: expected_shape,
            });
        }
        self.validate_initialized()?;
        match &self.backing {
            #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
            TensorBacking::Device { tensor, .. } => tensor.validate_access_available(),
            #[cfg(feature = "mlx")]
            TensorBacking::Mlx { .. } | TensorBacking::MlxEncoded { .. } => Ok(()),
            #[cfg(all(feature = "vulkan", feature = "tensor"))]
            TensorBacking::Vulkan { .. } => Ok(()),
            #[cfg(all(feature = "cpu", feature = "tensor"))]
            TensorBacking::Cpu { .. } => Ok(()),
        }
    }

    #[cfg(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        all(feature = "cpu", feature = "tensor"),
        all(feature = "vulkan", feature = "tensor")
    ))]
    fn read_argument(
        &self,
        target: PcuBindingRef,
        expected_shape: PcuSourceShape,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.validate_read(expected_shape)?;
        match &self.backing {
            #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
            TensorBacking::Device {
                tensor, session, ..
            } => Ok(PcuResidentReadArgument {
                argument: tensor.read_argument(target),
                shape: expected_shape,
                session,
            }
            .into_call_argument()),
            #[cfg(feature = "mlx")]
            TensorBacking::MlxEncoded { array, root, .. } => Ok(PcuCallArgument {
                shape: expected_shape,
                kind: PcuCallArgumentKind::MlxRead(PcuMlxReadArgument {
                    array,
                    root,
                    target,
                }),
            }),
            #[cfg(feature = "mlx")]
            TensorBacking::Mlx { .. } => Err(PcuArgumentError::UnsupportedResidentBorrow),
            #[cfg(all(feature = "vulkan", feature = "tensor"))]
            TensorBacking::Vulkan { buffer, root, .. } => Ok(PcuCallArgument {
                shape: expected_shape,
                kind: PcuCallArgumentKind::VulkanRead(PcuVulkanReadArgument {
                    argument: buffer.read_argument(target),
                    root,
                }),
            }),
            #[cfg(all(feature = "cpu", feature = "tensor"))]
            TensorBacking::Cpu { values, .. } => Ok(PcuCallArgument {
                shape: expected_shape,
                kind: PcuCallArgumentKind::CpuOwner(PcuHostArgument::read(target, values)),
            }),
        }
    }

    #[cfg(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        all(feature = "cpu", feature = "tensor"),
        all(feature = "vulkan", feature = "tensor")
    ))]
    fn write_argument(
        &mut self,
        target: PcuBindingRef,
        expected_shape: PcuSourceShape,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.validate_read(expected_shape)?;
        match &mut self.backing {
            #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
            TensorBacking::Device {
                tensor,
                session,
                validity,
            } => Ok(PcuResidentWriteArgument {
                argument: tensor.read_write_argument(target),
                shape: expected_shape,
                session,
                guard: ResidentWriteGuard::new(validity),
            }
            .into_call_argument()),
            #[cfg(feature = "mlx")]
            TensorBacking::MlxEncoded { array, root, .. } => Ok(PcuCallArgument {
                shape: expected_shape,
                kind: PcuCallArgumentKind::MlxWrite(PcuMlxWriteArgument {
                    array,
                    root,
                    target,
                }),
            }),
            #[cfg(feature = "mlx")]
            TensorBacking::Mlx { .. } => Err(PcuArgumentError::UnsupportedResidentBorrow),
            #[cfg(all(feature = "vulkan", feature = "tensor"))]
            TensorBacking::Vulkan {
                buffer,
                root,
                validity,
                ..
            } => Ok(PcuCallArgument {
                shape: expected_shape,
                kind: PcuCallArgumentKind::VulkanWrite(PcuVulkanWriteArgument {
                    argument: buffer.write_argument(target),
                    root,
                    guard: ResidentWriteGuard::new(validity),
                }),
            }),
            #[cfg(all(feature = "cpu", feature = "tensor"))]
            TensorBacking::Cpu { values, .. } => Ok(PcuCallArgument {
                shape: expected_shape,
                kind: PcuCallArgumentKind::CpuOwner(PcuHostArgument::read_write(target, values)),
            }),
        }
    }
}

#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    all(feature = "cpu", feature = "tensor"),
    all(feature = "vulkan", feature = "tensor")
))]
impl<T: PcuScalar, Shape> sealed::Sealed<T, Shape> for PcuTensor<T> {}
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    all(feature = "cpu", feature = "tensor"),
    all(feature = "vulkan", feature = "tensor")
))]
impl<T: PcuScalar> PcuReadStorage<T, ScalarShape> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.read_argument(target, PcuSourceShape::Scalar)
    }
}
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    all(feature = "cpu", feature = "tensor"),
    all(feature = "vulkan", feature = "tensor")
))]
impl<T: PcuScalar> PcuReadStorage<T, SliceShape> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        let length = self.len();
        self.read_argument(target, PcuSourceShape::Slice { length })
    }
}
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    all(feature = "cpu", feature = "tensor"),
    all(feature = "vulkan", feature = "tensor")
))]
impl<T: PcuScalar, const N: usize> PcuReadStorage<T, FixedArrayShape<N>> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.read_argument(target, PcuSourceShape::FixedArray { length: N })
    }
}
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    all(feature = "cpu", feature = "tensor"),
    all(feature = "vulkan", feature = "tensor")
))]
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
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    all(feature = "cpu", feature = "tensor"),
    all(feature = "vulkan", feature = "tensor")
))]
impl<T: PcuScalar> PcuWriteStorage<T, ScalarShape> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.write_argument(target, PcuSourceShape::Scalar)
    }
}
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    all(feature = "cpu", feature = "tensor"),
    all(feature = "vulkan", feature = "tensor")
))]
impl<T: PcuScalar> PcuWriteStorage<T, SliceShape> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        let length = self.len();
        self.write_argument(target, PcuSourceShape::Slice { length })
    }
}
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    all(feature = "cpu", feature = "tensor"),
    all(feature = "vulkan", feature = "tensor")
))]
impl<T: PcuScalar, const N: usize> PcuWriteStorage<T, FixedArrayShape<N>> for PcuTensor<T> {
    fn as_pcu_call_argument(
        &mut self,
        target: PcuBindingRef,
    ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
        self.write_argument(target, PcuSourceShape::FixedArray { length: N })
    }
}
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "mlx",
    all(feature = "cpu", feature = "tensor"),
    all(feature = "vulkan", feature = "tensor")
))]
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

#[cfg(feature = "mlx")]
pub(super) struct PcuMlxReadArgument<'a> {
    pub(super) array: &'a fusion_pcu_mlx::MlxEncodedArray,
    pub(super) root: &'a std::rc::Rc<MlxSourceRoot>,
    pub(super) target: PcuBindingRef,
}

#[cfg(feature = "mlx")]
pub(super) struct PcuMlxWriteArgument<'a> {
    pub(super) array: &'a mut fusion_pcu_mlx::MlxEncodedArray,
    pub(super) root: &'a std::rc::Rc<MlxSourceRoot>,
    pub(super) target: PcuBindingRef,
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

#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    all(feature = "vulkan", feature = "tensor")
))]
#[path = "arguments/resident_write/resident_write.rs"]
mod resident_write;
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal", all(feature = "vulkan", feature = "tensor")))]
#[rustfmt::skip]
pub(super) use resident_write::{
    ResidentWriteGuard,
    finish_resident_writes,
};

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

#[cfg(all(feature = "vulkan", feature = "tensor"))]
pub(super) struct PcuVulkanReadArgument<'a> {
    pub(super) argument: fusion_pcu_vulkan::PcuVulkanArgument<'a>,
    pub(super) root: &'a std::rc::Rc<fusion_pcu_vulkan::PcuVulkanBackend>,
}
#[cfg(all(feature = "vulkan", feature = "tensor"))]
pub(super) struct PcuVulkanWriteArgument<'a> {
    pub(super) argument: fusion_pcu_vulkan::PcuVulkanArgument<'a>,
    pub(super) root: &'a std::rc::Rc<fusion_pcu_vulkan::PcuVulkanBackend>,
    pub(super) guard: ResidentWriteGuard<'a>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unread_declarations_retain_type_access_but_no_storage_or_affinity() {
        let target = PcuBindingRef::new(4, 17);
        let argument = PcuCallArgument::unused_read::<crate::PcuU512>(target);
        assert_eq!(argument.shape, PcuSourceShape::Slice { length: 0 });
        #[cfg_attr(
            not(any(
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                feature = "mlx",
                all(feature = "cpu", feature = "tensor")
            )),
            allow(irrefutable_let_patterns)
        )]
        // A disabled facade has only the host variant; resident builds test this discrimination.
        let PcuCallArgumentKind::Host(metadata) = argument.into_parts().1 else {
            panic!("unread declaration must not carry a resident lease")
        };
        assert_eq!(metadata.target(), target);
        assert_eq!(metadata.scalar(), crate::PcuScalarType::U512);
        assert_eq!(metadata.access(), crate::PcuBindingAccess::ReadOnly);
        assert!(metadata.bytes().is_empty());
    }

    #[test]
    fn untouched_mutable_declarations_retain_read_write_metadata() {
        let target = PcuBindingRef::new(4, 17);
        let argument = PcuCallArgument::unused_read_write::<crate::PcuU512>(target);
        assert_eq!(argument.shape, PcuSourceShape::Slice { length: 0 });
        #[cfg_attr(
            not(any(
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                feature = "mlx",
                all(feature = "cpu", feature = "tensor")
            )),
            allow(irrefutable_let_patterns)
        )]
        // Disabled providers leave only the host variant; no resident lease is manufactured.
        let PcuCallArgumentKind::Host(metadata) = argument.into_parts().1 else {
            panic!("untouched declaration must not carry a resident lease")
        };
        assert_eq!(metadata.target(), target);
        assert_eq!(metadata.scalar(), crate::PcuScalarType::U512);
        assert_eq!(metadata.access(), crate::PcuBindingAccess::ReadWrite);
        assert!(metadata.bytes().is_empty());
    }

    #[cfg(not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "mlx",
        feature = "cpu",
        feature = "vulkan"
    )))]
    mod untouched_owner {
        use super::*;

        struct UnavailableOwner;
        impl sealed::Sealed<f32, SliceShape> for UnavailableOwner {}
        impl PcuWriteStorage<f32, SliceShape> for UnavailableOwner {
            fn as_pcu_call_argument(
                &mut self,
                _: PcuBindingRef,
            ) -> Result<PcuCallArgument<'_>, PcuArgumentError> {
                panic!("unused owner storage must never be inspected")
            }
        }

        #[crate::pcu(invocations = 1, crate_path = crate)]
        fn copy(input: &[f32], ghost: &mut [f32], output: &mut [f32]) {
            let id = pcu::context::global_invocation_id();
            output[id] = input[id];
        }

        #[test]
        fn ordinary_source_refuses_without_probing_unused_owner() {
            let input = [1.0_f32];
            let mut output = [7.0_f32];
            // No provider is enabled. The relevant failure is selection, not this
            // unrelated owner's unavailable storage or a hidden CPU fallback.
            assert!(copy(&input, &mut UnavailableOwner, &mut output).is_err());
            assert_eq!(output.map(f32::to_bits), [7.0_f32.to_bits()]);
        }
    }

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
        #[cfg(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "mlx",
            all(feature = "cpu", feature = "tensor"),
            all(feature = "vulkan", feature = "tensor")
        ))]
        let PcuCallArgumentKind::Host(host_argument) = argument.kind else {
            panic!("host scalar conversion remains host-backed");
        };
        #[cfg(not(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "mlx",
            all(feature = "cpu", feature = "tensor"),
            all(feature = "vulkan", feature = "tensor")
        )))]
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
        #[cfg(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "mlx",
            all(feature = "cpu", feature = "tensor"),
            all(feature = "vulkan", feature = "tensor")
        ))]
        let PcuCallArgumentKind::Host(host_argument) = argument.kind else {
            panic!("host matrix conversion remains host-backed");
        };
        #[cfg(not(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "mlx",
            all(feature = "cpu", feature = "tensor"),
            all(feature = "vulkan", feature = "tensor")
        )))]
        let PcuCallArgumentKind::Host(host_argument) = argument.kind;
        assert_eq!(host_argument.access(), crate::PcuBindingAccess::ReadWrite);
        assert_eq!(host_argument.bytes().len(), 6 * core::mem::size_of::<u32>());
    }

    #[cfg(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        all(feature = "vulkan", feature = "tensor")
    ))]
    #[test]
    fn resident_write_guard_distinguishes_prelaunch_discard_and_uncertain_completion() {
        let mut validity = ResidentValidity::Ready;
        {
            let _guard = ResidentWriteGuard::new(&mut validity);
        }
        assert_eq!(validity, ResidentValidity::Ready);
        {
            let mut guard = ResidentWriteGuard::new(&mut validity);
            guard.mark_may_have_written();
            guard.mark_not_submitted();
        }
        assert_eq!(validity, ResidentValidity::Ready);
        {
            let mut guard = ResidentWriteGuard::new(&mut validity);
            guard.mark_may_have_written();
            guard.mark_discarded();
        }
        assert_eq!(validity, ResidentValidity::Discarded);
        {
            let _guard = ResidentWriteGuard::new(&mut validity);
        }
        assert_eq!(validity, ResidentValidity::Discarded);
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

    #[cfg(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        all(feature = "vulkan", feature = "tensor")
    ))]
    #[test]
    fn resident_outcome_classification_applies_to_every_mutable_result() {
        use super::super::PcuExecutionError;
        let fatal = || {
            PcuExecutionError::ArithmeticFault(crate::PcuExecutionFault {
                kind: crate::PcuExecutionFaultKind::DivideByZero,
                invocation_id: 5,
                recovered: false,
            })
        };
        let recovered = PcuExecutionError::ArithmeticFault(crate::PcuExecutionFault {
            kind: crate::PcuExecutionFaultKind::ArithmeticOverflow,
            invocation_id: 5,
            recovered: true,
        });
        let cases = [
            (
                false,
                false,
                Err(PcuExecutionError::Argument(
                    PcuArgumentError::SessionMismatch,
                )),
                None,
            ),
            (false, false, Err(fatal()), None),
            (true, false, Ok(()), Some(ResidentValidity::Ready)),
            (true, false, Err(recovered), Some(ResidentValidity::Ready)),
            (true, false, Err(fatal()), Some(ResidentValidity::Discarded)),
            (
                true,
                false,
                Err(PcuExecutionError::Argument(
                    PcuArgumentError::SessionMismatch,
                )),
                Some(ResidentValidity::Discarded),
            ),
            (true, true, Ok(()), Some(ResidentValidity::Uncertain)),
            (false, true, Err(fatal()), Some(ResidentValidity::Uncertain)),
        ];
        for prior in [
            ResidentValidity::Ready,
            ResidentValidity::Uncertain,
            ResidentValidity::Discarded,
        ] {
            for (may_write, uncertain, result, expected) in &cases {
                let (mut first, mut second) = (prior, prior);
                {
                    let mut guards = [
                        Some(ResidentWriteGuard::new(&mut first)),
                        None,
                        Some(ResidentWriteGuard::new(&mut second)),
                    ];
                    if !uncertain {
                        for guard in guards.iter_mut().flatten() {
                            guard.mark_may_have_written();
                        }
                    }
                    finish_resident_writes(&mut guards, *may_write, *uncertain, result);
                }
                assert_eq!(first, expected.unwrap_or(prior));
                assert_eq!(second, expected.unwrap_or(prior));
            }
            let mut value = prior;
            {
                let _unsubmitted = ResidentWriteGuard::new(&mut value);
                // A later argument conversion/preparation rejection drops this borrow.
            }
            assert_eq!(value, prior);
        }
    }
}
