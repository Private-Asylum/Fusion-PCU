//! Typed construction for scalar identity copies.

#[rustfmt::skip]
use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchValueId,
    PcuError,
    PcuScalar,
    PcuValueType,
    PcuValueTypeCaps,
};
use crate::model::PcuDispatchKernelBuilder;

const GRID_STRIDE_BODY: [crate::PcuDispatchOp<'static>; 2] = [
    crate::PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: PcuDispatchValueId(1),
        binding: crate::PcuBindingRef::new(0, 0),
        index: PcuDispatchIndex::GridStrideId,
    }),
    crate::PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: crate::PcuBindingRef::new(0, 1),
        index: PcuDispatchIndex::GridStrideId,
        value: PcuDispatchValueId(1),
    }),
];

const GRID_STRIDE_BROADCAST_BODY: [crate::PcuDispatchOp<'static>; 2] = [
    crate::PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: PcuDispatchValueId(1),
        binding: crate::PcuBindingRef::new(0, 0),
        index: PcuDispatchIndex::BindingElementZero,
    }),
    GRID_STRIDE_BODY[1],
];

/// Why a typed scalar identity kernel could not be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuScalarIdentityBuildError {
    /// Identity copies require exactly the read-only source and writable destination bindings.
    BindingCount,
    /// Bindings must be ordered at set zero, slots zero and one.
    BindingLocation,
    /// Both resources must have `T`'s sealed scalar type.
    BindingType,
    /// Identity resources must use storage memory.
    StorageClass,
    /// The source must be read-only and the destination writable.
    Access,
    /// The invocation count must be nonzero.
    EmptyShape,
    /// A sealed scalar's Rust and canonical encoding layout metadata is inconsistent.
    InvalidScalarLayout,
    /// The bounded dispatch builder returned an error.
    Builder(PcuError),
}

impl core::fmt::Display for PcuScalarIdentityBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BindingCount => f.write_str("PCU identity requires two resource bindings"),
            Self::BindingLocation => {
                f.write_str("PCU identity bindings must occupy set zero, slots zero and one")
            }
            Self::BindingType => {
                f.write_str("PCU identity binding scalar types must match the specialization")
            }
            Self::StorageClass => f.write_str("PCU identity bindings require storage memory"),
            Self::Access => {
                f.write_str("PCU identity requires a readable source and writable destination")
            }
            Self::EmptyShape => f.write_str("PCU identity invocation count must be nonzero"),
            Self::InvalidScalarLayout => {
                f.write_str("PCU scalar host and transfer layouts are inconsistent")
            }
            Self::Builder(error) => write!(f, "PCU identity construction failed: {error}"),
        }
    }
}

impl core::error::Error for PcuScalarIdentityBuildError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Builder(error) => Some(error),
            _ => None,
        }
    }
}

/// Constructs one typed scalar load/store per invocation.
///
/// This profile describes memory transport only. It does not grant any backend a promise that
/// it supports the selected scalar or that the copy will be zero-copy. `T: PcuScalar` is sealed.
pub struct PcuScalarIdentityBuilder<T: PcuScalar>(core::marker::PhantomData<fn() -> T>);

impl<T: PcuScalar> PcuScalarIdentityBuilder<T> {
    /// Builds a direct identity-copy dispatch after checking its bindings and logical shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the invocation count or binding contract is invalid, or if the bounded
    /// dispatch builder rejects an appended operation.
    pub fn build<'a>(
        kernel_id: u32,
        invocations: u32,
        bindings: &'a [PcuBinding<'a>],
    ) -> Result<PcuDispatchKernelBuilder<'a, 3>, PcuScalarIdentityBuildError> {
        Self::build_direct(
            kernel_id,
            invocations,
            bindings,
            PcuDispatchIndex::InvocationId,
        )
    }

    /// Broadcasts one immutable scalar to each direct invocation's output element.
    ///
    /// # Errors
    /// Returns an error for an invalid shape, scalar layout or binding schema.
    pub fn build_broadcast<'a>(
        kernel_id: u32,
        invocations: u32,
        bindings: &'a [PcuBinding<'a>],
    ) -> Result<PcuDispatchKernelBuilder<'a, 3>, PcuScalarIdentityBuildError> {
        Self::build_direct(
            kernel_id,
            invocations,
            bindings,
            PcuDispatchIndex::BindingElementZero,
        )
    }

    fn build_direct<'a>(
        kernel_id: u32,
        invocations: u32,
        bindings: &'a [PcuBinding<'a>],
        source_index: PcuDispatchIndex,
    ) -> Result<PcuDispatchKernelBuilder<'a, 3>, PcuScalarIdentityBuildError> {
        let builder = Self::validate(kernel_id, invocations, bindings)?;
        let source = crate::PcuBindingRef::new(0, 0);
        let destination = crate::PcuBindingRef::new(0, 1);
        builder
            .with_data_op(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: source,
                index: source_index,
            })
            .and_then(|builder| {
                builder.with_data_op(PcuDispatchDataOp::BindingStore {
                    binding: destination,
                    index: PcuDispatchIndex::InvocationId,
                    value: PcuDispatchValueId(1),
                })
            })
            .and_then(|builder| builder.with_control_op(PcuDispatchControlOp::Return))
            .map_err(PcuScalarIdentityBuildError::Builder)
    }

    /// Builds a generic scalar identity copy in the canonical grid-stride region.
    ///
    /// Each invocation copies `id + n * invocations` while that index is below `extent`.
    /// The extent is an already-specialized value, so the macro can supply a const generic.
    ///
    /// # Errors
    ///
    /// Returns an error if either bound or the binding contract is invalid, or if the bounded
    /// dispatch builder rejects an appended operation.
    pub fn build_grid_stride<'a>(
        kernel_id: u32,
        invocations: u32,
        extent: u32,
        bindings: &'a [PcuBinding<'a>],
    ) -> Result<PcuDispatchKernelBuilder<'a, 3>, PcuScalarIdentityBuildError> {
        Self::build_grid(kernel_id, invocations, extent, bindings, &GRID_STRIDE_BODY)
    }

    /// Broadcasts one immutable scalar across the canonical grid-stride output extent.
    ///
    /// # Errors
    /// Returns an error for an empty bound, invalid layout or invalid binding schema.
    pub fn build_grid_stride_broadcast<'a>(
        kernel_id: u32,
        invocations: u32,
        extent: u32,
        bindings: &'a [PcuBinding<'a>],
    ) -> Result<PcuDispatchKernelBuilder<'a, 3>, PcuScalarIdentityBuildError> {
        Self::build_grid(
            kernel_id,
            invocations,
            extent,
            bindings,
            &GRID_STRIDE_BROADCAST_BODY,
        )
    }

    fn build_grid<'a>(
        kernel_id: u32,
        invocations: u32,
        extent: u32,
        bindings: &'a [PcuBinding<'a>],
        body: &'static [crate::PcuDispatchOp<'static>],
    ) -> Result<PcuDispatchKernelBuilder<'a, 3>, PcuScalarIdentityBuildError> {
        if extent == 0 {
            return Err(PcuScalarIdentityBuildError::EmptyShape);
        }
        let builder = Self::validate(kernel_id, invocations, bindings)?;
        builder
            .with_op(crate::PcuDispatchOp::GridStrideLoop { extent, body })
            .and_then(|builder| builder.with_control_op(PcuDispatchControlOp::Return))
            .map_err(PcuScalarIdentityBuildError::Builder)
    }

    fn validate<'a>(
        kernel_id: u32,
        invocations: u32,
        bindings: &'a [PcuBinding<'a>],
    ) -> Result<PcuDispatchKernelBuilder<'a, 3>, PcuScalarIdentityBuildError> {
        if invocations == 0 {
            return Err(PcuScalarIdentityBuildError::EmptyShape);
        }
        if T::HOST_SIZE != core::mem::size_of::<T>()
            || T::HOST_ALIGNMENT != core::mem::align_of::<T>()
            || T::ENCODED_SIZE != core::mem::size_of::<T::Encoded>()
        {
            return Err(PcuScalarIdentityBuildError::InvalidScalarLayout);
        }
        if bindings.len() != 2 {
            return Err(PcuScalarIdentityBuildError::BindingCount);
        }
        for (index, binding) in bindings.iter().enumerate() {
            let expected_slot = match index {
                0 => 0,
                1 => 1,
                _ => return Err(PcuScalarIdentityBuildError::BindingCount),
            };
            if binding.set != 0 || binding.binding != expected_slot {
                return Err(PcuScalarIdentityBuildError::BindingLocation);
            }
            if binding.storage != PcuBindingStorageClass::Storage {
                return Err(PcuScalarIdentityBuildError::StorageClass);
            }
            if binding.binding_type != PcuBindingType::Value(PcuValueType::Scalar(T::TYPE)) {
                return Err(PcuScalarIdentityBuildError::BindingType);
            }
        }
        if bindings[0].access != PcuBindingAccess::ReadOnly
            || bindings[1].access != PcuBindingAccess::ReadWrite
        {
            return Err(PcuScalarIdentityBuildError::Access);
        }
        let builder = PcuDispatchKernelBuilder::<3>::new(kernel_id, "main", [invocations, 1, 1])
            .with_bindings(bindings)
            .with_type_caps(
                PcuValueTypeCaps::for_scalar(T::TYPE).union(PcuValueTypeCaps::SCALAR_VALUES),
            )
            .with_feature_caps(
                crate::PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
                    .union(crate::PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
            );
        Ok(builder)
    }
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        PcuScalarIdentityBuildError as Error,
        PcuScalarIdentityBuilder,
    };
    #[rustfmt::skip]
    use crate::{
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuBindingType,
        PcuDispatchDataOp,
        PcuDispatchOp,
        PcuValueType,
        PcuValueTypeCaps,
    };

    fn bindings<T: crate::PcuScalar>() -> [PcuBinding<'static>; 2] {
        [
            PcuBinding::scalar::<T>(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<T>(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadWrite,
            ),
        ]
    }

    #[test]
    fn broadcast_builder_keeps_single_element_input_and_indexed_output_distinct() {
        let bindings = bindings::<crate::PcuU512>();
        let direct =
            PcuScalarIdentityBuilder::<crate::PcuU512>::build_broadcast(7, 9, &bindings).unwrap();
        let grid = PcuScalarIdentityBuilder::<crate::PcuU512>::build_grid_stride_broadcast(
            7, 3, 17, &bindings,
        )
        .unwrap();
        for kernel in [direct.ir(), grid.ir()] {
            crate::validate_scalar_broadcast_kernel(&kernel, crate::PcuScalarType::U512).unwrap();
            assert!(
                crate::validate_scalar_identity_kernel(&kernel, crate::PcuScalarType::U512)
                    .is_err()
            );
            crate::validate_typed_dispatch_value_flow(&kernel).unwrap();
            assert_eq!(
                kernel.minimum_binding_elements_for(crate::PcuBindingRef::new(0, 0), 9),
                1
            );
            assert_eq!(
                kernel.minimum_binding_elements_for(crate::PcuBindingRef::new(0, 1), 9),
                if matches!(
                    kernel.ops.first(),
                    Some(PcuDispatchOp::GridStrideLoop { .. })
                ) {
                    17
                } else {
                    9
                }
            );
        }
    }

    #[test]
    fn builder_preserves_generic_scalar_type_and_permissions() {
        let bindings = bindings::<u64>();
        let builder = PcuScalarIdentityBuilder::<u64>::build(7, 9, &bindings)
            .expect("well-formed typed identity bindings build");
        let kernel = builder.ir();
        assert_eq!(
            kernel.bindings[0].binding_type,
            PcuBindingType::Value(PcuValueType::u64())
        );
        assert_eq!(kernel.bindings[0].access, PcuBindingAccess::ReadOnly);
        assert_eq!(kernel.bindings[1].access, PcuBindingAccess::ReadWrite);
        assert!(
            kernel
                .required_type_support()
                .contains(PcuValueTypeCaps::UINT64)
        );
        assert!(matches!(
            kernel.ops[0],
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                binding: PcuBindingRef { binding: 0, .. },
                ..
            })
        ));
        assert!(matches!(
            kernel.ops[1],
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef { binding: 1, .. },
                ..
            })
        ));
    }

    #[test]
    fn builder_rejects_wrong_type_or_access() {
        let mut wrong_type = bindings::<u64>();
        wrong_type[0] = PcuBinding::scalar::<f32>(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        );
        assert_eq!(
            PcuScalarIdentityBuilder::<u64>::build(7, 9, &wrong_type).unwrap_err(),
            Error::BindingType
        );

        let mut wrong_access = bindings::<u64>();
        wrong_access[0].access = PcuBindingAccess::ReadWrite;
        assert_eq!(
            PcuScalarIdentityBuilder::<u64>::build(7, 9, &wrong_access).unwrap_err(),
            Error::Access
        );
    }

    #[test]
    fn builder_rejects_empty_grid_extent_and_emits_bounded_region() {
        let bindings = bindings::<u16>();
        assert_eq!(
            PcuScalarIdentityBuilder::<u16>::build_grid_stride(7, 3, 0, &bindings).unwrap_err(),
            Error::EmptyShape
        );
        let builder = PcuScalarIdentityBuilder::<u16>::build_grid_stride(7, 3, 11, &bindings)
            .expect("valid grid-stride copy builds");
        let kernel = builder.ir();
        assert!(matches!(
            kernel.ops,
            [
                crate::PcuDispatchOp::GridStrideLoop { extent: 11, body },
                crate::PcuDispatchOp::Control(crate::PcuDispatchControlOp::Return)
            ] if body.len() == 2
        ));
    }
}
