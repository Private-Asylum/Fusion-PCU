//! Routing and scheduling contracts for render-family PCU work.
//!
//! This module owns the submission seam for render-family kernels without dragging framebuffer or
//! presentation policy into `fusion-pcu`. Backends may implement only the render families they can
//! honestly lower, while runtime support still decides whether the current device/executor can
//! actually admit that family.

use crate::contract::{
    PcuBaseContract,
    PcuError,
    PcuInvocationBindings,
    PcuInvocationParameters,
    PcuKernelIrContract,
};
use crate::dispatch::{
    PcuFiniteHandle,
    validate_invocation_bindings,
    validate_parameters,
};
use crate::{
    PcuMeshKernelIr,
    PcuRasterKernelIr,
    PcuRayTraceKernelIr,
    PcuRenderKernel,
};

/// Borrowed render submission descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuRenderSubmission<'a> {
    pub kernel: &'a PcuRenderKernel<'a>,
}

/// Borrowed raster-family render submission descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuRasterSubmission<'a> {
    pub kernel: &'a PcuRasterKernelIr<'a>,
}

/// Borrowed mesh-family render submission descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMeshSubmission<'a> {
    pub kernel: &'a PcuMeshKernelIr<'a>,
}

/// Borrowed ray-trace-family render submission descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuRayTraceSubmission<'a> {
    pub kernel: &'a PcuRayTraceKernelIr<'a>,
}

/// Routing contract for finite render-family work.
pub trait PcuRenderContract {
    type RenderHandle: PcuFiniteHandle;

    /// Submits one finite render-family kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest admission, scheduling, or execution-substrate failure.
    fn submit_render(
        &self,
        submission: PcuRenderSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::RenderHandle, PcuError>;
}

/// Base trait for backends that surface one or more render families.
pub trait PcuRenderFamilyBackend: PcuBaseContract {
    type RenderHandle: PcuFiniteHandle;
}

/// Optional direct-lowering surface for raster-family work.
pub trait PcuRasterFamilyBackend: PcuRenderFamilyBackend {
    /// Returns the raster-family runtime support surfaced by this backend.
    #[must_use]
    fn raster_support(&self) -> crate::PcuRasterSupport {
        self.support().render_support.raster
    }

    /// Submits one already-validated direct raster-family kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest backend admission or execution failure.
    fn submit_raster_direct(
        &self,
        submission: PcuRasterSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::RenderHandle, PcuError>;
}

/// Optional direct-lowering surface for mesh-family work.
pub trait PcuMeshFamilyBackend: PcuRenderFamilyBackend {
    /// Returns the mesh-family runtime support surfaced by this backend.
    #[must_use]
    fn mesh_support(&self) -> crate::PcuMeshSupport {
        self.support().render_support.mesh
    }

    /// Submits one already-validated direct mesh-family kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest backend admission or execution failure.
    fn submit_mesh_direct(
        &self,
        submission: PcuMeshSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::RenderHandle, PcuError>;
}

/// Optional direct-lowering surface for ray-trace-family work.
pub trait PcuRayTraceFamilyBackend: PcuRenderFamilyBackend {
    /// Returns the ray-trace-family runtime support surfaced by this backend.
    #[must_use]
    fn ray_trace_support(&self) -> crate::PcuRayTraceSupport {
        self.support().render_support.ray_trace
    }

    /// Submits one already-validated direct ray-trace-family kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest backend admission or execution failure.
    fn submit_ray_trace_direct(
        &self,
        submission: PcuRayTraceSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::RenderHandle, PcuError>;
}

/// Direct-execution backend contract for render-family submission.
pub trait PcuDirectRenderBackend: PcuRenderFamilyBackend {
    /// Submits one already-validated direct render-family kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest backend admission or execution failure.
    fn submit_render_direct(
        &self,
        submission: PcuRenderSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::RenderHandle, PcuError>;
}

impl<T> PcuRenderContract for T
where
    T: PcuDirectRenderBackend,
{
    type RenderHandle = T::RenderHandle;

    fn submit_render(
        &self,
        submission: PcuRenderSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::RenderHandle, PcuError> {
        validate_parameters(submission.kernel.signature(), parameters)?;
        validate_invocation_bindings(submission.kernel.signature(), bindings)?;

        if self
            .support()
            .supports_kernel_direct(crate::PcuKernel::Render(*submission.kernel))
        {
            return self.submit_render_direct(submission, bindings, parameters);
        }

        Err(PcuError::unsupported())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuDirectRenderBackend,
        PcuRenderContract,
        PcuRenderSubmission,
    };
    use crate::{
        PcuBaseContract,
        PcuBinding,
        PcuBindingAccess,
        PcuBindingStorageClass,
        PcuDispatchPolicyCaps,
        PcuExecutorDescriptor,
        PcuExecutorId,
        PcuExecutorOrigin,
        PcuExecutorClass,
        PcuExecutorSupport,
        PcuError,
        PcuFeatureSupport,
        PcuFiniteHandle,
        PcuFiniteState,
        PcuImplementationKind,
        PcuInvocationBindings,
        PcuInvocationParameters,
        PcuKernelId,
        PcuParameter,
        PcuParameterBinding,
        PcuParameterSlot,
        PcuParameterValue,
        PcuPrimitiveCaps,
        PcuPrimitiveSupport,
        PcuRasterFeatureCaps,
        PcuRasterKernelIr,
        PcuRenderFamilyCaps,
        PcuRenderKernel,
        PcuRenderSupport,
        PcuSupport,
        PcuValueType,
        PcuValueTypeCaps,
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestRenderHandle;

    impl PcuFiniteHandle for TestRenderHandle {
        fn state(&self) -> Result<PcuFiniteState, PcuError> {
            Ok(PcuFiniteState::Complete)
        }

        fn wait(self) -> Result<(), PcuError> {
            Ok(())
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct TestRenderBackend {
        support: PcuSupport,
        executors: &'static [PcuExecutorDescriptor],
    }

    impl PcuBaseContract for TestRenderBackend {
        fn support(&self) -> PcuSupport {
            self.support
        }

        fn executors(&self) -> &'static [PcuExecutorDescriptor] {
            self.executors
        }
    }

    impl super::PcuRenderFamilyBackend for TestRenderBackend {
        type RenderHandle = TestRenderHandle;
    }

    impl PcuDirectRenderBackend for TestRenderBackend {
        fn submit_render_direct(
            &self,
            _submission: PcuRenderSubmission<'_>,
            _bindings: PcuInvocationBindings<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::RenderHandle, PcuError> {
            Ok(TestRenderHandle)
        }
    }

    const RENDER_EXECUTOR: [PcuExecutorDescriptor; 1] = [PcuExecutorDescriptor {
        id: PcuExecutorId(7),
        name: "gpu0",
        class: PcuExecutorClass::Compute,
        origin: PcuExecutorOrigin::Synthetic,
        support: PcuExecutorSupport {
            primitives: PcuPrimitiveCaps::RENDER,
            dispatch_policy: PcuDispatchPolicyCaps::ORDERED_SUBMISSION,
            value_types: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::VECTOR_VALUES),
            dispatch_instructions: crate::PcuDispatchOpCaps::empty(),
            dispatch_features: crate::PcuDispatchFeatureCaps::empty(),
            render_families: PcuRenderFamilyCaps::RASTER,
            raster_features: PcuRasterFeatureCaps::VERTEX_STAGE
                .union(PcuRasterFeatureCaps::FRAGMENT_STAGE),
            mesh_features: crate::PcuMeshFeatureCaps::empty(),
            ray_trace_features: crate::PcuRayTraceFeatureCaps::empty(),
            stream_instructions: crate::PcuStreamCapabilities::empty(),
            command_instructions: crate::PcuCommandOpCaps::empty(),
            transaction_features: crate::PcuTransactionFeatureCaps::empty(),
            signal_instructions: crate::PcuSignalOpCaps::empty(),
        },
    }];

    const RASTER_BINDINGS: [PcuBinding<'static>; 1] = [PcuBinding::value(
        Some("uniforms"),
        0,
        0,
        PcuBindingStorageClass::Uniform,
        PcuBindingAccess::ReadOnly,
        PcuValueType::vector(crate::PcuScalarType::F32, 4),
    )];

    const RASTER_PARAMETERS: [PcuParameter<'static>; 1] = [PcuParameter::named(
        PcuParameterSlot(0),
        "scale",
        PcuValueType::f32(),
    )];

    fn raster_kernel<'a>() -> PcuRenderKernel<'a> {
        PcuRenderKernel::Raster(PcuRasterKernelIr {
            id: PcuKernelId(21),
            entry_point: "raster-main",
            bindings: &RASTER_BINDINGS,
            ports: &[],
            parameters: &RASTER_PARAMETERS,
            vertex_entry: "vs_main",
            fragment_entry: Some("fs_main"),
            type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::VECTOR_VALUES),
            features: PcuRasterFeatureCaps::VERTEX_STAGE
                .union(PcuRasterFeatureCaps::FRAGMENT_STAGE),
        })
    }

    fn direct_backend() -> TestRenderBackend {
        let mut support = PcuSupport::unsupported();
        support.implementation = PcuImplementationKind::Native;
        support.executor_count = 1;
        support.primitive_support = PcuPrimitiveSupport {
            primitives: PcuFeatureSupport::new(PcuPrimitiveCaps::RENDER, PcuPrimitiveCaps::empty()),
        };
        support.value_type_support = PcuFeatureSupport::new(
            PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::VECTOR_VALUES),
            PcuValueTypeCaps::empty(),
        );
        support.dispatch_support = crate::PcuDispatchSupport {
            flags: crate::PcuDispatchPolicyCaps::ORDERED_SUBMISSION,
            instructions: PcuFeatureSupport::new(
                crate::PcuDispatchOpCaps::empty(),
                crate::PcuDispatchOpCaps::empty(),
            ),
            features: PcuFeatureSupport::new(
                crate::PcuDispatchFeatureCaps::empty(),
                crate::PcuDispatchFeatureCaps::empty(),
            ),
        };
        support.render_support = PcuRenderSupport {
            families: PcuFeatureSupport::new(
                PcuRenderFamilyCaps::RASTER,
                PcuRenderFamilyCaps::empty(),
            ),
            raster: crate::PcuRasterSupport {
                features: PcuFeatureSupport::new(
                    PcuRasterFeatureCaps::VERTEX_STAGE.union(PcuRasterFeatureCaps::FRAGMENT_STAGE),
                    PcuRasterFeatureCaps::empty(),
                ),
            },
            mesh: crate::PcuMeshSupport::unsupported(),
            ray_trace: crate::PcuRayTraceSupport::unsupported(),
        };
        TestRenderBackend {
            support,
            executors: &RENDER_EXECUTOR,
        }
    }

    #[test]
    fn render_contract_submits_supported_raster_kernel() {
        let backend = direct_backend();
        let kernel = raster_kernel();
        let bindings = PcuInvocationBindings::empty();
        let parameters = PcuInvocationParameters {
            bindings: &[PcuParameterBinding::new(
                PcuParameterSlot(0),
                PcuParameterValue::from_f32(1.0),
            )],
        };

        let result = backend.submit_render(
            PcuRenderSubmission { kernel: &kernel },
            bindings,
            parameters,
        );

        assert!(result.is_ok());
    }
}
