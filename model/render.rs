//! Render-family vocabulary and backend-neutral kernel descriptors.

use core::ops::{
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
};

use crate::{
    PcuBinding,
    PcuDispatchPolicyCaps,
    PcuInvocationModel,
    PcuIrKind,
    PcuKernelId,
    PcuKernelIrContract,
    PcuKernelSignature,
    PcuParameter,
    PcuPort,
    PcuValueTypeCaps,
};

/// Coarse render-family support surfaced by one backend/device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuRenderFamilyCaps(u32);

impl PcuRenderFamilyCaps {
    pub const RASTER: Self = Self(1 << 0);
    pub const MESH: Self = Self(1 << 1);
    pub const RAY_TRACE: Self = Self(1 << 2);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for PcuRenderFamilyCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuRenderFamilyCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuRenderFamilyCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuRenderFamilyCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Feature caps for raster-family render work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuRasterFeatureCaps(u32);

impl PcuRasterFeatureCaps {
    pub const VERTEX_STAGE: Self = Self(1 << 0);
    pub const FRAGMENT_STAGE: Self = Self(1 << 1);
    pub const INDEXED_DRAW: Self = Self(1 << 2);
    pub const INDIRECT_DRAW: Self = Self(1 << 3);
    pub const MULTI_DRAW: Self = Self(1 << 4);
    pub const SAMPLE_RATE_SHADING: Self = Self(1 << 5);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for PcuRasterFeatureCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuRasterFeatureCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuRasterFeatureCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuRasterFeatureCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Feature caps for mesh-family render work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuMeshFeatureCaps(u32);

impl PcuMeshFeatureCaps {
    pub const TASK_STAGE: Self = Self(1 << 0);
    pub const MESH_STAGE: Self = Self(1 << 1);
    pub const FRAGMENT_STAGE: Self = Self(1 << 2);
    pub const INDIRECT_DRAW: Self = Self(1 << 3);
    pub const MULTI_DRAW: Self = Self(1 << 4);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for PcuMeshFeatureCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuMeshFeatureCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuMeshFeatureCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuMeshFeatureCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Feature caps for ray-trace-family render work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuRayTraceFeatureCaps(u32);

impl PcuRayTraceFeatureCaps {
    pub const MISS_SHADERS: Self = Self(1 << 0);
    pub const CLOSEST_HIT_SHADERS: Self = Self(1 << 1);
    pub const ANY_HIT_SHADERS: Self = Self(1 << 2);
    pub const CALLABLE_SHADERS: Self = Self(1 << 3);
    pub const INDIRECT_TRACE: Self = Self(1 << 4);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for PcuRayTraceFeatureCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuRayTraceFeatureCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuRayTraceFeatureCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuRayTraceFeatureCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Raster-family render kernel descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuRasterKernelIr<'a> {
    pub id: PcuKernelId,
    pub entry_point: &'a str,
    pub bindings: &'a [PcuBinding<'a>],
    pub ports: &'a [PcuPort<'a>],
    pub parameters: &'a [PcuParameter<'a>],
    pub vertex_entry: &'a str,
    pub fragment_entry: Option<&'a str>,
    pub type_caps: PcuValueTypeCaps,
    pub features: PcuRasterFeatureCaps,
}

impl PcuRasterKernelIr<'_> {
    #[must_use]
    pub const fn required_dispatch_policy(&self) -> PcuDispatchPolicyCaps {
        PcuDispatchPolicyCaps::ORDERED_SUBMISSION
    }

    #[must_use]
    pub const fn required_type_support(&self) -> PcuValueTypeCaps {
        self.type_caps
    }

    #[must_use]
    pub const fn required_feature_support(&self) -> PcuRasterFeatureCaps {
        self.features
    }
}

impl PcuKernelIrContract for PcuRasterKernelIr<'_> {
    fn id(&self) -> PcuKernelId {
        self.id
    }

    fn kind(&self) -> PcuIrKind {
        PcuIrKind::Render
    }

    fn entry_point(&self) -> &str {
        self.entry_point
    }

    fn signature(&self) -> PcuKernelSignature<'_> {
        PcuKernelSignature {
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            invocation: PcuInvocationModel::single(),
        }
    }
}

/// Mesh-family render kernel descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMeshKernelIr<'a> {
    pub id: PcuKernelId,
    pub entry_point: &'a str,
    pub bindings: &'a [PcuBinding<'a>],
    pub ports: &'a [PcuPort<'a>],
    pub parameters: &'a [PcuParameter<'a>],
    pub task_entry: Option<&'a str>,
    pub mesh_entry: &'a str,
    pub fragment_entry: Option<&'a str>,
    pub type_caps: PcuValueTypeCaps,
    pub features: PcuMeshFeatureCaps,
}

impl PcuMeshKernelIr<'_> {
    #[must_use]
    pub const fn required_dispatch_policy(&self) -> PcuDispatchPolicyCaps {
        PcuDispatchPolicyCaps::ORDERED_SUBMISSION
    }

    #[must_use]
    pub const fn required_type_support(&self) -> PcuValueTypeCaps {
        self.type_caps
    }

    #[must_use]
    pub const fn required_feature_support(&self) -> PcuMeshFeatureCaps {
        self.features
    }
}

impl PcuKernelIrContract for PcuMeshKernelIr<'_> {
    fn id(&self) -> PcuKernelId {
        self.id
    }

    fn kind(&self) -> PcuIrKind {
        PcuIrKind::Render
    }

    fn entry_point(&self) -> &str {
        self.entry_point
    }

    fn signature(&self) -> PcuKernelSignature<'_> {
        PcuKernelSignature {
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            invocation: PcuInvocationModel::single(),
        }
    }
}

/// Ray-trace-family render kernel descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuRayTraceKernelIr<'a> {
    pub id: PcuKernelId,
    pub entry_point: &'a str,
    pub bindings: &'a [PcuBinding<'a>],
    pub ports: &'a [PcuPort<'a>],
    pub parameters: &'a [PcuParameter<'a>],
    pub raygen_entry: &'a str,
    pub miss_entries: &'a [&'a str],
    pub callable_entries: &'a [&'a str],
    pub type_caps: PcuValueTypeCaps,
    pub features: PcuRayTraceFeatureCaps,
}

impl PcuRayTraceKernelIr<'_> {
    #[must_use]
    pub const fn required_dispatch_policy(&self) -> PcuDispatchPolicyCaps {
        PcuDispatchPolicyCaps::ORDERED_SUBMISSION
    }

    #[must_use]
    pub const fn required_type_support(&self) -> PcuValueTypeCaps {
        self.type_caps
    }

    #[must_use]
    pub const fn required_feature_support(&self) -> PcuRayTraceFeatureCaps {
        self.features
    }
}

impl PcuKernelIrContract for PcuRayTraceKernelIr<'_> {
    fn id(&self) -> PcuKernelId {
        self.id
    }

    fn kind(&self) -> PcuIrKind {
        PcuIrKind::Render
    }

    fn entry_point(&self) -> &str {
        self.entry_point
    }

    fn signature(&self) -> PcuKernelSignature<'_> {
        PcuKernelSignature {
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            invocation: PcuInvocationModel::single(),
        }
    }
}

/// Closed render-family kernel payload carried by generic PCU submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuRenderKernel<'a> {
    Raster(PcuRasterKernelIr<'a>),
    Mesh(PcuMeshKernelIr<'a>),
    RayTrace(PcuRayTraceKernelIr<'a>),
}

impl<'a> PcuRenderKernel<'a> {
    #[must_use]
    pub const fn required_family_support(self) -> PcuRenderFamilyCaps {
        match self {
            Self::Raster(_) => PcuRenderFamilyCaps::RASTER,
            Self::Mesh(_) => PcuRenderFamilyCaps::MESH,
            Self::RayTrace(_) => PcuRenderFamilyCaps::RAY_TRACE,
        }
    }

    #[must_use]
    pub const fn required_dispatch_policy(self) -> PcuDispatchPolicyCaps {
        match self {
            Self::Raster(kernel) => kernel.required_dispatch_policy(),
            Self::Mesh(kernel) => kernel.required_dispatch_policy(),
            Self::RayTrace(kernel) => kernel.required_dispatch_policy(),
        }
    }

    #[must_use]
    pub const fn required_type_support(self) -> PcuValueTypeCaps {
        match self {
            Self::Raster(kernel) => kernel.required_type_support(),
            Self::Mesh(kernel) => kernel.required_type_support(),
            Self::RayTrace(kernel) => kernel.required_type_support(),
        }
    }

    #[must_use]
    pub const fn as_raster(self) -> Option<PcuRasterKernelIr<'a>> {
        match self {
            Self::Raster(kernel) => Some(kernel),
            Self::Mesh(_) | Self::RayTrace(_) => None,
        }
    }

    #[must_use]
    pub const fn as_mesh(self) -> Option<PcuMeshKernelIr<'a>> {
        match self {
            Self::Mesh(kernel) => Some(kernel),
            Self::Raster(_) | Self::RayTrace(_) => None,
        }
    }

    #[must_use]
    pub const fn as_ray_trace(self) -> Option<PcuRayTraceKernelIr<'a>> {
        match self {
            Self::RayTrace(kernel) => Some(kernel),
            Self::Raster(_) | Self::Mesh(_) => None,
        }
    }
}

impl PcuKernelIrContract for PcuRenderKernel<'_> {
    fn id(&self) -> PcuKernelId {
        match self {
            Self::Raster(kernel) => kernel.id(),
            Self::Mesh(kernel) => kernel.id(),
            Self::RayTrace(kernel) => kernel.id(),
        }
    }

    fn kind(&self) -> PcuIrKind {
        PcuIrKind::Render
    }

    fn entry_point(&self) -> &str {
        match self {
            Self::Raster(kernel) => kernel.entry_point(),
            Self::Mesh(kernel) => kernel.entry_point(),
            Self::RayTrace(kernel) => kernel.entry_point(),
        }
    }

    fn signature(&self) -> PcuKernelSignature<'_> {
        match self {
            Self::Raster(kernel) => kernel.signature(),
            Self::Mesh(kernel) => kernel.signature(),
            Self::RayTrace(kernel) => kernel.signature(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuMeshFeatureCaps,
        PcuMeshKernelIr,
        PcuRasterFeatureCaps,
        PcuRasterKernelIr,
        PcuRayTraceFeatureCaps,
        PcuRayTraceKernelIr,
        PcuRenderFamilyCaps,
        PcuRenderKernel,
    };
    use crate::{
        PcuIrKind,
        PcuKernelId,
        PcuKernelIrContract,
        PcuValueTypeCaps,
    };

    #[test]
    fn render_kernel_reports_family_and_kind() {
        let raster = PcuRenderKernel::Raster(PcuRasterKernelIr {
            id: PcuKernelId(9),
            entry_point: "raster-main",
            bindings: &[],
            ports: &[],
            parameters: &[],
            vertex_entry: "vs_main",
            fragment_entry: Some("fs_main"),
            type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::VECTOR_VALUES),
            features: PcuRasterFeatureCaps::VERTEX_STAGE
                .union(PcuRasterFeatureCaps::FRAGMENT_STAGE),
        });

        assert_eq!(raster.kind(), PcuIrKind::Render);
        assert_eq!(
            raster.required_family_support(),
            PcuRenderFamilyCaps::RASTER
        );
        assert_eq!(raster.entry_point(), "raster-main");
    }

    #[test]
    fn mesh_and_ray_trace_kernels_keep_their_variant_specific_entries() {
        let mesh = PcuRenderKernel::Mesh(PcuMeshKernelIr {
            id: PcuKernelId(10),
            entry_point: "mesh-main",
            bindings: &[],
            ports: &[],
            parameters: &[],
            task_entry: Some("task_main"),
            mesh_entry: "mesh_stage",
            fragment_entry: None,
            type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::VECTOR_VALUES),
            features: PcuMeshFeatureCaps::TASK_STAGE.union(PcuMeshFeatureCaps::MESH_STAGE),
        });
        let ray = PcuRenderKernel::RayTrace(PcuRayTraceKernelIr {
            id: PcuKernelId(11),
            entry_point: "rt-main",
            bindings: &[],
            ports: &[],
            parameters: &[],
            raygen_entry: "raygen_main",
            miss_entries: &["miss0"],
            callable_entries: &[],
            type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::VECTOR_VALUES),
            features: PcuRayTraceFeatureCaps::MISS_SHADERS,
        });

        assert_eq!(
            mesh.as_mesh().map(|kernel| kernel.mesh_entry),
            Some("mesh_stage")
        );
        assert_eq!(
            ray.as_ray_trace().map(|kernel| kernel.raygen_entry),
            Some("raygen_main")
        );
    }
}
