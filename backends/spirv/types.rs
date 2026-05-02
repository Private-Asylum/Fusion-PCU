//! Public SPIR-V lowering options and descriptor types.

use core::ops::{
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
};

/// SPIR-V capability categories the lowering backend may emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSpirvCapability {
    Shader,
    Matrix,
    Image,
    StorageImage,
    CoordinateDerivative,
    RayTracing,
    RayQuery,
}

/// SPIR-V capability bitset allowed for one lowering run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuSpirvCapabilityCaps(u32);

impl PcuSpirvCapabilityCaps {
    pub const SHADER: Self = Self(1 << 0);
    pub const MATRIX: Self = Self(1 << 1);
    pub const IMAGE: Self = Self(1 << 2);
    pub const STORAGE_IMAGE: Self = Self(1 << 3);
    pub const COORDINATE_DERIVATIVE: Self = Self(1 << 4);
    pub const RAY_TRACING: Self = Self(1 << 5);
    pub const RAY_QUERY: Self = Self(1 << 6);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn shader() -> Self {
        Self::SHADER
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

    #[must_use]
    pub const fn for_capability(capability: PcuSpirvCapability) -> Self {
        match capability {
            PcuSpirvCapability::Shader => Self::SHADER,
            PcuSpirvCapability::Matrix => Self::MATRIX,
            PcuSpirvCapability::Image => Self::IMAGE,
            PcuSpirvCapability::StorageImage => Self::STORAGE_IMAGE,
            PcuSpirvCapability::CoordinateDerivative => Self::COORDINATE_DERIVATIVE,
            PcuSpirvCapability::RayTracing => Self::RAY_TRACING,
            PcuSpirvCapability::RayQuery => Self::RAY_QUERY,
        }
    }

    #[must_use]
    pub const fn supports(self, capability: PcuSpirvCapability) -> bool {
        self.contains(Self::for_capability(capability))
    }
}

impl BitOr for PcuSpirvCapabilityCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuSpirvCapabilityCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuSpirvCapabilityCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuSpirvCapabilityCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// SPIR-V version encoded into the module header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvVersion(pub u32);

impl PcuSpirvVersion {
    pub const V1_0: Self = Self(0x0001_0000);
    pub const V1_3: Self = Self(0x0001_0300);
    pub const V1_5: Self = Self(0x0001_0500);
    pub const V1_6: Self = Self(0x0001_0600);
}

/// Options for one SPIR-V lowering run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvLoweringOptions {
    pub version: PcuSpirvVersion,
    pub generator: u32,
    pub capabilities: PcuSpirvCapabilityCaps,
}

impl PcuSpirvLoweringOptions {
    #[must_use]
    pub const fn minimal_shader() -> Self {
        Self {
            version: PcuSpirvVersion::V1_0,
            generator: 0,
            capabilities: PcuSpirvCapabilityCaps::SHADER,
        }
    }

    #[must_use]
    pub const fn with_capabilities(mut self, capabilities: PcuSpirvCapabilityCaps) -> Self {
        self.capabilities = capabilities;
        self
    }

    #[must_use]
    pub const fn with_version(mut self, version: PcuSpirvVersion) -> Self {
        self.version = version;
        self
    }

    #[must_use]
    pub const fn with_generator(mut self, generator: u32) -> Self {
        self.generator = generator;
        self
    }
}

impl Default for PcuSpirvLoweringOptions {
    fn default() -> Self {
        Self::minimal_shader()
    }
}

/// Summary of one emitted SPIR-V module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvModuleInfo {
    pub version: PcuSpirvVersion,
    pub bound: u32,
    pub word_count: usize,
    pub capabilities: PcuSpirvCapabilityCaps,
}
