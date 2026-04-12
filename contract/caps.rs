//! Capability and support vocabulary for generic PCU backends.

use core::ops::{
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
};

use crate::{
    PcuKernel,
    PcuStreamCapabilities,
};

/// Indicates whether one surfaced capability is native, synthesized, or unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuImplementationKind {
    Native,
    Emulated,
    Unsupported,
}

/// Generic PCU features the backend can honestly surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuCaps(u32);

impl PcuCaps {
    pub const ENUMERATE_EXECUTORS: Self = Self(1 << 0);
    pub const CLAIM_EXECUTOR: Self = Self(1 << 1);
    pub const DISPATCH: Self = Self(1 << 2);
    pub const COMPLETION_STATUS: Self = Self(1 << 3);
    pub const EXTERNAL_RESOURCES: Self = Self(1 << 4);
    pub const COMPUTE_DISPATCH: Self = Self(1 << 5);
    pub const DEVICE_LOCAL_MEMORY: Self = Self(1 << 6);

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

impl BitOr for PcuCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Primitive model families the backend may expose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuPrimitiveCaps(u32);

impl PcuPrimitiveCaps {
    pub const DISPATCH: Self = Self(1 << 0);
    pub const STREAM: Self = Self(1 << 1);
    pub const COMMAND: Self = Self(1 << 2);
    pub const TRANSACTION: Self = Self(1 << 3);
    pub const SIGNAL: Self = Self(1 << 4);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn all() -> Self {
        Self::DISPATCH
            .union(Self::STREAM)
            .union(Self::COMMAND)
            .union(Self::TRANSACTION)
            .union(Self::SIGNAL)
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

impl BitOr for PcuPrimitiveCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuPrimitiveCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuPrimitiveCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuPrimitiveCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Dispatch policy/allowance flags surfaced by one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuDispatchPolicyCaps(u32);

impl PcuDispatchPolicyCaps {
    pub const SERIAL: Self = Self(1 << 0);
    pub const PIPELINED: Self = Self(1 << 1);
    pub const PARALLEL: Self = Self(1 << 2);
    pub const PERSISTENT_INSTALL: Self = Self(1 << 3);
    pub const CPU_FALLBACK: Self = Self(1 << 4);
    pub const MIXED_EXECUTION: Self = Self(1 << 5);
    pub const ORDERED_SUBMISSION: Self = Self(1 << 6);
    pub const OUT_OF_ORDER_SUBMISSION: Self = Self(1 << 7);

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

impl BitOr for PcuDispatchPolicyCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuDispatchPolicyCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuDispatchPolicyCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuDispatchPolicyCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Per-op support flags for the dispatch model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuDispatchOpCaps(u64);

impl PcuDispatchOpCaps {
    pub const VALUE_CONSTANT: Self = Self(1 << 0);
    pub const VALUE_CAST: Self = Self(1 << 1);
    pub const VALUE_PACK: Self = Self(1 << 2);
    pub const VALUE_UNPACK: Self = Self(1 << 3);
    pub const VALUE_SWIZZLE: Self = Self(1 << 4);
    pub const ALU_ADD: Self = Self(1 << 5);
    pub const ALU_SUB: Self = Self(1 << 6);
    pub const ALU_MUL: Self = Self(1 << 7);
    pub const ALU_DIV: Self = Self(1 << 8);
    pub const ALU_MIN: Self = Self(1 << 9);
    pub const ALU_MAX: Self = Self(1 << 10);
    pub const ALU_AND: Self = Self(1 << 11);
    pub const ALU_OR: Self = Self(1 << 12);
    pub const ALU_XOR: Self = Self(1 << 13);
    pub const ALU_SHIFT_LEFT: Self = Self(1 << 14);
    pub const ALU_SHIFT_RIGHT: Self = Self(1 << 15);
    pub const ALU_COMPARE: Self = Self(1 << 16);
    pub const ALU_SELECT: Self = Self(1 << 17);
    pub const CONTROL_BRANCH: Self = Self(1 << 18);
    pub const CONTROL_LOOP: Self = Self(1 << 19);
    pub const CONTROL_RETURN: Self = Self(1 << 20);
    pub const BINDING_LOAD: Self = Self(1 << 21);
    pub const BINDING_STORE: Self = Self(1 << 22);
    pub const BINDING_ATOMIC: Self = Self(1 << 23);
    pub const BINDING_SAMPLE: Self = Self(1 << 24);
    pub const PORT_RECEIVE: Self = Self(1 << 25);
    pub const PORT_SEND: Self = Self(1 << 26);
    pub const PORT_PEEK: Self = Self(1 << 27);
    pub const PORT_DISCARD: Self = Self(1 << 28);
    pub const SYNC_BARRIER: Self = Self(1 << 29);
    pub const SYNC_FENCE: Self = Self(1 << 30);
    pub const INTRINSIC: Self = Self(1 << 31);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn all() -> Self {
        Self((1u64 << 32) - 1)
    }

    #[must_use]
    pub const fn bits(self) -> u64 {
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

impl BitOr for PcuDispatchOpCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuDispatchOpCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuDispatchOpCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuDispatchOpCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Per-op support flags for the command model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuCommandOpCaps(u32);

impl PcuCommandOpCaps {
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const MODIFY: Self = Self(1 << 2);
    pub const COPY: Self = Self(1 << 3);
    pub const INVOKE: Self = Self(1 << 4);
    pub const AWAIT: Self = Self(1 << 5);
    pub const STALL: Self = Self(1 << 6);
    pub const SLEEP: Self = Self(1 << 7);
    pub const BARRIER: Self = Self(1 << 8);
    pub const RETURN: Self = Self(1 << 9);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn all() -> Self {
        Self((1u32 << 10) - 1)
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

impl BitOr for PcuCommandOpCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuCommandOpCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuCommandOpCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuCommandOpCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Feature support flags for the opaque transaction model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuTransactionFeatureCaps(u32);

impl PcuTransactionFeatureCaps {
    pub const TIMEOUT: Self = Self(1 << 0);
    pub const ATOMICITY: Self = Self(1 << 1);
    pub const EXCLUSIVITY: Self = Self(1 << 2);
    pub const UNORDERED: Self = Self(1 << 3);
    pub const IDEMPOTENT: Self = Self(1 << 4);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn all() -> Self {
        Self((1u32 << 5) - 1)
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

impl BitOr for PcuTransactionFeatureCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuTransactionFeatureCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuTransactionFeatureCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuTransactionFeatureCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Per-op support flags for the signal model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuSignalOpCaps(u32);

impl PcuSignalOpCaps {
    pub const ACK: Self = Self(1 << 0);
    pub const READ: Self = Self(1 << 1);
    pub const WRITE: Self = Self(1 << 2);
    pub const PUBLISH: Self = Self(1 << 3);
    pub const NOTIFY: Self = Self(1 << 4);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn all() -> Self {
        Self((1u32 << 5) - 1)
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

impl BitOr for PcuSignalOpCaps {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PcuSignalOpCaps {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PcuSignalOpCaps {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PcuSignalOpCaps {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// Dual-surface support reporting for one feature family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuFeatureSupport<T> {
    pub direct: T,
    pub cpu_fallback: T,
}

impl<T: Copy> PcuFeatureSupport<T> {
    #[must_use]
    pub const fn new(direct: T, cpu_fallback: T) -> Self {
        Self {
            direct,
            cpu_fallback,
        }
    }
}

/// Primitive-family support surfaced by one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuPrimitiveSupport {
    pub primitives: PcuFeatureSupport<PcuPrimitiveCaps>,
}

impl PcuPrimitiveSupport {
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            primitives: PcuFeatureSupport::new(
                PcuPrimitiveCaps::empty(),
                PcuPrimitiveCaps::empty(),
            ),
        }
    }

    #[must_use]
    pub const fn supports_direct(self, required: PcuPrimitiveCaps) -> bool {
        self.primitives.direct.contains(required)
    }

    #[must_use]
    pub const fn supports_cpu_fallback(self, required: PcuPrimitiveCaps) -> bool {
        self.primitives.cpu_fallback.contains(required)
    }
}

/// Dispatch-model support surfaced by one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchSupport {
    pub flags: PcuDispatchPolicyCaps,
    pub instructions: PcuFeatureSupport<PcuDispatchOpCaps>,
}

impl PcuDispatchSupport {
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            flags: PcuDispatchPolicyCaps::empty(),
            instructions: PcuFeatureSupport::new(
                PcuDispatchOpCaps::empty(),
                PcuDispatchOpCaps::empty(),
            ),
        }
    }

    #[must_use]
    pub const fn allows(self, flags: PcuDispatchPolicyCaps) -> bool {
        self.flags.contains(flags)
    }

    #[must_use]
    pub const fn supports_direct_instructions(self, required: PcuDispatchOpCaps) -> bool {
        self.instructions.direct.contains(required)
    }

    #[must_use]
    pub const fn supports_cpu_fallback_instructions(self, required: PcuDispatchOpCaps) -> bool {
        self.instructions.cpu_fallback.contains(required)
    }
}

/// Command-model support surfaced by one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuCommandSupport {
    pub instructions: PcuFeatureSupport<PcuCommandOpCaps>,
}

impl PcuCommandSupport {
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            instructions: PcuFeatureSupport::new(
                PcuCommandOpCaps::empty(),
                PcuCommandOpCaps::empty(),
            ),
        }
    }

    #[must_use]
    pub const fn supports_direct_instructions(self, required: PcuCommandOpCaps) -> bool {
        self.instructions.direct.contains(required)
    }

    #[must_use]
    pub const fn supports_cpu_fallback_instructions(self, required: PcuCommandOpCaps) -> bool {
        self.instructions.cpu_fallback.contains(required)
    }
}

/// Stream-model support surfaced by one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuStreamSupport {
    pub instructions: PcuFeatureSupport<PcuStreamCapabilities>,
}

impl PcuStreamSupport {
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            instructions: PcuFeatureSupport::new(
                PcuStreamCapabilities::empty(),
                PcuStreamCapabilities::empty(),
            ),
        }
    }

    #[must_use]
    pub const fn supports_direct_instructions(self, required: PcuStreamCapabilities) -> bool {
        self.instructions.direct.contains(required)
    }

    #[must_use]
    pub const fn supports_cpu_fallback_instructions(self, required: PcuStreamCapabilities) -> bool {
        self.instructions.cpu_fallback.contains(required)
    }
}

/// Transaction-model support surfaced by one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuTransactionSupport {
    pub features: PcuFeatureSupport<PcuTransactionFeatureCaps>,
}

impl PcuTransactionSupport {
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            features: PcuFeatureSupport::new(
                PcuTransactionFeatureCaps::empty(),
                PcuTransactionFeatureCaps::empty(),
            ),
        }
    }

    #[must_use]
    pub const fn supports_direct_features(self, required: PcuTransactionFeatureCaps) -> bool {
        self.features.direct.contains(required)
    }

    #[must_use]
    pub const fn supports_cpu_fallback_features(self, required: PcuTransactionFeatureCaps) -> bool {
        self.features.cpu_fallback.contains(required)
    }
}

/// Signal-model support surfaced by one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSignalSupport {
    pub instructions: PcuFeatureSupport<PcuSignalOpCaps>,
}

impl PcuSignalSupport {
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            instructions: PcuFeatureSupport::new(
                PcuSignalOpCaps::empty(),
                PcuSignalOpCaps::empty(),
            ),
        }
    }

    #[must_use]
    pub const fn supports_direct_instructions(self, required: PcuSignalOpCaps) -> bool {
        self.instructions.direct.contains(required)
    }

    #[must_use]
    pub const fn supports_cpu_fallback_instructions(self, required: PcuSignalOpCaps) -> bool {
        self.instructions.cpu_fallback.contains(required)
    }
}

/// Full capability surface for one generic PCU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSupport {
    /// Backend-supported generic PCU features.
    pub caps: PcuCaps,
    /// Native, lowered-with-restrictions, or unsupported implementation category.
    pub implementation: PcuImplementationKind,
    /// Number of surfaced PCU execution substrates.
    pub executor_count: u8,
    /// Primitive-family support.
    pub primitive_support: PcuPrimitiveSupport,
    /// Dispatch-model support.
    pub dispatch_support: PcuDispatchSupport,
    /// Stream-model support.
    pub stream_support: PcuStreamSupport,
    /// Command-model support.
    pub command_support: PcuCommandSupport,
    /// Transaction-model support.
    pub transaction_support: PcuTransactionSupport,
    /// Signal-model support.
    pub signal_support: PcuSignalSupport,
}

impl PcuSupport {
    /// Returns a fully unsupported generic PCU surface.
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            caps: PcuCaps::empty(),
            implementation: PcuImplementationKind::Unsupported,
            executor_count: 0,
            primitive_support: PcuPrimitiveSupport::unsupported(),
            dispatch_support: PcuDispatchSupport::unsupported(),
            stream_support: PcuStreamSupport::unsupported(),
            command_support: PcuCommandSupport::unsupported(),
            transaction_support: PcuTransactionSupport::unsupported(),
            signal_support: PcuSignalSupport::unsupported(),
        }
    }

    /// Returns whether this backend can execute the supplied kernel directly.
    #[must_use]
    pub fn supports_kernel_direct(&self, kernel: PcuKernel<'_>) -> bool {
        let required_policy = kernel.required_dispatch_policy();
        match kernel {
            PcuKernel::Dispatch(kernel) => {
                self.primitive_support
                    .supports_direct(PcuPrimitiveCaps::DISPATCH)
                    && self.dispatch_support.allows(required_policy)
                    && self
                        .dispatch_support
                        .supports_direct_instructions(kernel.required_instruction_support())
            }
            PcuKernel::Stream(kernel) => {
                self.primitive_support
                    .supports_direct(PcuPrimitiveCaps::STREAM)
                    && self.dispatch_support.allows(required_policy)
                    && self
                        .stream_support
                        .supports_direct_instructions(kernel.required_instruction_support())
            }
            PcuKernel::Command(kernel) => {
                self.primitive_support
                    .supports_direct(PcuPrimitiveCaps::COMMAND)
                    && self.dispatch_support.allows(required_policy)
                    && self
                        .command_support
                        .supports_direct_instructions(kernel.required_instruction_support())
            }
            PcuKernel::Transaction(kernel) => {
                self.primitive_support
                    .supports_direct(PcuPrimitiveCaps::TRANSACTION)
                    && self.dispatch_support.allows(required_policy)
                    && self
                        .transaction_support
                        .supports_direct_features(kernel.required_features())
            }
            PcuKernel::Signal(kernel) => {
                self.primitive_support
                    .supports_direct(PcuPrimitiveCaps::SIGNAL)
                    && self.dispatch_support.allows(required_policy)
                    && self
                        .signal_support
                        .supports_direct_instructions(kernel.required_instruction_support())
            }
        }
    }

    /// Returns whether this backend can execute the supplied kernel through CPU fallback.
    #[must_use]
    pub fn supports_kernel_cpu_fallback(&self, kernel: PcuKernel<'_>) -> bool {
        let required_policy = kernel.required_dispatch_policy();
        match kernel {
            PcuKernel::Dispatch(kernel) => {
                self.primitive_support
                    .supports_cpu_fallback(PcuPrimitiveCaps::DISPATCH)
                    && self.dispatch_support.allows(required_policy)
                    && self
                        .dispatch_support
                        .supports_cpu_fallback_instructions(kernel.required_instruction_support())
            }
            PcuKernel::Stream(kernel) => {
                self.primitive_support
                    .supports_cpu_fallback(PcuPrimitiveCaps::STREAM)
                    && self.dispatch_support.allows(required_policy)
                    && self
                        .stream_support
                        .supports_cpu_fallback_instructions(kernel.required_instruction_support())
            }
            PcuKernel::Command(kernel) => {
                self.primitive_support
                    .supports_cpu_fallback(PcuPrimitiveCaps::COMMAND)
                    && self.dispatch_support.allows(required_policy)
                    && self
                        .command_support
                        .supports_cpu_fallback_instructions(kernel.required_instruction_support())
            }
            PcuKernel::Transaction(kernel) => {
                self.primitive_support
                    .supports_cpu_fallback(PcuPrimitiveCaps::TRANSACTION)
                    && self.dispatch_support.allows(required_policy)
                    && self
                        .transaction_support
                        .supports_cpu_fallback_features(kernel.required_features())
            }
            PcuKernel::Signal(kernel) => {
                self.primitive_support
                    .supports_cpu_fallback(PcuPrimitiveCaps::SIGNAL)
                    && self.dispatch_support.allows(required_policy)
                    && self
                        .signal_support
                        .supports_cpu_fallback_instructions(kernel.required_instruction_support())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuCommandOpCaps,
        PcuCommandSupport,
        PcuDispatchOpCaps,
        PcuDispatchPolicyCaps,
        PcuDispatchSupport,
        PcuFeatureSupport,
        PcuPrimitiveCaps,
        PcuPrimitiveSupport,
        PcuSupport,
    };
    use crate::{
        PcuCommandKernelIr,
        PcuCommandOp,
        PcuCommandStep,
        PcuKernel,
        PcuKernelId,
        PcuTarget,
    };

    fn command_kernel() -> PcuKernel<'static> {
        PcuKernel::Command(PcuCommandKernelIr {
            id: PcuKernelId(7),
            entry_point: "write-register",
            bindings: &[],
            ports: &[],
            parameters: &[],
            steps: &[PcuCommandStep {
                name: Some("write"),
                op: PcuCommandOp::Write {
                    target: PcuTarget::Named("register"),
                    value: crate::PcuOperand::PreviousResult,
                },
            }],
        })
    }

    #[test]
    fn support_reports_direct_command_coverage() {
        let mut support = PcuSupport::unsupported();
        support.primitive_support = PcuPrimitiveSupport {
            primitives: PcuFeatureSupport::new(
                PcuPrimitiveCaps::COMMAND,
                PcuPrimitiveCaps::COMMAND,
            ),
        };
        support.command_support = PcuCommandSupport {
            instructions: PcuFeatureSupport::new(PcuCommandOpCaps::WRITE, PcuCommandOpCaps::WRITE),
        };
        support.dispatch_support = PcuDispatchSupport {
            flags: PcuDispatchPolicyCaps::ORDERED_SUBMISSION,
            instructions: PcuFeatureSupport::new(
                PcuDispatchOpCaps::empty(),
                PcuDispatchOpCaps::empty(),
            ),
        };

        let kernel = command_kernel();

        assert!(support.supports_kernel_direct(kernel));
        assert!(support.supports_kernel_cpu_fallback(kernel));
    }

    #[test]
    fn support_reports_cpu_fallback_when_direct_support_is_missing() {
        let mut support = PcuSupport::unsupported();
        support.primitive_support = PcuPrimitiveSupport {
            primitives: PcuFeatureSupport::new(
                PcuPrimitiveCaps::COMMAND,
                PcuPrimitiveCaps::COMMAND,
            ),
        };
        support.command_support = PcuCommandSupport {
            instructions: PcuFeatureSupport::new(
                PcuCommandOpCaps::empty(),
                PcuCommandOpCaps::WRITE,
            ),
        };
        support.dispatch_support = PcuDispatchSupport {
            flags: PcuDispatchPolicyCaps::ORDERED_SUBMISSION,
            instructions: PcuFeatureSupport::new(
                PcuDispatchOpCaps::empty(),
                PcuDispatchOpCaps::empty(),
            ),
        };

        let kernel = command_kernel();

        assert!(!support.supports_kernel_direct(kernel));
        assert!(support.supports_kernel_cpu_fallback(kernel));
    }
}
