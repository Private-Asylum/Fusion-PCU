//! Transaction-model vocabulary and backend-neutral kernel builder.

use crate::{
    PcuBinding,
    PcuDispatchPolicyCaps,
    PcuKernel,
    PcuKernelIrContract,
    PcuKernelId,
    PcuKernelSignature,
    PcuInvocationModel,
    PcuIrKind,
    PcuParameter,
    PcuPort,
    PcuTransactionFeatureCaps,
};

/// Atomicity contract for one transaction kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuTransactionAtomicity {
    BestEffort,
    Atomic,
}

/// Exclusivity contract for one transaction kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuTransactionExclusivity {
    Shared,
    Exclusive,
}

/// Ordering contract for one transaction family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuTransactionOrdering {
    InOrder,
    Unordered,
}

/// Minimal semantic transaction-profile IR payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuTransactionKernelIr<'a> {
    pub id: PcuKernelId,
    pub entry_point: &'a str,
    pub bindings: &'a [PcuBinding<'a>],
    pub ports: &'a [PcuPort<'a>],
    pub parameters: &'a [PcuParameter<'a>],
    pub timeout_ticks: Option<u32>,
    pub atomicity: PcuTransactionAtomicity,
    pub exclusivity: PcuTransactionExclusivity,
    pub ordering: PcuTransactionOrdering,
    pub idempotent: bool,
}

impl PcuKernelIrContract for PcuTransactionKernelIr<'_> {
    fn id(&self) -> PcuKernelId {
        self.id
    }

    fn kind(&self) -> PcuIrKind {
        PcuIrKind::Transaction
    }

    fn entry_point(&self) -> &str {
        self.entry_point
    }

    fn signature(&self) -> PcuKernelSignature<'_> {
        PcuKernelSignature {
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            invocation: PcuInvocationModel::transaction(),
        }
    }
}

impl PcuTransactionKernelIr<'_> {
    #[must_use]
    pub const fn required_dispatch_policy(&self) -> PcuDispatchPolicyCaps {
        match self.ordering {
            PcuTransactionOrdering::InOrder => PcuDispatchPolicyCaps::ORDERED_SUBMISSION,
            PcuTransactionOrdering::Unordered => PcuDispatchPolicyCaps::empty(),
        }
    }

    #[must_use]
    pub const fn required_features(&self) -> PcuTransactionFeatureCaps {
        let mut features = PcuTransactionFeatureCaps::empty();
        if self.timeout_ticks.is_some() {
            features = features.union(PcuTransactionFeatureCaps::TIMEOUT);
        }
        if matches!(self.atomicity, PcuTransactionAtomicity::Atomic) {
            features = features.union(PcuTransactionFeatureCaps::ATOMICITY);
        }
        if matches!(self.exclusivity, PcuTransactionExclusivity::Exclusive) {
            features = features.union(PcuTransactionFeatureCaps::EXCLUSIVITY);
        }
        if matches!(self.ordering, PcuTransactionOrdering::Unordered) {
            features = features.union(PcuTransactionFeatureCaps::UNORDERED);
        }
        if self.idempotent {
            features = features.union(PcuTransactionFeatureCaps::IDEMPOTENT);
        }
        features
    }
}

/// Builder for one backend-neutral transaction kernel.
#[derive(Debug, Clone, Copy)]
pub struct PcuTransactionKernelBuilder<'a> {
    kernel_id: PcuKernelId,
    entry_point: &'a str,
    bindings: &'a [PcuBinding<'a>],
    ports: &'a [PcuPort<'a>],
    parameters: &'a [PcuParameter<'a>],
    timeout_ticks: Option<u32>,
    atomicity: PcuTransactionAtomicity,
    exclusivity: PcuTransactionExclusivity,
    ordering: PcuTransactionOrdering,
    idempotent: bool,
}

impl<'a> PcuTransactionKernelBuilder<'a> {
    /// Creates one transaction-kernel builder.
    #[must_use]
    pub fn new(kernel_id: u32, entry_point: &'a str) -> Self {
        Self {
            kernel_id: PcuKernelId(kernel_id),
            entry_point,
            bindings: &[],
            ports: &[],
            parameters: &[],
            timeout_ticks: None,
            atomicity: PcuTransactionAtomicity::BestEffort,
            exclusivity: PcuTransactionExclusivity::Shared,
            ordering: PcuTransactionOrdering::InOrder,
            idempotent: false,
        }
    }

    /// Replaces the binding slice.
    #[must_use]
    pub const fn with_bindings(mut self, bindings: &'a [PcuBinding<'a>]) -> Self {
        self.bindings = bindings;
        self
    }

    /// Replaces the port slice.
    #[must_use]
    pub const fn with_ports(mut self, ports: &'a [PcuPort<'a>]) -> Self {
        self.ports = ports;
        self
    }

    /// Replaces the parameter slice.
    #[must_use]
    pub const fn with_parameters(mut self, parameters: &'a [PcuParameter<'a>]) -> Self {
        self.parameters = parameters;
        self
    }

    /// Sets the honest timeout budget.
    #[must_use]
    pub const fn with_timeout_ticks(mut self, timeout_ticks: u32) -> Self {
        self.timeout_ticks = Some(timeout_ticks);
        self
    }

    /// Sets the atomicity contract.
    #[must_use]
    pub const fn with_atomicity(mut self, atomicity: PcuTransactionAtomicity) -> Self {
        self.atomicity = atomicity;
        self
    }

    /// Sets the exclusivity contract.
    #[must_use]
    pub const fn with_exclusivity(mut self, exclusivity: PcuTransactionExclusivity) -> Self {
        self.exclusivity = exclusivity;
        self
    }

    /// Sets the ordering contract.
    #[must_use]
    pub const fn with_ordering(mut self, ordering: PcuTransactionOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Sets whether the transaction is idempotent.
    #[must_use]
    pub const fn with_idempotent(mut self, idempotent: bool) -> Self {
        self.idempotent = idempotent;
        self
    }

    /// Builds the transaction-kernel IR payload.
    #[must_use]
    pub fn ir(&self) -> PcuTransactionKernelIr<'_> {
        PcuTransactionKernelIr {
            id: self.kernel_id,
            entry_point: self.entry_point,
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            timeout_ticks: self.timeout_ticks,
            atomicity: self.atomicity,
            exclusivity: self.exclusivity,
            ordering: self.ordering,
            idempotent: self.idempotent,
        }
    }

    /// Builds the generic kernel wrapper.
    #[must_use]
    pub fn kernel(&self) -> PcuKernel<'_> {
        PcuKernel::Transaction(self.ir())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuTransactionKernelBuilder,
        PcuTransactionOrdering,
    };
    use crate::{
        PcuDispatchPolicyCaps,
        PcuIrKind,
        PcuKernel,
        PcuKernelIrContract,
        PcuTransactionAtomicity,
    };

    #[test]
    fn builder_synthesizes_transaction_kernel() {
        let builder = PcuTransactionKernelBuilder::new(0x44, "transfer")
            .with_timeout_ticks(100)
            .with_atomicity(PcuTransactionAtomicity::Atomic)
            .with_idempotent(true);
        let kernel = builder.ir();

        assert_eq!(kernel.id.0, 0x44);
        assert_eq!(kernel.kind(), PcuIrKind::Transaction);
        assert_eq!(kernel.timeout_ticks, Some(100));
        assert_eq!(kernel.atomicity, PcuTransactionAtomicity::Atomic);
        assert!(kernel.idempotent);
    }

    #[test]
    fn transaction_dispatch_policy_tracks_ordering_contract() {
        let ordered_builder = PcuTransactionKernelBuilder::new(1, "ordered")
            .with_ordering(PcuTransactionOrdering::InOrder);
        let unordered_builder = PcuTransactionKernelBuilder::new(2, "unordered")
            .with_ordering(PcuTransactionOrdering::Unordered);
        let ordered = ordered_builder.ir();
        let unordered = unordered_builder.ir();

        assert_eq!(
            ordered.required_dispatch_policy(),
            PcuDispatchPolicyCaps::ORDERED_SUBMISSION
        );
        assert_eq!(
            unordered.required_dispatch_policy(),
            PcuDispatchPolicyCaps::empty()
        );
    }

    #[test]
    fn builder_wraps_generic_transaction_kernel() {
        let builder = PcuTransactionKernelBuilder::new(12, "probe");
        let kernel = builder.kernel();

        match kernel {
            PcuKernel::Transaction(transaction) => {
                assert_eq!(transaction.kind(), PcuIrKind::Transaction);
                assert_eq!(transaction.id.0, 12);
            }
            _ => panic!("expected transaction kernel"),
        }
    }
}
