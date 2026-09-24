//! Execution-model families built on the PCU core.

pub mod command;
pub mod dispatch;
pub mod signal;
pub mod stream;
pub mod transaction;

use crate::{
    PcuDispatchPolicyCaps,
    PcuPrimitiveCaps,
    PcuKernelId,
    PcuKernelIrContract,
    PcuKernelSignature,
};

pub use command::*;
pub use dispatch::*;
pub use signal::*;
pub use stream::*;
pub use transaction::*;

/// One generic semantic PCU program-unit payload composed from one model over the shared core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuKernel<'a> {
    Dispatch(PcuDispatchKernelIr<'a>),
    Stream(PcuStreamKernelIr<'a>),
    Command(PcuCommandKernelIr<'a>),
    Transaction(PcuTransactionKernelIr<'a>),
    Signal(PcuSignalKernelIr<'a>),
}

impl<'a> PcuKernel<'a> {
    #[must_use]
    pub const fn required_primitive_support(self) -> PcuPrimitiveCaps {
        match self {
            Self::Dispatch(_) => PcuPrimitiveCaps::DISPATCH,
            Self::Stream(_) => PcuPrimitiveCaps::STREAM,
            Self::Command(_) => PcuPrimitiveCaps::COMMAND,
            Self::Transaction(_) => PcuPrimitiveCaps::TRANSACTION,
            Self::Signal(_) => PcuPrimitiveCaps::SIGNAL,
        }
    }

    #[must_use]
    pub const fn required_dispatch_policy(self) -> PcuDispatchPolicyCaps {
        match self {
            Self::Dispatch(kernel) => kernel.required_dispatch_policy(),
            Self::Stream(kernel) => kernel.required_dispatch_policy(),
            Self::Command(kernel) => kernel.required_dispatch_policy(),
            Self::Transaction(kernel) => kernel.required_dispatch_policy(),
            Self::Signal(kernel) => kernel.required_dispatch_policy(),
        }
    }

    #[must_use]
    pub const fn as_stream(self) -> Option<PcuStreamKernelIr<'a>> {
        match self {
            Self::Dispatch(_) | Self::Command(_) | Self::Transaction(_) | Self::Signal(_) => None,
            Self::Stream(kernel) => Some(kernel),
        }
    }

    #[must_use]
    pub const fn as_dispatch(self) -> Option<PcuDispatchKernelIr<'a>> {
        match self {
            Self::Dispatch(kernel) => Some(kernel),
            Self::Stream(_) | Self::Command(_) | Self::Transaction(_) | Self::Signal(_) => None,
        }
    }

    #[must_use]
    pub const fn as_command(self) -> Option<PcuCommandKernelIr<'a>> {
        match self {
            Self::Command(kernel) => Some(kernel),
            Self::Dispatch(_) | Self::Stream(_) | Self::Transaction(_) | Self::Signal(_) => None,
        }
    }

    #[must_use]
    pub const fn as_transaction(self) -> Option<PcuTransactionKernelIr<'a>> {
        match self {
            Self::Transaction(kernel) => Some(kernel),
            Self::Dispatch(_) | Self::Stream(_) | Self::Command(_) | Self::Signal(_) => None,
        }
    }

    #[must_use]
    pub const fn as_signal(self) -> Option<PcuSignalKernelIr<'a>> {
        match self {
            Self::Signal(kernel) => Some(kernel),
            Self::Dispatch(_) | Self::Stream(_) | Self::Command(_) | Self::Transaction(_) => None,
        }
    }
}

impl PcuKernelIrContract for PcuKernel<'_> {
    fn id(&self) -> PcuKernelId {
        match self {
            Self::Dispatch(kernel) => kernel.id(),
            Self::Stream(kernel) => kernel.id(),
            Self::Command(kernel) => kernel.id(),
            Self::Transaction(kernel) => kernel.id(),
            Self::Signal(kernel) => kernel.id(),
        }
    }

    fn kind(&self) -> crate::PcuIrKind {
        match self {
            Self::Dispatch(kernel) => kernel.kind(),
            Self::Stream(kernel) => kernel.kind(),
            Self::Command(kernel) => kernel.kind(),
            Self::Transaction(kernel) => kernel.kind(),
            Self::Signal(kernel) => kernel.kind(),
        }
    }

    fn entry_point(&self) -> &str {
        match self {
            Self::Dispatch(kernel) => kernel.entry_point(),
            Self::Stream(kernel) => kernel.entry_point(),
            Self::Command(kernel) => kernel.entry_point(),
            Self::Transaction(kernel) => kernel.entry_point(),
            Self::Signal(kernel) => kernel.entry_point(),
        }
    }

    fn signature(&self) -> PcuKernelSignature<'_> {
        match self {
            Self::Dispatch(kernel) => kernel.signature(),
            Self::Stream(kernel) => kernel.signature(),
            Self::Command(kernel) => kernel.signature(),
            Self::Transaction(kernel) => kernel.signature(),
            Self::Signal(kernel) => kernel.signature(),
        }
    }
}
