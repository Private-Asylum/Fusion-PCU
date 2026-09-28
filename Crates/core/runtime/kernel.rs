//! Shared cold kernel preparation errors.

/// Cold frontend construction or backend preparation failure.
#[derive(Debug)]
pub enum PcuKernelPrepareError<I, E> {
    Ir(I),
    Backend(E),
}

impl<I: core::fmt::Display, E: core::fmt::Display> core::fmt::Display
    for PcuKernelPrepareError<I, E>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Ir(error) => write!(f, "PCU kernel construction failed: {error}"),
            Self::Backend(error) => write!(f, "PCU kernel preparation failed: {error}"),
        }
    }
}

impl<I: core::error::Error + 'static, E: core::error::Error + 'static> core::error::Error
    for PcuKernelPrepareError<I, E>
{
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Ir(error) => Some(error),
            Self::Backend(error) => Some(error),
        }
    }
}
