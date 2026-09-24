//! Fixed-capacity facade for composing statically linked discovery providers.
//!
//! The registry erases provider types only at a monomorphized function-pointer boundary. It
//! allocates nothing and does not define a loadable plugin ABI.

use core::fmt::{
    self,
    Write,
};

use crate::{
    PcuCapabilitySnapshot,
    PcuContextDescriptor,
    PcuDeviceDescriptor,
    PcuExecutorDescriptor,
    PcuMemoryDomainDescriptor,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuRuntimeDiscovery,
    PcuTargetDescriptor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuDiscoveryOperation {
    Register,
    Providers,
    Targets,
    Devices,
    Contexts,
    MemoryDomains,
    TargetCapabilities,
    DeviceCapabilities,
    Executors,
}

/// Bounded, allocation-free diagnostic returned by a registry operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuRegistryError {
    pub provider: Option<PcuProviderId>,
    pub operation: PcuDiscoveryOperation,
    pub message: [u8; 96],
    pub message_len: u8,
    pub truncated: bool,
}

impl PcuRegistryError {
    fn new(
        provider: Option<PcuProviderId>,
        operation: PcuDiscoveryOperation,
        error: impl fmt::Display,
    ) -> Self {
        struct Buffer<'a>(&'a mut PcuRegistryError);
        impl Write for Buffer<'_> {
            fn write_str(&mut self, value: &str) -> fmt::Result {
                let start = self.0.message_len as usize;
                let available = self.0.message.len().saturating_sub(start);
                let n = value.len().min(available);
                self.0.message[start..start + n].copy_from_slice(&value.as_bytes()[..n]);
                self.0.message_len = self.0.message_len.saturating_add(
                    u8::try_from(n).expect("the message buffer is at most 96 bytes"),
                );
                if n != value.len() {
                    self.0.truncated = true;
                }
                Ok(())
            }
        }
        let mut result = Self {
            provider,
            operation,
            message: [0; 96],
            message_len: 0,
            truncated: false,
        };
        let _ = write!(Buffer(&mut result), "{error}");
        result
    }

    #[must_use]
    pub fn message(&self) -> &str {
        // The buffer is populated from UTF-8 `Display` strings and truncation is at a byte
        // boundary. If a multi-byte codepoint is split, expose the valid prefix.
        let bytes = &self.message[..self.message_len as usize];
        core::str::from_utf8(bytes)
            .unwrap_or_else(|e| core::str::from_utf8(&bytes[..e.valid_up_to()]).unwrap_or(""))
    }
}

type Providers = for<'a> fn(
    *const (),
    PcuProviderId,
    &mut [PcuProviderDescriptor<'a>],
) -> Result<usize, PcuRegistryError>;
type Targets = for<'a> fn(
    *const (),
    PcuProviderId,
    u64,
    &mut [PcuTargetDescriptor<'a>],
) -> Result<usize, PcuRegistryError>;
type Devices = for<'a> fn(
    *const (),
    PcuObjectRef,
    &mut [PcuDeviceDescriptor<'a>],
) -> Result<usize, PcuRegistryError>;
type Contexts = for<'a> fn(
    *const (),
    PcuObjectRef,
    &mut [PcuContextDescriptor<'a>],
) -> Result<usize, PcuRegistryError>;
type Domains = for<'a> fn(
    *const (),
    PcuObjectRef,
    &mut [PcuMemoryDomainDescriptor<'a>],
) -> Result<usize, PcuRegistryError>;
type Caps = fn(*const (), PcuObjectRef) -> Result<PcuCapabilitySnapshot, PcuRegistryError>;
type Executors =
    fn(*const (), PcuObjectRef, &mut [PcuExecutorDescriptor]) -> Result<usize, PcuRegistryError>;

#[derive(Clone, Copy)]
struct Vtable {
    providers: Providers,
    targets: Targets,
    devices: Devices,
    contexts: Contexts,
    domains: Domains,
    target_caps: Caps,
    device_caps: Caps,
    executors: Executors,
}
#[derive(Clone, Copy)]
struct Entry<'a> {
    id: PcuProviderId,
    context: *const (),
    _borrow: core::marker::PhantomData<&'a ()>,
    vtable: Vtable,
}

/// A bounded registry over concrete, statically linked discovery implementations.
pub struct PcuRuntimeDiscoveryRegistry<'a, const N: usize> {
    entries: [Option<Entry<'a>>; N],
    len: usize,
}

impl<'a, const N: usize> PcuRuntimeDiscoveryRegistry<'a, N> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: [None; N],
            len: 0,
        }
    }
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Registers one provider under its stable ID. The backend's descriptor IDs must use the
    /// same ID, making all returned object references directly routable and unambiguous.
    ///
    /// # Errors
    ///
    /// Returns a registry error when discovery fails, the provider does not expose exactly one
    /// descriptor, its ID is already registered, or the fixed registry is full.
    pub fn register<D>(&mut self, provider: &'a D) -> Result<PcuProviderId, PcuRegistryError>
    where
        D: PcuRuntimeDiscovery + 'static,
        D::Error: fmt::Display,
    {
        let mut descriptor = [empty_provider()];
        let n = providers::<D>(
            core::ptr::from_ref(provider).cast::<()>(),
            PcuProviderId(u32::MAX),
            &mut descriptor,
        )
        .map_err(|mut e| {
            e.provider = None;
            e.operation = PcuDiscoveryOperation::Register;
            e
        })?;
        if n != 1 {
            return Err(PcuRegistryError::new(
                None,
                PcuDiscoveryOperation::Register,
                "provider must expose exactly one descriptor",
            ));
        }
        let id = descriptor[0].id;
        if self.entries[..self.len]
            .iter()
            .flatten()
            .any(|e| e.id == id)
        {
            return Err(PcuRegistryError::new(
                Some(id),
                PcuDiscoveryOperation::Register,
                "duplicate provider id",
            ));
        }
        if self.len == N {
            return Err(PcuRegistryError::new(
                Some(id),
                PcuDiscoveryOperation::Register,
                "registry capacity exhausted",
            ));
        }
        let entry = Entry {
            id,
            context: core::ptr::from_ref(provider).cast::<()>(),
            _borrow: core::marker::PhantomData,
            vtable: Vtable {
                providers: providers::<D>,
                targets: targets::<D>,
                devices: devices::<D>,
                contexts: contexts::<D>,
                domains: domains::<D>,
                target_caps: target_caps::<D>,
                device_caps: device_caps::<D>,
                executors: executors::<D>,
            },
        };
        // SAFETY: context points to `provider`, which is borrowed for 'a. Every thunk was
        // monomorphized for exactly D and casts this pointer back to D.
        self.entries[self.len] = Some(entry);
        self.len += 1;
        Ok(id)
    }

    /// Enumerates registered providers into caller-owned bounded storage.
    ///
    /// # Errors
    ///
    /// Returns a provider's discovery error.
    pub fn providers<'s>(
        &'s self,
        output: &mut [PcuProviderDescriptor<'s>],
    ) -> Result<usize, PcuRegistryError> {
        let mut total = 0usize;
        let mut written = 0usize;
        for e in self.entries[..self.len].iter().flatten() {
            let remaining = &mut output[written..];
            let capacity = remaining.len();
            let n = (e.vtable.providers)(e.context, e.id, remaining)?;
            written = written.saturating_add(n.min(capacity));
            total = total.saturating_add(n);
        }
        Ok(total)
    }
    /// Enumerates targets for one registered provider and generation.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is unknown or discovery fails.
    pub fn targets<'s>(
        &'s self,
        provider: PcuProviderId,
        generation: u64,
        out: &mut [PcuTargetDescriptor<'s>],
    ) -> Result<usize, PcuRegistryError> {
        let e = self.find(provider)?;
        (e.vtable.targets)(e.context, provider, generation, out)
    }
    /// Enumerates devices for one target.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is unknown or discovery fails.
    pub fn devices<'s>(
        &'s self,
        target: PcuObjectRef,
        out: &mut [PcuDeviceDescriptor<'s>],
    ) -> Result<usize, PcuRegistryError> {
        let e = self.find(target.provider)?;
        (e.vtable.devices)(e.context, target, out)
    }
    /// Enumerates contexts for one device.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is unknown or discovery fails.
    pub fn contexts<'s>(
        &'s self,
        device: PcuObjectRef,
        out: &mut [PcuContextDescriptor<'s>],
    ) -> Result<usize, PcuRegistryError> {
        let e = self.find(device.provider)?;
        (e.vtable.contexts)(e.context, device, out)
    }
    /// Enumerates memory domains for one context.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is unknown or discovery fails.
    pub fn memory_domains<'s>(
        &'s self,
        context: PcuObjectRef,
        out: &mut [PcuMemoryDomainDescriptor<'s>],
    ) -> Result<usize, PcuRegistryError> {
        let e = self.find(context.provider)?;
        (e.vtable.domains)(e.context, context, out)
    }
    /// Queries capabilities for one target.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is unknown or the query fails.
    pub fn target_capabilities(
        &self,
        target: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, PcuRegistryError> {
        let e = self.find(target.provider)?;
        (e.vtable.target_caps)(e.context, target)
    }
    /// Queries capabilities for one device.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is unknown or the query fails.
    pub fn device_capabilities(
        &self,
        device: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, PcuRegistryError> {
        let e = self.find(device.provider)?;
        (e.vtable.device_caps)(e.context, device)
    }
    /// Enumerates executor descriptors for one discovered object.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is unknown or discovery fails.
    pub fn executors(
        &self,
        object: PcuObjectRef,
        out: &mut [PcuExecutorDescriptor],
    ) -> Result<usize, PcuRegistryError> {
        let e = self.find(object.provider)?;
        (e.vtable.executors)(e.context, object, out)
    }
    fn find(&self, id: PcuProviderId) -> Result<&Entry<'a>, PcuRegistryError> {
        self.entries[..self.len]
            .iter()
            .flatten()
            .find(|e| e.id == id)
            .ok_or_else(|| {
                PcuRegistryError::new(
                    Some(id),
                    PcuDiscoveryOperation::Register,
                    "unknown provider id",
                )
            })
    }
}
impl<const N: usize> Default for PcuRuntimeDiscoveryRegistry<'_, N> {
    fn default() -> Self {
        Self::new()
    }
}

fn call_error(
    e: impl fmt::Display,
    p: PcuProviderId,
    op: PcuDiscoveryOperation,
) -> PcuRegistryError {
    PcuRegistryError::new(Some(p), op, e)
}
const unsafe fn cast<'a, D: 'a>(x: *const ()) -> &'a D {
    // SAFETY: Caller guarantees that `x` points to a live `D` for `'a`.
    unsafe { &*x.cast::<D>() }
}

const fn empty_provider<'a>() -> PcuProviderDescriptor<'a> {
    use crate::{
        PcuProviderReadiness,
        PcuProviderStatus,
    };
    PcuProviderDescriptor {
        id: PcuProviderId(u32::MAX),
        generation: 0,
        backend: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }
}

fn providers<D: PcuRuntimeDiscovery + 'static>(
    x: *const (),
    id: PcuProviderId,
    o: &mut [PcuProviderDescriptor<'_>],
) -> Result<usize, PcuRegistryError>
where
    D::Error: fmt::Display,
{
    unsafe { cast::<D>(x) }
        .providers(o)
        .map_err(|e| call_error(e, id, PcuDiscoveryOperation::Providers))
}
fn targets<D: PcuRuntimeDiscovery + 'static>(
    x: *const (),
    p: PcuProviderId,
    g: u64,
    o: &mut [PcuTargetDescriptor<'_>],
) -> Result<usize, PcuRegistryError>
where
    D::Error: fmt::Display,
{
    unsafe { cast::<D>(x) }
        .targets(p, g, o)
        .map_err(|e| call_error(e, p, PcuDiscoveryOperation::Targets))
}
fn devices<D: PcuRuntimeDiscovery + 'static>(
    x: *const (),
    r: PcuObjectRef,
    o: &mut [PcuDeviceDescriptor<'_>],
) -> Result<usize, PcuRegistryError>
where
    D::Error: fmt::Display,
{
    unsafe { cast::<D>(x) }
        .devices(r, o)
        .map_err(|e| call_error(e, r.provider, PcuDiscoveryOperation::Devices))
}
fn contexts<D: PcuRuntimeDiscovery + 'static>(
    x: *const (),
    r: PcuObjectRef,
    o: &mut [PcuContextDescriptor<'_>],
) -> Result<usize, PcuRegistryError>
where
    D::Error: fmt::Display,
{
    unsafe { cast::<D>(x) }
        .contexts(r, o)
        .map_err(|e| call_error(e, r.provider, PcuDiscoveryOperation::Contexts))
}
fn domains<D: PcuRuntimeDiscovery + 'static>(
    x: *const (),
    r: PcuObjectRef,
    o: &mut [PcuMemoryDomainDescriptor<'_>],
) -> Result<usize, PcuRegistryError>
where
    D::Error: fmt::Display,
{
    unsafe { cast::<D>(x) }
        .memory_domains(r, o)
        .map_err(|e| call_error(e, r.provider, PcuDiscoveryOperation::MemoryDomains))
}
fn target_caps<D: PcuRuntimeDiscovery>(
    x: *const (),
    r: PcuObjectRef,
) -> Result<PcuCapabilitySnapshot, PcuRegistryError>
where
    D::Error: fmt::Display,
{
    unsafe { cast::<D>(x) }
        .target_capabilities(r)
        .map_err(|e| call_error(e, r.provider, PcuDiscoveryOperation::TargetCapabilities))
}
fn device_caps<D: PcuRuntimeDiscovery>(
    x: *const (),
    r: PcuObjectRef,
) -> Result<PcuCapabilitySnapshot, PcuRegistryError>
where
    D::Error: fmt::Display,
{
    unsafe { cast::<D>(x) }
        .device_capabilities(r)
        .map_err(|e| call_error(e, r.provider, PcuDiscoveryOperation::DeviceCapabilities))
}
fn executors<D: PcuRuntimeDiscovery>(
    x: *const (),
    r: PcuObjectRef,
    o: &mut [PcuExecutorDescriptor],
) -> Result<usize, PcuRegistryError>
where
    D::Error: fmt::Display,
{
    unsafe { cast::<D>(x) }
        .executors(r, o)
        .map_err(|e| call_error(e, r.provider, PcuDiscoveryOperation::Executors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PcuContextDescriptor,
        PcuDeviceDescriptor,
        PcuMemoryDomainDescriptor,
        PcuProviderReadiness,
        PcuProviderStatus,
        PcuSupport,
        PcuTargetDescriptor,
    };

    struct Mock {
        id: PcuProviderId,
        fail_targets: bool,
    }
    impl PcuRuntimeDiscovery for Mock {
        type Error = &'static str;
        fn providers<'a>(
            &'a self,
            out: &mut [PcuProviderDescriptor<'a>],
        ) -> Result<usize, Self::Error> {
            if let Some(slot) = out.first_mut() {
                *slot = PcuProviderDescriptor {
                    id: self.id,
                    generation: 1,
                    backend: if self.id.0 == 2 { "two" } else { "one" },
                    readiness: PcuProviderReadiness {
                        status: PcuProviderStatus::Ready,
                        reason: None,
                    },
                };
            }
            Ok(1)
        }
        fn targets<'a>(
            &'a self,
            p: PcuProviderId,
            g: u64,
            out: &mut [PcuTargetDescriptor<'a>],
        ) -> Result<usize, Self::Error> {
            if self.fail_targets {
                return Err("probe said no");
            }
            if let Some(slot) = out.first_mut() {
                *slot = PcuTargetDescriptor {
                    reference: PcuObjectRef {
                        provider: p,
                        generation: g,
                        kind: crate::PcuObjectKind::Target,
                        id: 0,
                    },
                    name: "target",
                    readiness: PcuProviderReadiness {
                        status: PcuProviderStatus::Ready,
                        reason: None,
                    },
                };
            }
            Ok(1)
        }
        fn devices<'a>(
            &'a self,
            _: PcuObjectRef,
            _: &mut [PcuDeviceDescriptor<'a>],
        ) -> Result<usize, Self::Error> {
            Ok(0)
        }
        fn contexts<'a>(
            &'a self,
            _: PcuObjectRef,
            _: &mut [PcuContextDescriptor<'a>],
        ) -> Result<usize, Self::Error> {
            Ok(0)
        }
        fn memory_domains<'a>(
            &'a self,
            _: PcuObjectRef,
            _: &mut [PcuMemoryDomainDescriptor<'a>],
        ) -> Result<usize, Self::Error> {
            Ok(0)
        }
        fn target_capabilities(
            &self,
            _: PcuObjectRef,
        ) -> Result<PcuCapabilitySnapshot, Self::Error> {
            Ok(PcuCapabilitySnapshot {
                support: PcuSupport::unsupported(),
            })
        }
        fn device_capabilities(
            &self,
            r: PcuObjectRef,
        ) -> Result<PcuCapabilitySnapshot, Self::Error> {
            self.target_capabilities(r)
        }
        fn executors(
            &self,
            _: PcuObjectRef,
            _: &mut [PcuExecutorDescriptor],
        ) -> Result<usize, Self::Error> {
            Ok(0)
        }
    }

    #[test]
    fn orders_providers_rejects_duplicates_and_capacity_and_isolates_diagnostics() {
        let a = Mock {
            id: PcuProviderId(1),
            fail_targets: false,
        };
        let b = Mock {
            id: PcuProviderId(2),
            fail_targets: true,
        };
        let mut r = PcuRuntimeDiscoveryRegistry::<2>::new();
        assert_eq!(r.register(&a).unwrap(), a.id);
        assert_eq!(r.register(&b).unwrap(), b.id);
        let duplicate = r.register(&a).unwrap_err();
        assert_eq!(duplicate.message(), "duplicate provider id");
        let mut p = [empty_provider()];
        assert_eq!(r.providers(&mut p).unwrap(), 2);
        assert_eq!(p[0].id, PcuProviderId(1));
        let mut empty = [];
        assert!(r.targets(PcuProviderId(1), 1, &mut empty).unwrap() == 1);
        let err = r.targets(PcuProviderId(2), 1, &mut empty).unwrap_err();
        assert_eq!(err.provider, Some(PcuProviderId(2)));
        assert_eq!(err.operation, PcuDiscoveryOperation::Targets);
        assert_eq!(err.message(), "probe said no");
        assert_eq!(r.targets(PcuProviderId(1), 1, &mut empty).unwrap(), 1);
        assert_eq!(r.len(), 2);
        let mut full = PcuRuntimeDiscoveryRegistry::<1>::new();
        full.register(&a).unwrap();
        assert!(
            full.register(&b)
                .unwrap_err()
                .message()
                .contains("capacity")
        );
    }
}
