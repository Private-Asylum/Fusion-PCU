//! Optional cold physical-device facts, separate from executable support and admission.
//!
//! These observations describe hardware or native API limits. They prescribe no selection
//! policy, throughput score, memory ownership, or guarantee that a kernel can execute.

/// Invalid bounded stable-identity input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuStableDeviceIdentityError {
    /// The namespace is empty, exceeds 32 bytes, or contains non-ASCII bytes.
    InvalidNamespace,
    /// The opaque identity is empty or exceeds 32 bytes.
    InvalidValue,
}

/// Owned, allocation-free identity in an explicitly named backend/API namespace.
///
/// Identity values are opaque: equality is meaningful only with the same namespace. Providers
/// must use namespaces that distinguish incompatible identity formats and must report directly
/// supplied stable identities rather than deriving them from names or enumeration indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuStableDeviceIdentity {
    namespace: [u8; 32],
    namespace_len: u8,
    value: [u8; 32],
    value_len: u8,
}

impl PcuStableDeviceIdentity {
    /// Copies a nonempty ASCII namespace and nonempty opaque identity into bounded storage.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or oversized input, or a non-ASCII namespace. Inputs are
    /// never truncated, so malformed identities cannot silently alias valid ones.
    pub fn new(namespace: &str, value: &[u8]) -> Result<Self, PcuStableDeviceIdentityError> {
        if namespace.is_empty() || namespace.len() > 32 || !namespace.is_ascii() {
            return Err(PcuStableDeviceIdentityError::InvalidNamespace);
        }
        if value.is_empty() || value.len() > 32 {
            return Err(PcuStableDeviceIdentityError::InvalidValue);
        }
        let mut result = Self {
            namespace: [0; 32],
            namespace_len: u8::try_from(namespace.len())
                .map_err(|_| PcuStableDeviceIdentityError::InvalidNamespace)?,
            value: [0; 32],
            value_len: u8::try_from(value.len())
                .map_err(|_| PcuStableDeviceIdentityError::InvalidValue)?,
        };
        result.namespace[..namespace.len()].copy_from_slice(namespace.as_bytes());
        result.value[..value.len()].copy_from_slice(value);
        Ok(result)
    }

    /// Returns the validated namespace, without storage padding.
    ///
    /// # Panics
    ///
    /// Panics if the private ASCII namespace invariant is violated by an internal implementation
    /// error. Public construction validates ASCII and cannot produce such an identity.
    #[must_use]
    pub fn namespace(&self) -> &str {
        core::str::from_utf8(&self.namespace[..usize::from(self.namespace_len)])
            .expect("identity constructor validates an ASCII namespace")
    }

    /// Returns the opaque identity bytes, without storage padding.
    #[must_use]
    pub fn value(&self) -> &[u8] {
        &self.value[..usize::from(self.value_len)]
    }
}

/// Optional physical observations and native API upper bounds for a discovered device.
///
/// Every absent value is unknown or unavailable; providers must not guess zeros or parse
/// marketing names. A directly reported zero (for example, zero copy engines) remains a known
/// value. Workgroup limits describe native API launches and may depend on configuration; they
/// do not admit arbitrary kernels or imply corresponding [`crate::PcuSupport`]. Facts can be
/// inspected before selection without activating a context, opening a session, or allocating
/// queues, streams, or buffers. They carry no unified-memory ownership guarantee.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDeviceFacts {
    /// Directly reported stable hardware/API identity with its format namespace.
    pub stable_identity: Option<PcuStableDeviceIdentity>,
    /// Native reported processor-unit count; APIs may count SMs, CUs, or WGPs.
    /// Unit meaning is API-specific, not a physical-CU guarantee or throughput score.
    pub compute_unit_count: Option<u32>,
    /// Reported native subgroup, warp, or SIMD width, not an IR-wide semantic guarantee.
    pub subgroup_width: Option<u32>,
    /// Native API upper bound on invocations in one workgroup.
    pub max_workgroup_invocations: Option<u32>,
    /// Native API upper bounds on workgroup dimensions in x, y, and z order.
    pub max_workgroup_dimensions: Option<[u32; 3]>,
    /// Native API upper bounds on workgroup counts in x, y, and z order.
    pub max_workgroup_count: Option<[u32; 3]>,
    /// Reported local/shared memory upper bound for one native workgroup, in bytes.
    pub local_memory_bytes_per_workgroup: Option<u64>,
    /// Native asynchronous copy-engine count; no overlap or bandwidth guarantee is implied.
    pub async_copy_engine_count: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_rejects_malformed_inputs_without_truncation() {
        for namespace in ["", "é", "123456789012345678901234567890123"] {
            assert_eq!(
                PcuStableDeviceIdentity::new(namespace, &[1]),
                Err(PcuStableDeviceIdentityError::InvalidNamespace)
            );
        }
        for value in [&[][..], &[0; 33][..]] {
            assert_eq!(
                PcuStableDeviceIdentity::new("api.uuid", value),
                Err(PcuStableDeviceIdentityError::InvalidValue)
            );
        }
        let identity = PcuStableDeviceIdentity::new("api.uuid", &[0, 255, 0]).unwrap();
        assert_eq!(identity.namespace(), "api.uuid");
        assert_eq!(identity.value(), &[0, 255, 0]);
        assert_ne!(
            identity,
            PcuStableDeviceIdentity::new("other.uuid", &[0, 255, 0]).unwrap()
        );
        assert_ne!(
            identity,
            PcuStableDeviceIdentity::new("api.uuid", &[0, 255]).unwrap()
        );
        let maximum =
            PcuStableDeviceIdentity::new("12345678901234567890123456789012", &[255; 32]).unwrap();
        assert_eq!(maximum.namespace().len(), 32);
        assert_eq!(maximum.value(), &[255; 32]);
    }

    #[test]
    fn unknown_facts_are_distinct_from_reported_zero() {
        let unknown = PcuDeviceFacts::default();
        assert_eq!(unknown.stable_identity, None);
        assert_eq!(unknown.compute_unit_count, None);
        assert_eq!(unknown.subgroup_width, None);
        assert_eq!(unknown.max_workgroup_invocations, None);
        assert_eq!(unknown.max_workgroup_dimensions, None);
        assert_eq!(unknown.max_workgroup_count, None);
        assert_eq!(unknown.local_memory_bytes_per_workgroup, None);
        assert_eq!(unknown.async_copy_engine_count, None);
        assert_ne!(
            unknown,
            PcuDeviceFacts {
                async_copy_engine_count: Some(0),
                ..unknown
            }
        );
    }
}
