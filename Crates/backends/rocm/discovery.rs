//! Runtime-backed discovery for the `ROCm` provider.

use std::{
    process::Command,
    sync::atomic::{
        AtomicU64,
        Ordering,
    },
};

use fusion_pcu::{
    PcuCapabilitySnapshot,
    PcuCaps,
    PcuContextDescriptor,
    PcuContextKind,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuDeviceLocation,
    PcuDeviceActivation,
    PcuDispatchFeatureCaps,
    PcuDispatchOpCaps,
    PcuDispatchPolicyCaps,
    PcuDispatchSupport,
    PcuExecutorClass,
    PcuExecutorDescriptor,
    PcuExecutorId,
    PcuExecutorOrigin,
    PcuExecutorSupport,
    PcuFeatureSupport,
    PcuImplementationKind,
    PcuMemoryDomainDescriptor,
    PcuMemoryDomainKind,
    PcuObjectKind,
    PcuObjectRef,
    PcuPrimitiveCaps,
    PcuPrimitiveSupport,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus as Status,
    PcuRuntimeDiscovery,
    PcuSupport,
    PcuTargetDescriptor,
    PcuValueTypeCaps,
    PcuDispatchKernelIr,
};

use crate::{
    HipDeviceInfo,
    HipError,
    HipRuntime,
    RocmDispatchError,
    compile_hip_source,
    compile_hip_source_for_device,
    codegen::rtc::hiprtc_available,
    lower_dispatch_to_hip_source,
};

const PROVIDER: PcuProviderId = PcuProviderId(0x524f_434d);
const TARGET_ID: u32 = 0;

const fn f32_alu_caps() -> PcuDispatchOpCaps {
    PcuDispatchOpCaps::ALU_ADD
        .union(PcuDispatchOpCaps::ALU_SUB)
        .union(PcuDispatchOpCaps::ALU_MUL)
        .union(PcuDispatchOpCaps::ALU_DIV)
}

const fn int_alu_caps() -> PcuDispatchOpCaps {
    PcuDispatchOpCaps::ALU_ADD
        .union(PcuDispatchOpCaps::ALU_SUB)
        .union(PcuDispatchOpCaps::ALU_MUL)
}

const fn checked_u32_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps().union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
}

const fn checked_u64_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps().union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
}

const fn checked_i32_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps().union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
}
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// A stable `ROCm` runtime snapshot. Device names are owned by this value and borrowed by discovery.
pub struct RocmDiscovery {
    generation: u64,
    devices: Vec<HipDeviceInfo>,
    unavailable_reason: Option<String>,
    compiler_available: bool,
    compiler_reason: Option<String>,
    hiprtc_available: bool,
    hiprtc_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchCompiler {
    Hipcc,
    HipRtc,
}

impl RocmDiscovery {
    /// Probe and snapshot the HIP runtime. Probe failures remain visible as unavailable readiness.
    pub fn new() -> Self {
        let generation = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed).max(1);
        let compiler = std::env::var_os("HIPCC").unwrap_or_else(|| "hipcc".into());
        let compiler_available = Command::new(compiler)
            .arg("--version")
            .output()
            .is_ok_and(|out| out.status.success());
        let compiler_reason = (!compiler_available).then(|| {
            "HIP compiler is unavailable; source-lowered dispatch cannot be compiled".to_string()
        });
        let hiprtc_result = hiprtc_available();
        let hiprtc_available = hiprtc_result.is_ok();
        let hiprtc_reason = hiprtc_result.err().map(|error| error.to_string());
        match HipRuntime::enumerate_devices() {
            Ok(devices) => Self {
                generation,
                devices,
                unavailable_reason: None,
                compiler_available,
                compiler_reason,
                hiprtc_available,
                hiprtc_reason,
            },
            Err(error) => Self {
                generation,
                devices: Vec::new(),
                unavailable_reason: Some(error.to_string()),
                compiler_available,
                compiler_reason,
                hiprtc_available,
                hiprtc_reason,
            },
        }
    }

    fn readiness(&self) -> PcuProviderReadiness<'_> {
        match &self.unavailable_reason {
            Some(reason) => PcuProviderReadiness {
                status: Status::Unavailable,
                reason: Some(reason),
            },
            None if self.devices.is_empty() => PcuProviderReadiness {
                status: Status::Degraded,
                reason: Some("HIP runtime is available but reports no visible devices"),
            },
            None if !self.compiler_available && !self.hiprtc_available => PcuProviderReadiness {
                status: Status::Degraded,
                reason: self
                    .compiler_reason
                    .as_deref()
                    .or(self.hiprtc_reason.as_deref()),
            },
            None => PcuProviderReadiness {
                status: Status::Ready,
                reason: None,
            },
        }
    }

    const fn reference(&self, kind: PcuObjectKind, id: u32) -> PcuObjectRef {
        PcuObjectRef {
            provider: PROVIDER,
            generation: self.generation,
            kind,
            id,
        }
    }

    fn validate(&self, reference: PcuObjectRef, kind: PcuObjectKind) -> Result<(), HipError> {
        if reference.provider != PROVIDER
            || reference.generation != self.generation
            || reference.kind != kind
        {
            return Err(HipError::InvalidDiscoveryReference);
        }
        let valid_id = match kind {
            PcuObjectKind::Target => reference.id == TARGET_ID,
            PcuObjectKind::Device | PcuObjectKind::Context | PcuObjectKind::MemoryDomain => self
                .devices
                .iter()
                .any(|device| u32::try_from(device.index).ok() == Some(reference.id)),
        };
        if valid_id {
            Ok(())
        } else {
            Err(HipError::InvalidDiscoveryReference)
        }
    }

    /// Return the immutable facts recorded for a selected device reference.
    ///
    /// # Errors
    ///
    /// Returns [`HipError::InvalidDiscoveryReference`] if the reference does not identify a
    /// device in this discovery snapshot.
    pub fn device_info(&self, device: PcuObjectRef) -> Result<&HipDeviceInfo, HipError> {
        self.validate(device, PcuObjectKind::Device)?;
        self.devices
            .iter()
            .find(|info| u32::try_from(info.index).ok() == Some(device.id))
            .ok_or(HipError::InvalidDiscoveryReference)
    }

    pub(crate) fn dispatch_compiler(
        &self,
        device: PcuObjectRef,
    ) -> Result<DispatchCompiler, HipError> {
        let info = self.device_info(device)?;
        if info.architecture.is_some() && self.compiler_available {
            return Ok(DispatchCompiler::Hipcc);
        }
        if self.hiprtc_available {
            return Ok(DispatchCompiler::HipRtc);
        }
        Err(HipError::MissingArchitecture)
    }

    /// Open an explicitly selected device, confirming its physical identity after reopening HIP.
    ///
    /// HIP ordinals can change between discovery and activation. To avoid silently opening a
    /// different GPU after such a change, activation requires the PCI bus ID to be available in
    /// both the discovery snapshot and the live HIP query, and requires them to match.
    ///
    /// # Errors
    ///
    /// Returns an error when the reference is invalid, HIP cannot open the device, its stable
    /// identity is unavailable, or the live identity differs from the discovery snapshot.
    pub fn open_device(&self, device: PcuObjectRef) -> Result<HipRuntime, HipError> {
        let snapshot = self.device_info(device)?;
        let expected = snapshot
            .pci_bus_id
            .as_deref()
            .ok_or(HipError::MissingStableDeviceIdentity)?;
        let index =
            u32::try_from(snapshot.index).map_err(|_| HipError::InvalidDiscoveryReference)?;
        let runtime = HipRuntime::new(index)?;
        let current = runtime.device_info()?;
        let actual = current
            .pci_bus_id
            .as_deref()
            .ok_or(HipError::MissingStableDeviceIdentity)?;
        verify_device_identity(expected, actual)?;
        Ok(runtime)
    }
}

fn verify_device_identity(expected: &str, actual: &str) -> Result<(), HipError> {
    if expected.eq_ignore_ascii_case(actual) {
        Ok(())
    } else {
        Err(HipError::DeviceIdentityChanged {
            expected: expected.to_string(),
            actual: actual.to_string(),
        })
    }
}

impl Default for RocmDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl PcuDeviceActivation for RocmDiscovery {
    type Session = HipRuntime;
    type Error = HipError;

    fn open_device(&self, device: PcuObjectRef) -> Result<Self::Session, Self::Error> {
        Self::open_device(self, device)
    }
}

impl PcuRuntimeDiscovery for RocmDiscovery {
    type Error = HipError;

    fn providers<'a>(
        &'a self,
        output: &mut [PcuProviderDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
        if let Some(slot) = output.first_mut() {
            *slot = PcuProviderDescriptor {
                id: PROVIDER,
                generation: self.generation,
                backend: "rocm",
                readiness: self.readiness(),
            };
        }
        Ok(1)
    }

    fn targets<'a>(
        &'a self,
        provider: PcuProviderId,
        generation: u64,
        output: &mut [PcuTargetDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
        if provider != PROVIDER || generation != self.generation {
            return Err(HipError::InvalidDiscoveryReference);
        }
        if let Some(slot) = output.first_mut() {
            *slot = PcuTargetDescriptor {
                reference: self.reference(PcuObjectKind::Target, TARGET_ID),
                name: "AMD HIP runtime",
                readiness: self.readiness(),
            };
        }
        Ok(1)
    }

    fn devices<'a>(
        &'a self,
        target: PcuObjectRef,
        output: &mut [PcuDeviceDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
        self.validate(target, PcuObjectKind::Target)?;
        for (info, slot) in self.devices.iter().zip(output.iter_mut()) {
            let id = u32::try_from(info.index).map_err(|_| HipError::InvalidDiscoveryReference)?;
            *slot = PcuDeviceDescriptor {
                reference: self.reference(PcuObjectKind::Device, id),
                target,
                name: &info.name,
                vendor: Some(&info.vendor),
                architecture: info.architecture.as_deref(),
                generation: info.generation.as_deref(),
                location: info.pci_bus_id.as_deref().map(PcuDeviceLocation::PciBusId),
                class: PcuDeviceClass::Gpu,
            };
        }
        Ok(self.devices.len())
    }

    fn contexts<'a>(
        &'a self,
        device: PcuObjectRef,
        output: &mut [PcuContextDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
        self.validate(device, PcuObjectKind::Device)?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuContextDescriptor {
                reference: self.reference(PcuObjectKind::Context, device.id),
                device,
                name: "HIP compute context",
                kind: PcuContextKind::Compute,
            };
        }
        Ok(1)
    }

    fn memory_domains<'a>(
        &'a self,
        context: PcuObjectRef,
        output: &mut [PcuMemoryDomainDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
        self.validate(context, PcuObjectKind::Context)?;
        let info = self
            .devices
            .iter()
            .find(|device| u32::try_from(device.index).ok() == Some(context.id))
            .ok_or(HipError::InvalidDiscoveryReference)?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuMemoryDomainDescriptor {
                reference: self.reference(PcuObjectKind::MemoryDomain, context.id),
                context,
                name: "HIP device-local memory",
                kind: PcuMemoryDomainKind::DeviceLocal,
                capacity_bytes: Some(info.total_memory),
            };
        }
        Ok(1)
    }

    fn target_capabilities(
        &self,
        target: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, Self::Error> {
        self.validate(target, PcuObjectKind::Target)?;
        let has_codegen_target = self
            .devices
            .iter()
            .any(|device| self.compiler_for_info(device));
        let mut support = self.support(has_codegen_target);
        // Executors are enumerated under individual devices, not the aggregate target.
        support.executor_count = 0;
        Ok(PcuCapabilitySnapshot { support })
    }

    fn device_capabilities(
        &self,
        device: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, Self::Error> {
        self.validate(device, PcuObjectKind::Device)?;
        let has_codegen_target = self.dispatch_compiler(device).is_ok();
        Ok(PcuCapabilitySnapshot {
            support: self.support(has_codegen_target),
        })
    }

    fn executors(
        &self,
        object: PcuObjectRef,
        output: &mut [PcuExecutorDescriptor],
    ) -> Result<usize, Self::Error> {
        match object.kind {
            PcuObjectKind::Target => self.validate(object, PcuObjectKind::Target)?,
            PcuObjectKind::Device => self.validate(object, PcuObjectKind::Device)?,
            PcuObjectKind::Context => self.validate(object, PcuObjectKind::Context)?,
            PcuObjectKind::MemoryDomain => self.validate(object, PcuObjectKind::MemoryDomain)?,
        }
        if object.kind != PcuObjectKind::Device && object.kind != PcuObjectKind::Context {
            return Ok(0);
        }
        let has_codegen_target = self.devices.iter().any(|device| {
            u32::try_from(device.index).ok() == Some(object.id) && self.compiler_for_info(device)
        });
        if self.devices.is_empty() || self.unavailable_reason.is_some() || !has_codegen_target {
            return Ok(0);
        }
        if let Some(slot) = output.first_mut() {
            *slot = ROCM_EXECUTOR;
        }
        Ok(1)
    }
}

impl RocmDiscovery {
    const fn compiler_for_info(&self, info: &HipDeviceInfo) -> bool {
        (info.architecture.is_some() && self.compiler_available) || self.hiprtc_available
    }

    const fn support(&self, has_codegen_target: bool) -> PcuSupport {
        let mut support = PcuSupport::unsupported();
        support.caps = PcuCaps::ENUMERATE_EXECUTORS;
        if !self.devices.is_empty() && self.unavailable_reason.is_none() {
            support.caps = support.caps.union(PcuCaps::DEVICE_LOCAL_MEMORY);
        }
        if self.devices.is_empty() || self.unavailable_reason.is_some() || !has_codegen_target {
            return support;
        }
        support.caps = support
            .caps
            .union(PcuCaps::DISPATCH)
            .union(PcuCaps::COMPUTE_DISPATCH);
        support.implementation = PcuImplementationKind::Native;
        support.executor_count = 1;
        support.primitive_support = PcuPrimitiveSupport {
            primitives: PcuFeatureSupport::new(
                PcuPrimitiveCaps::DISPATCH,
                PcuPrimitiveCaps::empty(),
            ),
        };
        let types = PcuValueTypeCaps::FLOAT32
            .union(PcuValueTypeCaps::FLOAT64)
            .union(PcuValueTypeCaps::INT8)
            .union(PcuValueTypeCaps::UINT8)
            .union(PcuValueTypeCaps::UINT16)
            .union(PcuValueTypeCaps::UINT32)
            .union(PcuValueTypeCaps::INT32)
            .union(PcuValueTypeCaps::INT16)
            .union(PcuValueTypeCaps::UINT64)
            .union(PcuValueTypeCaps::INT64)
            .union(PcuValueTypeCaps::SCALAR_VALUES);
        support.value_type_support = PcuFeatureSupport::new(types, PcuValueTypeCaps::empty());
        let instructions = PcuDispatchOpCaps::VALUE_CONSTANT
            .union(PcuDispatchOpCaps::ALU_ADD)
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL)
            .union(PcuDispatchOpCaps::ALU_DIV)
            .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
            .union(PcuDispatchOpCaps::CONTROL_RETURN)
            .union(PcuDispatchOpCaps::BINDING_LOAD)
            .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
            .union(PcuDispatchOpCaps::BINDING_STORE);
        let features = PcuDispatchFeatureCaps::MUTABLE_RESOURCES
            .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES);
        support.dispatch_support = PcuDispatchSupport {
            flags: PcuDispatchPolicyCaps::SERIAL.union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION),
            instructions: PcuFeatureSupport::new(instructions, PcuDispatchOpCaps::empty()),
            scalar_alu: PcuFeatureSupport::new(
                fusion_pcu::PcuDispatchScalarAluSupport::empty()
                    .with(fusion_pcu::PcuScalarType::F32, f32_alu_caps())
                    .with(fusion_pcu::PcuScalarType::F64, f32_alu_caps())
                    .with(fusion_pcu::PcuScalarType::U32, checked_u32_alu_caps())
                    .with(fusion_pcu::PcuScalarType::U16, int_alu_caps())
                    .with(fusion_pcu::PcuScalarType::I16, int_alu_caps())
                    .with(fusion_pcu::PcuScalarType::U8, int_alu_caps())
                    .with(fusion_pcu::PcuScalarType::I8, int_alu_caps())
                    .with(fusion_pcu::PcuScalarType::I32, checked_i32_alu_caps())
                    .with(fusion_pcu::PcuScalarType::U64, checked_u64_alu_caps())
                    .with(fusion_pcu::PcuScalarType::I64, int_alu_caps()),
                fusion_pcu::PcuDispatchScalarAluSupport::empty(),
            ),
            features: PcuFeatureSupport::new(features, PcuDispatchFeatureCaps::empty()),
        };
        support
    }

    /// Assess a dispatch program using the compiler path advertised for its device.
    ///
    /// # Errors
    ///
    /// Returns an error when the reference is invalid, lowering rejects the kernel, or HIP
    /// compilation fails.
    pub fn assess_dispatch(
        &self,
        device: PcuObjectRef,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<(), RocmDispatchError> {
        let compiler = self.dispatch_compiler(device).map_err(|error| {
            if matches!(error, HipError::MissingArchitecture) {
                RocmDispatchError::CompilerUnavailable
            } else {
                RocmDispatchError::Hip(error)
            }
        })?;
        let source = lower_dispatch_to_hip_source(kernel).map_err(RocmDispatchError::Lower)?;
        match compiler {
            DispatchCompiler::Hipcc => {
                let architecture = self
                    .device_info(device)
                    .map_err(RocmDispatchError::Hip)?
                    .architecture
                    .as_deref()
                    .ok_or(RocmDispatchError::Hip(HipError::MissingArchitecture))?;
                compile_hip_source(&source, architecture)
                    .map(|_| ())
                    .map_err(RocmDispatchError::Compile)
            }
            DispatchCompiler::HipRtc => {
                let runtime = self.open_device(device).map_err(RocmDispatchError::Hip)?;
                let rtc_source = crate::lower_dispatch_to_hip_rtc_source(kernel)
                    .map_err(RocmDispatchError::Lower)?;
                compile_hip_source_for_device(&runtime, &rtc_source)
                    .map(|_| ())
                    .map_err(RocmDispatchError::HipRtc)
            }
        }
    }
}

const ROCM_EXECUTOR: PcuExecutorDescriptor = PcuExecutorDescriptor {
    id: PcuExecutorId(0),
    name: "rocm-dispatch",
    class: PcuExecutorClass::Compute,
    origin: PcuExecutorOrigin::TopologyBound,
    support: PcuExecutorSupport {
        primitives: PcuPrimitiveCaps::DISPATCH,
        dispatch_policy: PcuDispatchPolicyCaps::SERIAL
            .union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION),
        value_types: PcuValueTypeCaps::FLOAT32
            .union(PcuValueTypeCaps::FLOAT64)
            .union(PcuValueTypeCaps::INT8)
            .union(PcuValueTypeCaps::UINT8)
            .union(PcuValueTypeCaps::UINT16)
            .union(PcuValueTypeCaps::UINT32)
            .union(PcuValueTypeCaps::INT32)
            .union(PcuValueTypeCaps::INT16)
            .union(PcuValueTypeCaps::UINT64)
            .union(PcuValueTypeCaps::INT64)
            .union(PcuValueTypeCaps::SCALAR_VALUES),
        dispatch_instructions: PcuDispatchOpCaps::VALUE_CONSTANT
            .union(PcuDispatchOpCaps::ALU_ADD)
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL)
            .union(PcuDispatchOpCaps::ALU_DIV)
            .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
            .union(PcuDispatchOpCaps::CONTROL_RETURN)
            .union(PcuDispatchOpCaps::BINDING_LOAD)
            .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
            .union(PcuDispatchOpCaps::BINDING_STORE),
        dispatch_scalar_alu: fusion_pcu::PcuDispatchScalarAluSupport::empty()
            .with(fusion_pcu::PcuScalarType::F32, f32_alu_caps())
            .with(fusion_pcu::PcuScalarType::F64, f32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U32, checked_u32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U16, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::I16, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::U8, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::I8, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::I32, checked_i32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U64, checked_u64_alu_caps())
            .with(fusion_pcu::PcuScalarType::I64, int_alu_caps()),
        dispatch_features: PcuDispatchFeatureCaps::MUTABLE_RESOURCES
            .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES),
        stream_instructions: fusion_pcu::PcuStreamCapabilities::empty(),
        command_instructions: fusion_pcu::PcuCommandOpCaps::empty(),
        transaction_features: fusion_pcu::PcuTransactionFeatureCaps::empty(),
        signal_instructions: fusion_pcu::PcuSignalOpCaps::empty(),
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RocmDiscovery {
        RocmDiscovery {
            generation: 77,
            devices: vec![HipDeviceInfo {
                index: 2,
                name: "Test AMD GPU".into(),
                vendor: "AMD".into(),
                architecture: Some("gfx1030".into()),
                generation: None,
                pci_bus_id: Some("0000:03:00.0".into()),
                total_memory: 8 * 1024 * 1024,
            }],
            unavailable_reason: None,
            compiler_available: true,
            compiler_reason: None,
            hiprtc_available: false,
            hiprtc_reason: Some("HIPRTC unavailable".into()),
        }
    }

    #[test]
    fn discovery_advertises_f64_type_with_its_alu_operations() {
        let required = PcuValueTypeCaps::FLOAT64.union(PcuValueTypeCaps::SCALAR_VALUES);
        assert!(
            sample()
                .support(true)
                .value_type_support
                .direct
                .contains(required)
        );
        assert!(ROCM_EXECUTOR.support.value_types.contains(required));
        assert!(
            ROCM_EXECUTOR
                .support
                .dispatch_scalar_alu
                .for_scalar(fusion_pcu::PcuScalarType::F64)
                .contains(PcuDispatchOpCaps::ALU_ADD)
        );
    }

    #[test]
    fn discovery_advertises_checked_div_rem_for_i32_u32_and_u64_only() {
        let support = sample().support(true);
        let checked = PcuDispatchOpCaps::ALU_CHECKED_DIV_REM;
        assert!(
            support
                .dispatch_support
                .instructions
                .direct
                .contains(checked)
        );
        assert!(
            ROCM_EXECUTOR
                .support
                .dispatch_instructions
                .contains(checked)
        );
        for scalar in [
            fusion_pcu::PcuScalarType::U32,
            fusion_pcu::PcuScalarType::U64,
            fusion_pcu::PcuScalarType::I32,
        ] {
            for alu in [
                support
                    .dispatch_support
                    .scalar_alu
                    .direct
                    .for_scalar(scalar),
                ROCM_EXECUTOR.support.dispatch_scalar_alu.for_scalar(scalar),
            ] {
                assert!(alu.contains(checked));
                assert!(!alu.contains(PcuDispatchOpCaps::ALU_DIV));
            }
        }
        assert!(
            !support
                .dispatch_support
                .scalar_alu
                .direct
                .for_scalar(fusion_pcu::PcuScalarType::I64)
                .contains(checked)
        );
    }

    #[test]
    fn discovery_advertises_u16_wrapping_alu_support() {
        let support = sample().support(true);
        assert!(
            support
                .value_type_support
                .direct
                .contains(PcuValueTypeCaps::UINT16.union(PcuValueTypeCaps::SCALAR_VALUES))
        );
        let alu = ROCM_EXECUTOR
            .support
            .dispatch_scalar_alu
            .for_scalar(fusion_pcu::PcuScalarType::U16);
        let expected = PcuDispatchOpCaps::ALU_ADD
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL);
        assert!(alu.contains(expected));
        assert!(!alu.contains(PcuDispatchOpCaps::ALU_DIV));
        let device_alu = support
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(fusion_pcu::PcuScalarType::U16);
        assert!(device_alu.contains(expected));
        assert!(!device_alu.contains(PcuDispatchOpCaps::ALU_DIV));
    }

    #[test]
    fn discovery_advertises_u8_wrapping_alu_support() {
        let support = sample().support(true);
        assert!(
            support
                .value_type_support
                .direct
                .contains(PcuValueTypeCaps::UINT8.union(PcuValueTypeCaps::SCALAR_VALUES))
        );
        let expected = PcuDispatchOpCaps::ALU_ADD
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL);
        for alu in [
            ROCM_EXECUTOR
                .support
                .dispatch_scalar_alu
                .for_scalar(fusion_pcu::PcuScalarType::U8),
            support
                .dispatch_support
                .scalar_alu
                .direct
                .for_scalar(fusion_pcu::PcuScalarType::U8),
        ] {
            assert!(alu.contains(expected));
            assert!(!alu.contains(PcuDispatchOpCaps::ALU_DIV));
        }
    }

    #[test]
    fn discovery_advertises_i16_wrapping_alu_support() {
        let support = sample().support(true);
        assert!(
            support
                .value_type_support
                .direct
                .contains(PcuValueTypeCaps::INT16.union(PcuValueTypeCaps::SCALAR_VALUES))
        );
        let expected = PcuDispatchOpCaps::ALU_ADD
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL);
        let alu = support
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(fusion_pcu::PcuScalarType::I16);
        assert!(alu.contains(expected));
        assert!(!alu.contains(PcuDispatchOpCaps::ALU_DIV));
    }

    #[test]
    fn discovery_advertises_i8_wrapping_alu_support() {
        let support = sample().support(true);
        assert!(
            support
                .value_type_support
                .direct
                .contains(PcuValueTypeCaps::INT8.union(PcuValueTypeCaps::SCALAR_VALUES))
        );
        let expected = PcuDispatchOpCaps::ALU_ADD
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL);
        let alu = support
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(fusion_pcu::PcuScalarType::I8);
        assert!(alu.contains(expected));
        assert!(!alu.contains(PcuDispatchOpCaps::ALU_DIV));
    }

    #[test]
    fn bounded_snapshot_traversal_and_device_scoped_executor_are_consistent() {
        let discovery = sample();
        let mut provider = [PcuProviderDescriptor {
            id: PROVIDER,
            generation: 0,
            backend: "",
            readiness: PcuProviderReadiness {
                status: Status::Unavailable,
                reason: None,
            },
        }];
        assert_eq!(discovery.providers(&mut provider).unwrap(), 1);
        assert_eq!(provider[0].readiness.status, Status::Ready);

        let mut target = [PcuTargetDescriptor {
            reference: discovery.reference(PcuObjectKind::Target, 0),
            name: "",
            readiness: PcuProviderReadiness {
                status: Status::Unavailable,
                reason: None,
            },
        }];
        assert_eq!(discovery.targets(PROVIDER, 77, &mut target).unwrap(), 1);
        let mut devices = [PcuDeviceDescriptor {
            reference: discovery.reference(PcuObjectKind::Device, 0),
            target: target[0].reference,
            name: "",
            vendor: None,
            architecture: None,
            generation: None,
            location: None,
            class: PcuDeviceClass::Other,
        }];
        assert_eq!(
            discovery
                .devices(target[0].reference, &mut devices)
                .unwrap(),
            1
        );
        assert_eq!(devices[0].reference.id, 2);
        assert_eq!(devices[0].vendor, Some("AMD"));

        let mut context = [PcuContextDescriptor {
            reference: discovery.reference(PcuObjectKind::Context, 2),
            device: devices[0].reference,
            name: "",
            kind: PcuContextKind::Other,
        }];
        assert_eq!(
            discovery
                .contexts(devices[0].reference, &mut context)
                .unwrap(),
            1
        );
        let mut domain = [PcuMemoryDomainDescriptor {
            reference: discovery.reference(PcuObjectKind::MemoryDomain, 2),
            context: context[0].reference,
            name: "",
            kind: PcuMemoryDomainKind::Other,
            capacity_bytes: None,
        }];
        assert_eq!(
            discovery
                .memory_domains(context[0].reference, &mut domain)
                .unwrap(),
            1
        );
        assert_eq!(domain[0].capacity_bytes, Some(8 * 1024 * 1024));

        let mut executors = [];
        assert_eq!(
            discovery
                .executors(target[0].reference, &mut executors)
                .unwrap(),
            0
        );
        assert_eq!(
            discovery
                .executors(devices[0].reference, &mut executors)
                .unwrap(),
            1
        );
        let support = discovery
            .device_capabilities(devices[0].reference)
            .unwrap()
            .support;
        assert!(support.caps.contains(PcuCaps::DISPATCH));
        assert!(support.dispatch_support.instructions.direct.contains(
            PcuDispatchOpCaps::BINDING_LOAD.union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
        ));
    }

    #[test]
    fn missing_runtime_is_reported_as_unavailable_and_old_references_are_rejected() {
        let mut discovery = sample();
        discovery.unavailable_reason = Some("HIP runtime unavailable: test".into());
        let mut provider = [PcuProviderDescriptor {
            id: PROVIDER,
            generation: 0,
            backend: "",
            readiness: PcuProviderReadiness {
                status: Status::Ready,
                reason: None,
            },
        }];
        assert_eq!(discovery.providers(&mut provider).unwrap(), 1);
        assert_eq!(provider[0].readiness.status, Status::Unavailable);
        assert_eq!(
            discovery
                .device_capabilities(discovery.reference(PcuObjectKind::Device, 2))
                .unwrap()
                .support
                .caps,
            PcuCaps::ENUMERATE_EXECUTORS
        );
        let mut stale = discovery.reference(PcuObjectKind::Target, 0);
        stale.generation -= 1;
        assert!(matches!(
            discovery.devices(stale, &mut []),
            Err(HipError::InvalidDiscoveryReference)
        ));
    }

    #[test]
    fn activation_reference_validation_is_provider_generation_kind_and_id_scoped() {
        let discovery = sample();
        let valid = discovery.reference(PcuObjectKind::Device, 2);
        assert_eq!(discovery.device_info(valid).unwrap().index, 2);

        let mut wrong_provider = valid;
        wrong_provider.provider = PcuProviderId(99);
        assert!(matches!(
            discovery.device_info(wrong_provider),
            Err(HipError::InvalidDiscoveryReference)
        ));

        let mut stale = valid;
        stale.generation += 1;
        assert!(matches!(
            discovery.device_info(stale),
            Err(HipError::InvalidDiscoveryReference)
        ));

        let wrong_kind = discovery.reference(PcuObjectKind::Target, 0);
        assert!(matches!(
            discovery.device_info(wrong_kind),
            Err(HipError::InvalidDiscoveryReference)
        ));

        let missing_id = discovery.reference(PcuObjectKind::Device, 1);
        assert!(matches!(
            discovery.device_info(missing_id),
            Err(HipError::InvalidDiscoveryReference)
        ));
    }

    #[test]
    fn activation_requires_live_pci_identity_to_match_snapshot() {
        assert!(verify_device_identity("0000:03:00.0", "0000:03:00.0").is_ok());
        assert!(verify_device_identity("0000:03:00.0", "0000:04:00.0").is_err());
        assert!(verify_device_identity("0000:03:00.0", "0000:03:00.1").is_err());
    }

    #[test]
    fn compiler_unavailable_suppresses_dispatch_claims() {
        let mut discovery = sample();
        discovery.compiler_available = false;
        discovery.compiler_reason = Some("hipcc unavailable".into());
        discovery.hiprtc_available = false;
        discovery.hiprtc_reason = Some("HIPRTC unavailable".into());
        let mut provider = [PcuProviderDescriptor {
            id: PROVIDER,
            generation: 0,
            backend: "",
            readiness: PcuProviderReadiness {
                status: Status::Ready,
                reason: None,
            },
        }];
        discovery.providers(&mut provider).unwrap();
        assert_eq!(provider[0].readiness.status, Status::Degraded);
        let snapshot = discovery.support(false);
        assert!(!snapshot.caps.contains(PcuCaps::DISPATCH));
        assert!(snapshot.caps.contains(PcuCaps::ENUMERATE_EXECUTORS));
        let device = discovery.reference(PcuObjectKind::Device, 2);
        assert_eq!(discovery.executors(device, &mut []).unwrap(), 0);
    }

    #[test]
    fn missing_codegen_target_preserves_device_memory_but_suppresses_dispatch() {
        let mut discovery = sample();
        discovery.devices[0].architecture = None;
        let device = discovery.reference(PcuObjectKind::Device, 2);
        let support = discovery.device_capabilities(device).unwrap().support;
        assert!(support.caps.contains(PcuCaps::DEVICE_LOCAL_MEMORY));
        assert!(!support.caps.contains(PcuCaps::DISPATCH));
        assert_eq!(discovery.executors(device, &mut []).unwrap(), 0);
    }

    #[test]
    fn runtime_compiler_claims_dispatch_without_architecture_or_hipcc() {
        let mut discovery = sample();
        discovery.devices[0].architecture = None;
        discovery.compiler_available = false;
        discovery.compiler_reason = Some("hipcc unavailable".into());
        discovery.hiprtc_available = true;
        discovery.hiprtc_reason = None;
        let device = discovery.reference(PcuObjectKind::Device, 2);
        assert_eq!(
            discovery.dispatch_compiler(device).unwrap(),
            DispatchCompiler::HipRtc
        );
        assert!(
            discovery
                .device_capabilities(device)
                .unwrap()
                .support
                .caps
                .contains(PcuCaps::DISPATCH)
        );
        assert_eq!(discovery.executors(device, &mut []).unwrap(), 1);
    }

    #[test]
    fn known_architecture_prefers_hipcc_even_when_runtime_compiler_exists() {
        let mut discovery = sample();
        discovery.hiprtc_available = true;
        assert_eq!(
            discovery
                .dispatch_compiler(discovery.reference(PcuObjectKind::Device, 2))
                .unwrap(),
            DispatchCompiler::Hipcc
        );
    }

    #[test]
    fn assessment_rejects_stale_device_before_lowering_or_compiling() {
        let discovery = sample();
        let kernel = fusion_pcu::PcuDispatchKernelIr {
            id: fusion_pcu::PcuKernelId(1),
            entry: fusion_pcu::PcuDispatchEntryPoint {
                name: "empty",
                logical_shape: [1, 1, 1],
            },
            bindings: &[],
            ports: &[],
            parameters: &[],
            ops: &[],
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        };
        let mut stale = discovery.reference(PcuObjectKind::Device, 2);
        stale.generation -= 1;
        assert!(matches!(
            discovery.assess_dispatch(stale, &kernel),
            Err(RocmDispatchError::Hip(HipError::InvalidDiscoveryReference))
        ));
    }
}
