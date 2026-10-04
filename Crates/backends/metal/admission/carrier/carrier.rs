//! Cold exact scalar transport admission, independent from arithmetic support.
#[rustfmt::skip]
use fusion_pcu::{PcuDispatchKernelIr,PcuScalarType,PcuBindingRef,PcuBindingAccess,PcuBindingType,PcuValueType,validate_scalar_identity_kernel,validate_scalar_broadcast_kernel,PcuDispatchOp};
#[rustfmt::skip]
use crate::{MetalSession,MetalBuffer,MetalError};
pub struct MetalPreparedCarrierKernel {
    map: crate::runtime::carrier::Carrier,
    scalar: PcuScalarType,
    input: PcuBindingRef,
    output: PcuBindingRef,
    count: usize,
    input_bytes: usize,
    bytes: usize,
}
impl MetalPreparedCarrierKernel {
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.count
    }
    #[must_use]
    pub const fn input_binding(&self) -> PcuBindingRef {
        self.input
    }
    #[must_use]
    pub const fn output_binding(&self) -> PcuBindingRef {
        self.output
    }
    pub(crate) const fn binding_bytes(&self, binding: PcuBindingRef) -> usize {
        if binding.set == self.input.set && binding.binding == self.input.binding {
            self.input_bytes
        } else {
            self.bytes
        }
    }
    /// Executes exact initialized representations without interpreting numerical payloads.
    /// # Errors
    /// Returns extent, affinity, quarantine or terminal operational failure.
    pub fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        self.map.execute(input)
    }
    /// Writes only the admitted logical prefix of an exact-session borrowed owner.
    /// # Errors
    /// Returns extent, affinity, quarantine or terminal operational failure.
    pub fn execute_into(
        &self,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        self.map.execute_into(input, output)
    }
}
impl MetalSession {
    /// Prepares dense identity or scalar broadcast for all22 byte-addressed sealed carriers.
    /// Arithmetic and Portable reproducibility are admitted separately.
    /// # Errors
    /// Rejects invalid numerical requests/schema/SSA/extents before native compilation.
    pub fn prepare_carrier_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedCarrierKernel, MetalError> {
        crate::dispatch_shape::require_non_nested(kernel)?;
        super::require_scalar_numerics(kernel)?;
        let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|b| b.binding_type)
        else {
            return Err(MetalError::Unsupported);
        };
        if scalar.bit_width() < 8 {
            return Err(MetalError::Unsupported);
        }
        let broadcast = if validate_scalar_identity_kernel(kernel, scalar).is_ok() {
            false
        } else if validate_scalar_broadcast_kernel(kernel, scalar).is_ok() {
            true
        } else {
            return Err(MetalError::Unsupported);
        };
        if kernel.entry.logical_shape[1..] != [1, 1] {
            return Err(MetalError::Unsupported);
        }
        let count = if let [PcuDispatchOp::GridStrideLoop { extent, .. }, _] = kernel.ops {
            *extent as usize
        } else {
            kernel.entry.logical_shape[0] as usize
        };
        let input = kernel
            .bindings
            .iter()
            .find(|b| b.access == PcuBindingAccess::ReadOnly)
            .ok_or(MetalError::Unsupported)?
            .reference();
        let output = kernel
            .bindings
            .iter()
            .find(|b| b.access != PcuBindingAccess::ReadOnly)
            .ok_or(MetalError::Unsupported)?
            .reference();
        let width = usize::from(scalar.bit_width()) / 8;
        let bytes = count.checked_mul(width).ok_or(MetalError::InvalidExtent)?;
        let map = crate::runtime::carrier::Carrier::prepare(self, count, width, broadcast)?;
        Ok(MetalPreparedCarrierKernel {
            map,
            scalar,
            input,
            output,
            count,
            input_bytes: if broadcast { width } else { bytes },
            bytes,
        })
    }
}

pub fn is_carrier_kernel(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    if crate::dispatch_shape::require_non_nested(kernel).is_err() {
        return false;
    }
    let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
        kernel.bindings.first().map(|b| b.binding_type)
    else {
        return false;
    };
    scalar.bit_width() >= 8
        && (validate_scalar_identity_kernel(kernel, scalar).is_ok()
            || validate_scalar_broadcast_kernel(kernel, scalar).is_ok())
}
