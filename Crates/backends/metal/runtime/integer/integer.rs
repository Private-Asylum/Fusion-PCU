//! Exact fourteen-width integer arithmetic using only defined unsigned32-bit limb operations.
#[rustfmt::skip]
use fusion_pcu::{PcuScalarType,PcuRangePolicy};
#[rustfmt::skip]
use super::{MetalSession,MetalBuffer,MetalError,MetalFault,MetalIntegerOp,ffi,
    execute_byte_profile,execute_byte_profile_completed,execute_byte_profile_into};
pub struct IntegerMap {
    session: MetalSession,
    pipeline: ffi::Pipeline,
    operation: u32,
    fault_law: Option<fusion_pcu::PcuCheckedScalarFaultLaw>,
    bytes: usize,
    count: usize,
    input_bytes: [usize; 2],
}
impl MetalSession {
    pub(crate) fn prepare_checked_integer_map(
        &self,
        scalar: PcuScalarType,
        operation: MetalIntegerOp,
        range: PcuRangePolicy,
        count: usize,
        broadcast: [bool; 2],
    ) -> Result<IntegerMap, MetalError> {
        let signed = match scalar {
            PcuScalarType::I8
            | PcuScalarType::I16
            | PcuScalarType::I32
            | PcuScalarType::I64
            | PcuScalarType::I128
            | PcuScalarType::I256
            | PcuScalarType::I512 => true,
            PcuScalarType::U8
            | PcuScalarType::U16
            | PcuScalarType::U32
            | PcuScalarType::U64
            | PcuScalarType::U128
            | PcuScalarType::U256
            | PcuScalarType::U512 => false,
            _ => return Err(MetalError::Unsupported),
        };
        let width = usize::from(scalar.bit_width()) / 8;
        let bytes = count.checked_mul(width).ok_or(MetalError::InvalidExtent)?;
        // Every byte index in the shader is UInt32; cold extent validation proves no wrap.
        if count == 0 || u32::try_from(bytes).is_err() {
            return Err(MetalError::InvalidExtent);
        }
        super::validate_byte_extent(bytes, self.0.facts.max_buffer_bytes)?;
        super::validate_extent(count, self.0.facts.max_buffer_bytes)?;
        let bits = scalar.bit_width();
        let mask = if bits < 32 {
            (1_u32 << bits) - 1
        } else {
            u32::MAX
        };
        let sign = 1_u32 << if bits < 32 { bits - 1 } else { 31 };
        let source = format!(
            "#define PCU_WIDTH {bits}u\n#define PCU_SIGNED {}u\n#define PCU_MASK {mask}u\n#define PCU_SIGN {sign}u\n{}",
            u32::from(signed),
            include_str!("shader/checked_integer.metal")
        );
        let code = match operation {
            MetalIntegerOp::Add => 0,
            MetalIntegerOp::Subtract => 1,
            MetalIntegerOp::Multiply => 2,
            MetalIntegerOp::Identity => 3,
            MetalIntegerOp::Divide => return Err(MetalError::Unsupported),
        };
        let pipeline = self.0.native.compile(&source, "pcu_checked_integer")?;
        Ok(IntegerMap {
            session: self.clone(),
            fault_law: super::fault::integer(scalar, operation, range)?,
            pipeline,
            bytes,
            count,
            operation: code
                | (u32::from(range == PcuRangePolicy::Clamp) << 8)
                | (u32::from(broadcast[0]) << 16)
                | (u32::from(broadcast[1]) << 17),
            input_bytes: broadcast.map(|scalar| if scalar { width } else { bytes }),
        })
    }
}
impl IntegerMap {
    pub(crate) fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        execute_byte_profile(
            &self.session,
            &self.pipeline,
            inputs,
            self.operation,
            self.bytes,
            self.count,
            self.input_bytes,
            self.fault_law,
        )
    }
    pub(crate) fn execute_completed(
        &self,
        inputs: [&MetalBuffer; 2],
    ) -> Result<(MetalBuffer, Option<MetalFault>), MetalError> {
        execute_byte_profile_completed(
            &self.session,
            &self.pipeline,
            inputs,
            self.operation,
            self.bytes,
            self.count,
            self.input_bytes,
            self.fault_law,
        )
    }
    pub(crate) fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        execute_byte_profile_into(
            &self.session,
            &self.pipeline,
            inputs,
            output,
            self.operation,
            self.bytes,
            self.count,
            self.input_bytes,
            self.fault_law,
        )
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

/// Direct exact integer-limb control with the same fresh staging and terminal host publication boundary.
pub struct IntegerControl {
    session: MetalSession,
    map: IntegerMap,
    scalar: PcuScalarType,
    count: usize,
    broadcast: [bool; 2],
}
impl MetalSession {
    /// Compiles a fixed fourteen-width checked integer control without source or neutral IR lowering.
    ///
    /// # Errors
    /// Rejects dtype, extent or unproved operation before work; preserves native compiler errors.
    pub fn prepare_checked_integer_control(
        &self,
        scalar: PcuScalarType,
        operation: MetalIntegerOp,
        range: PcuRangePolicy,
        count: usize,
        broadcast: [bool; 2],
    ) -> Result<IntegerControl, MetalError> {
        Ok(IntegerControl {
            session: self.clone(),
            map: self.prepare_checked_integer_map(scalar, operation, range, count, broadcast)?,
            scalar,
            count,
            broadcast,
        })
    }
}
impl IntegerControl {
    /// Benchmark-only raw device boundary matching a fresh completed host result allocation.
    ///
    /// # Errors
    /// Returns checked arithmetic, extent, session or terminal runtime failure.
    #[cfg(feature = "benchmark-control")]
    pub fn execute_completed_control(
        &self,
        inputs: [&MetalBuffer; 2],
    ) -> Result<(MetalBuffer, Option<MetalFault>), MetalError> {
        self.map.execute_completed(inputs)
    }
    /// Stages fresh initialized typed prefixes, waits for all diagnostics and publishes one host prefix.
    /// Completed Clamp returns an observable recovered arithmetic error with useful borrowed output.
    ///
    /// # Errors
    /// Rejects type/short inputs before work; fatal errors preserve host output, unknown completion quarantines.
    pub fn call<T: fusion_pcu::PcuScalar>(
        &mut self,
        left: &[T],
        right: &[T],
        output: &mut [T],
    ) -> Result<(), MetalError> {
        if T::TYPE != self.scalar {
            return Err(MetalError::Unsupported);
        }
        let counts = self
            .broadcast
            .map(|scalar| if scalar { 1 } else { self.count });
        if left.len() < counts[0] || right.len() < counts[1] || output.len() < self.count {
            return Err(MetalError::InvalidExtent);
        }
        let left = fusion_pcu::PcuHostArgument::read(
            fusion_pcu::PcuBindingRef::new(0, 0),
            &left[..counts[0]],
        );
        let right = fusion_pcu::PcuHostArgument::read(
            fusion_pcu::PcuBindingRef::new(0, 1),
            &right[..counts[1]],
        );
        let a = self.session.upload_bytes(left.bytes())?;
        let b = self.session.upload_bytes(right.bytes())?;
        let (result, fault) = self.map.execute_completed([&a, &b])?;
        let mut output = fusion_pcu::PcuHostArgument::read_write(
            fusion_pcu::PcuBindingRef::new(0, 2),
            &mut output[..self.count],
        );
        result.read_into_bytes(output.bytes_mut().ok_or(MetalError::InvalidExtent)?)?;
        fault.map_or(Ok(()), |fault| Err(MetalError::Arithmetic(fault)))
    }
}
