//! Packed private primitive quotient/remainder using only defined U32 limb operations.
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuScalarType,
    PcuHostArgument,
    PcuBindingRef,
};
#[rustfmt::skip]
use super::{
    MetalSession,
    MetalBuffer,
    MetalError,
    ffi,
};
/// Retained exact primitive `DivRem` control. This is not a generic dispatch admission claim.
pub struct MetalPreparedDivRemControl {
    fault_law: fusion_pcu::PcuCheckedScalarFaultLaw,
    session: MetalSession,
    pipeline: ffi::Pipeline,
    scalar: PcuScalarType,
    count: usize,
    bytes: usize,
    input_bytes: [usize; 2],
}
impl MetalSession {
    /// Freezes exact checked primitive integer quotient/remainder and private packed output.
    /// Both outputs publish transactionally only after the complete terminal fault scan.
    ///
    /// # Errors
    /// Rejects unsupported scalar, zero/overflowing extent before shader compilation.
    /// Returns native compiler or session errors; no fallback evaluates this operation.
    pub fn prepare_checked_div_rem_control(
        &self,
        scalar: PcuScalarType,
        count: usize,
    ) -> Result<MetalPreparedDivRemControl, MetalError> {
        self.prepare_integer_div_rem_control(scalar, count, [count; 2], [false; 2])
    }
    /// Freezes bounded mathematical input spans and independent scalar reads.
    ///
    /// Each input span is either one element for a scalar operand or the full logical count.
    /// Shared inputs may retain a full span even when one operand reads element zero.
    /// The native session, integer kernel and broadcast indexing are retained during preparation.
    ///
    /// # Errors
    /// Rejects unsupported types, invalid spans/byte extents and native compilation errors.
    pub fn prepare_checked_div_rem_role_control(
        &self,
        scalar: PcuScalarType,
        count: usize,
        input_counts: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<MetalPreparedDivRemControl, MetalError> {
        self.prepare_integer_div_rem_control(scalar, count, input_counts, broadcast)
    }
    #[cfg(test)]
    fn prepare_wide_div_rem_proof(
        &self,
        scalar: PcuScalarType,
        count: usize,
    ) -> Result<MetalPreparedDivRemControl, MetalError> {
        self.prepare_integer_div_rem_control(scalar, count, [count; 2], [false; 2])
    }
    fn prepare_integer_div_rem_control(
        &self,
        scalar: PcuScalarType,
        count: usize,
        input_counts: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<MetalPreparedDivRemControl, MetalError> {
        for (extent, scalar_read) in input_counts.into_iter().zip(broadcast) {
            if extent != count && !(scalar_read && extent == 1) {
                return Err(MetalError::InvalidExtent);
            }
        }
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
        let bits = scalar.bit_width();
        let bytes = count
            .checked_mul(usize::from(bits) / 8)
            .ok_or(MetalError::InvalidExtent)?;
        let packed = bytes.checked_mul(2).ok_or(MetalError::InvalidExtent)?;
        if count == 0 || u32::try_from(packed).is_err() {
            return Err(MetalError::InvalidExtent);
        }
        super::validate_byte_extent(packed, self.0.facts.max_buffer_bytes)?;
        super::validate_extent(count, self.0.facts.max_buffer_bytes)?;
        self.ensure_quiescent()?;
        let mask = if bits < 32 {
            (1_u32 << bits) - 1
        } else {
            u32::MAX
        };
        let sign = 1_u32 << if bits < 32 { bits - 1 } else { 31 };
        let source = format!(
            "#define PCU_WIDTH {bits}u\n#define PCU_SIGNED {}u\n#define PCU_MASK {mask}u\n#define PCU_SIGN {sign}u\n#define PCU_LEFT_SCALAR {}u\n#define PCU_RIGHT_SCALAR {}u\n{}",
            u32::from(signed),
            u32::from(broadcast[0]),
            u32::from(broadcast[1]),
            include_str!("shader/checked_div_rem.metal")
        );
        let pipeline = self.0.native.compile(&source, "pcu_checked_div_rem")?;
        Ok(MetalPreparedDivRemControl {
            fault_law: fusion_pcu::PcuCheckedScalarFaultLaw::integer_div_rem(scalar)
                .ok_or(MetalError::Unsupported)?,
            session: self.clone(),
            pipeline,
            scalar,
            count,
            bytes,
            input_bytes: input_counts.map(|extent| extent * (usize::from(bits) / 8)),
        })
    }
}
impl MetalPreparedDivRemControl {
    pub(crate) const fn session(&self) -> &MetalSession {
        &self.session
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.count
    }
    /// Stages each actual unique input once and resolves mathematical operands by slot.
    ///
    /// The caller supplies one or two actually used resources. Both destinations and every
    /// read span preflight before staging. This direct primitive control performs no IR lowering.
    ///
    /// # Errors
    /// Rejects mismatched types, nonexistent or unused slots, short inputs/outputs, arithmetic
    /// domain faults and native failures. Fatal arithmetic preserves both host destinations.
    pub fn call_unique<T: PcuScalar>(
        &mut self,
        inputs: &[&[T]],
        operand_inputs: [usize; 2],
        quotient: &mut [T],
        remainder: &mut [T],
    ) -> Result<(), MetalError> {
        if T::TYPE != self.scalar || inputs.is_empty() || inputs.len() > 2 {
            return Err(MetalError::Unsupported);
        }
        if quotient.len() < self.count || remainder.len() < self.count {
            return Err(MetalError::InvalidExtent);
        }
        let mut required = [0; 2];
        for (operand, slot) in operand_inputs.into_iter().enumerate() {
            if slot >= inputs.len() {
                return Err(MetalError::Unsupported);
            }
            required[slot] = required[slot].max(self.input_bytes[operand]);
        }
        let width = self.bytes / self.count;
        for (slot, input) in inputs.iter().enumerate() {
            if required[slot] == 0 || input.len() < required[slot] / width {
                return Err(MetalError::InvalidExtent);
            }
        }
        let mut staged = [None, None];
        for (slot, input) in inputs.iter().enumerate() {
            let argument = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
            staged[slot] = Some(
                self.session
                    .upload_bytes(&argument.bytes()[..required[slot]])?,
            );
        }
        let get = |slot: usize| staged[slot].as_ref().ok_or(MetalError::Unsupported);
        let output = self.execute([get(operand_inputs[0])?, get(operand_inputs[1])?])?;
        let mut q =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut quotient[..self.count]);
        let mut r =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut remainder[..self.count]);
        output.read_pair_bytes(
            q.bytes_mut().ok_or(MetalError::InvalidExtent)?,
            r.bytes_mut().ok_or(MetalError::InvalidExtent)?,
        )
    }
    pub(crate) fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        self.session.ensure_quiescent()?;
        for (input, required) in inputs.into_iter().zip(self.input_bytes) {
            if !self.session.same_session(input.session()) {
                return Err(MetalError::ForeignSession);
            }
            if input.byte_len() < required {
                return Err(MetalError::InvalidExtent);
            }
        }
        let output = self.session.allocate_zeroed_bytes(self.bytes * 2)?;
        let records = self.session.0.native.allocate(self.count * 4)?;
        records.fill_ones();
        self.session.0.native.execute(
            &self.pipeline,
            [
                &inputs[0].native,
                &inputs[1].native,
                &output.native,
                &records,
            ],
            [
                u32::try_from(self.count).map_err(|_| MetalError::InvalidExtent)?,
                0,
            ],
            self.count,
        )?;
        records.inspect_words(self.count, |records| {
            super::fault::validate(
                records,
                self.count,
                Some(self.fault_law),
                super::fault::Encoding::DivRem,
            )?;
            select_div_rem_fault(records)
        })?;
        Ok(output)
    }
    pub(crate) fn call_bytes(
        &self,
        left: &[u8],
        right: &[u8],
        quotient: &mut [u8],
        remainder: &mut [u8],
    ) -> Result<(), MetalError> {
        if left.len() < self.input_bytes[0]
            || right.len() < self.input_bytes[1]
            || quotient.len() < self.bytes
            || remainder.len() < self.bytes
        {
            return Err(MetalError::InvalidExtent);
        }
        let a = self.session.upload_bytes(&left[..self.input_bytes[0]])?;
        let b = self.session.upload_bytes(&right[..self.input_bytes[1]])?;
        let output = self.execute([&a, &b])?;
        output.read_pair_bytes(&mut quotient[..self.bytes], &mut remainder[..self.bytes])
    }
    /// Stages current inputs and jointly publishes exact quotient/remainder prefixes.
    /// Both destinations preflight before work; any fatal lane preserves both destinations.
    ///
    /// # Errors
    /// Returns exact tag/extent, checked zero-divisor or signed MIN/-1 faults, or native failure.
    pub fn call<T: PcuScalar>(
        &mut self,
        left: &[T],
        right: &[T],
        quotient: &mut [T],
        remainder: &mut [T],
    ) -> Result<(), MetalError> {
        if T::TYPE != self.scalar {
            return Err(MetalError::Unsupported);
        }
        let width = self.bytes / self.count;
        let counts = self.input_bytes.map(|bytes| bytes / width);
        if left.len() < counts[0]
            || right.len() < counts[1]
            || quotient.len() < self.count
            || remainder.len() < self.count
        {
            return Err(MetalError::InvalidExtent);
        }
        let a = PcuHostArgument::read(PcuBindingRef::new(0, 0), &left[..counts[0]]);
        let b = PcuHostArgument::read(PcuBindingRef::new(0, 1), &right[..counts[1]]);
        let left = self.session.upload_bytes(a.bytes())?;
        let right = self.session.upload_bytes(b.bytes())?;
        let output = self.execute([&left, &right])?;
        let mut q =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut quotient[..self.count]);
        let mut r =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut remainder[..self.count]);
        output.read_pair_bytes(
            q.bytes_mut().ok_or(MetalError::InvalidExtent)?,
            r.bytes_mut().ok_or(MetalError::InvalidExtent)?,
        )
    }
}

fn select_div_rem_fault(records: &[u32]) -> Result<(), MetalError> {
    if records.iter().any(|record| !matches!(record, 0 | 2 | 5)) {
        return Err(MetalError::Runtime("invalid checked DivRem record".into()));
    }
    if let Some(index) = records.iter().position(|record| *record != 0) {
        return Err(MetalError::Arithmetic(super::MetalFault {
            invocation_id: u64::try_from(index).map_err(|_| MetalError::InvalidExtent)?,
            recovered: false,
            kind: if records[index] == 2 {
                fusion_pcu::PcuExecutionFaultKind::DivideByZero
            } else {
                fusion_pcu::PcuExecutionFaultKind::SignedDivisionOverflow
            },
        }));
    }
    Ok(())
}
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

#[path = "publication/publication.rs"]
pub mod publication;
