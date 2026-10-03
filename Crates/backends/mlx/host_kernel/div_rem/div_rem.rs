//! Exact MLX-owned quotient/remainder control; mutable host publication is joint.
#[path = "roles/roles.rs"]
mod roles;
#[rustfmt::skip]
pub use roles::{
    MlxCheckedDivRemRolePlan,
    MlxCheckedDivRemRoleBackend,
    MlxPreparedDivRemRoleHostKernel,
};
#[path = "admission/admission.rs"]
mod admission;
#[path = "plan/plan.rs"]
mod plan;
#[rustfmt::skip]
pub use {
    plan::MlxCheckedDivRemPlan,
    admission::{
        MlxCheckedDivRemBackend,
        MlxPreparedDivRemHostKernel,
    },
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuScalarType,
    PcuHostArgument,
    PcuBindingRef,
};
#[rustfmt::skip]
use crate::{
    ffi,
    MlxSession,
    MlxEncodedArray,
    MlxError,
};
/// Retained custom primitive with separate actual quotient/remainder/status siblings.
/// Creating this control alone does not establish a generic implementation offer.
pub struct MlxCheckedDivRemControl {
    native: ffi::CheckedDivRem,
    session: MlxSession,
    scalar: PcuScalarType,
    count: usize,
    inputs: [usize; 2],
}
impl MlxSession {
    /// Prepares exact fourteen-width integer division with explicit retained input extents.
    ///
    /// # Errors
    /// Rejects unsupported logical width/span/session and native compile/completion failures.
    pub fn prepare_checked_div_rem_control(
        &self,
        scalar: PcuScalarType,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<MlxCheckedDivRemControl, MlxError> {
        Ok(MlxCheckedDivRemControl {
            native: self.prepare_checked_div_rem_native(scalar, count, inputs, broadcast)?,
            session: self.clone(),
            scalar,
            count,
            inputs,
        })
    }
    /// Freezes full dense input capacities while shader indexes remain bounded to `count` or zero.
    ///
    /// This separate constructor primes the retained primitive with those exact full shapes.
    /// It neither slices nor reads back a resident owner; subsequent calls must match full capacities.
    ///
    /// # Errors
    /// Rejects capacities shorter than actual read obligations, unsupported scalar/physical spans,
    /// invalid sessions and native compile/completion failures.
    pub fn prepare_checked_div_rem_control_with_input_extents(
        &self,
        scalar: PcuScalarType,
        count: usize,
        inputs: [usize; 2],
        broadcast: [bool; 2],
    ) -> Result<MlxCheckedDivRemControl, MlxError> {
        Ok(MlxCheckedDivRemControl {
            native: self.prepare_checked_div_rem_native_with_input_extents(
                scalar, count, inputs, broadcast,
            )?,
            session: self.clone(),
            scalar,
            count,
            inputs,
        })
    }
}
impl MlxCheckedDivRemControl {
    /// Native-control host boundary with one upload per actual unique input.
    ///
    /// The retained primitive already freezes mathematical input extents and broadcast roles.
    /// This call supplies only the unique caller resources and their operand-slot mapping;
    /// it does not construct IR, discover a provider or infer any implementation offer.
    ///
    /// # Errors
    /// Rejects invalid slot/count/type/span before staging. Fatal checked/native work preserves
    /// both destinations; both private reads and checked cleanup precede joint publication.
    pub fn call_input_roles<T: PcuScalar>(
        &mut self,
        inputs: &[&[T]],
        counts: &[usize],
        operands: [usize; 2],
        outputs: [&mut [T]; 2],
    ) -> Result<(), MlxError> {
        self.native.reset_write_fact();
        if T::TYPE != self.scalar {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
        }
        if !(1..=2).contains(&inputs.len())
            || counts.len() != inputs.len()
            || inputs
                .iter()
                .zip(counts)
                .any(|(input, &count)| input.len() < count)
            || outputs.iter().any(|output| output.len() < self.count)
        {
            return Err(MlxError::InvalidExtent);
        }
        let args = std::array::from_fn::<_, 2, _>(|slot| {
            (slot < inputs.len()).then(|| {
                PcuHostArgument::read(
                    PcuBindingRef::new(0, u32::from(slot != 0)),
                    &inputs[slot][..counts[slot]],
                )
            })
        });
        let bytes = std::array::from_fn::<_, 2, _>(|slot| {
            args[slot].as_ref().map_or(&[][..], PcuHostArgument::bytes)
        });
        let [quotient, remainder] = outputs;
        let mut q =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut quotient[..self.count]);
        let mut r =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut remainder[..self.count]);
        self.native.execute_host_roles(
            &bytes[..inputs.len()],
            counts,
            operands,
            [
                q.bytes_mut().ok_or(MlxError::InvalidExtent)?,
                r.bytes_mut().ok_or(MlxError::InvalidExtent)?,
            ],
        )
    }
    /// Borrows actual same-session inputs into two completed immutable new owners.
    ///
    /// # Errors
    /// Preflights both inputs; fatal checked/native work publishes neither new owner.
    pub fn execute_resident(
        &mut self,
        inputs: [&MlxEncodedArray; 2],
    ) -> Result<[MlxEncodedArray; 2], MlxError> {
        self.native.reset_write_fact();
        for (input, count) in inputs.iter().zip(self.inputs) {
            if input.scalar_type() != self.scalar {
                return Err(MlxError::UnsupportedScalar(input.scalar_type()));
            }
            if input.element_count() != count {
                return Err(MlxError::InvalidExtent);
            }
            if !input.same_session(&self.session) {
                return Err(MlxError::ForeignSession);
            }
            input.validate_access_available()?;
        }
        let [quotient, remainder] = self
            .native
            .execute_encoded(inputs.map(MlxEncodedArray::native))?;
        Ok([
            self.session.wrap_encoded(quotient),
            self.session.wrap_encoded(remainder),
        ])
    }
    /// Stages both typed prefixes once and returns a completed immutable output pair.
    ///
    /// # Errors
    /// Rejects unsupported scalar/span before staging; reports fatal arithmetic/native work.
    pub fn execute_encoded<T: PcuScalar>(
        &mut self,
        inputs: [&[T]; 2],
    ) -> Result<[MlxEncodedArray; 2], MlxError> {
        self.native.reset_write_fact();
        if T::TYPE != self.scalar {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
        }
        if inputs
            .iter()
            .zip(self.inputs)
            .any(|(input, count)| input.len() < count)
        {
            return Err(MlxError::InvalidExtent);
        }
        let left = self.session.upload_encoded(&inputs[0][..self.inputs[0]])?;
        let right = self.session.upload_encoded(&inputs[1][..self.inputs[1]])?;
        let result = self.execute_resident([&left, &right]);
        if self.last_call_completion_uncertain() {
            return result;
        }
        left.release()?;
        right.release()?;
        result
    }
    /// Preflights and publishes both host prefixes jointly after both private reads and cleanup.
    ///
    /// # Errors
    /// Returns scalar/span/checked/native failures with both previous destinations preserved.
    pub fn call<T: PcuScalar>(
        &mut self,
        inputs: [&[T]; 2],
        outputs: [&mut [T]; 2],
    ) -> Result<(), MlxError> {
        self.native.reset_write_fact();
        if T::TYPE != self.scalar {
            return Err(MlxError::UnsupportedScalar(T::TYPE));
        }
        if inputs
            .iter()
            .zip(self.inputs)
            .any(|(input, count)| input.len() < count)
            || outputs.iter().any(|output| output.len() < self.count)
        {
            return Err(MlxError::InvalidExtent);
        }
        let left = PcuHostArgument::read(PcuBindingRef::new(0, 0), &inputs[0][..self.inputs[0]]);
        let right = PcuHostArgument::read(PcuBindingRef::new(0, 1), &inputs[1][..self.inputs[1]]);
        let [quotient, remainder] = outputs;
        let mut q =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut quotient[..self.count]);
        let mut r =
            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut remainder[..self.count]);
        self.native.execute_host(
            [left.bytes(), right.bytes()],
            [
                q.bytes_mut().ok_or(MlxError::InvalidExtent)?,
                r.bytes_mut().ok_or(MlxError::InvalidExtent)?,
            ],
        )
    }
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.native.may_have_written()
    }
    /// These outputs replace immutable owners only after success; no old owner is written.
    #[must_use]
    pub const fn last_call_may_have_written_existing_encoded_owner(&self) -> bool {
        false
    }
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.native.completion_uncertain()
    }
}
