//! Frozen selected source replay through distinct encoded ownership and MLX's checked primitive.
#[rustfmt::skip]
use std::{
    rc::Rc,
    sync::Arc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuScalar,
    PcuScalarType,
    PcuHostArgument,
    PcuBindingRef,
    PcuImplementationRequirements,
    PcuImplementationId,
    PcuDispatchFloatUnaryOp,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    TensorOwnedSelectedProgram,
    ValueId,
};
#[rustfmt::skip]
use crate::{
    ffi,
    MlxSession,
    MlxError,
    MlxEncodedArray,
    MlxCheckedTensorPlan,
};
/// Current initialized host bytes or a retained immutable exact-session logical array.
pub enum MlxCheckedProgramInput<'a> {
    Host {
        scalar: PcuScalarType,
        bytes: &'a [u8],
    },
    Resident(&'a MlxEncodedArray),
}
/// Backend-owned authentic selected program, frozen policy/shape and retained actual MLX session.
pub struct MlxPreparedCheckedProgram {
    program: Arc<TensorOwnedSelectedProgram>,
    session: MlxSession,
    plan: MlxCheckedTensorPlan,
    relu: Option<Rc<ffi::CheckedUnary>>,
}
impl MlxSession {
    /// Prepares a selected byte-aligned Identity/six-format `ReLU` source closure without rewriting its effects.
    ///
    /// # Errors
    /// Returns exact cold contract/shape/topology rejection or checked native compilation failure.
    pub fn prepare_checked_program(
        &self,
        program: Arc<TensorOwnedSelectedProgram>,
        requirements: PcuImplementationRequirements,
    ) -> Result<MlxPreparedCheckedProgram, MlxError> {
        let plan = MlxCheckedTensorPlan::assess_program(&program, requirements)
            .map_err(MlxError::Unsupported)?;
        self.validate_access_available()?;
        let relu = if plan.relu_effect().is_some() {
            Some(Rc::new(self.prepare_checked_unary_native(
                plan.scalar_type(),
                PcuDispatchFloatUnaryOp::Relu,
                requirements.float_underflow,
                requirements.range_policy,
                plan.element_count(),
                false,
            )?))
        } else {
            None
        };
        Ok(MlxPreparedCheckedProgram {
            program,
            session: self.clone(),
            plan,
            relu,
        })
    }
}
impl MlxPreparedCheckedProgram {
    #[must_use]
    pub fn program(&self) -> &TensorOwnedSelectedProgram {
        &self.program
    }
    #[must_use]
    pub const fn plan(&self) -> &MlxCheckedTensorPlan {
        &self.plan
    }
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        &self.session
    }
    #[must_use]
    pub fn implementation_id(&self) -> Option<PcuImplementationId> {
        self.session
            .identity()
            .map(|device| self.plan.implementation_id(device))
    }
    /// Executes current authentic input binding; no warm capture, discovery or recompilation.
    /// An unused checked effect is evaluated before returning the retained identity owner.
    ///
    /// # Errors
    /// Returns input identity/type/extent/affinity/quarantine or native/checked fatal failure.
    pub fn execute_mixed(
        &self,
        inputs: &[(ValueId, MlxCheckedProgramInput<'_>)],
    ) -> Result<MlxEncodedArray, MlxError> {
        let [(value, input)] = inputs else {
            return Err(MlxError::InvalidRequest(
                "checked MLX graph requires one input binding".into(),
            ));
        };
        if *value != self.plan.input() {
            return Err(MlxError::InvalidRequest(
                "checked MLX graph input identity mismatch".into(),
            ));
        }
        self.session.validate_access_available()?;
        let count = self.plan.element_count();
        let scalar = self.plan.scalar_type();
        let width = usize::from(scalar.bit_width()) / 8;
        let staged = match input {
            MlxCheckedProgramInput::Host {
                scalar: actual,
                bytes,
            } => {
                if *actual != scalar {
                    return Err(MlxError::UnsupportedScalar(*actual));
                }
                if bytes.len() != count * width {
                    return Err(MlxError::InvalidExtent);
                }
                self.session.upload_encoded_bytes(scalar, count, bytes)?
            }
            MlxCheckedProgramInput::Resident(owner) => {
                if owner.scalar_type() != scalar {
                    return Err(MlxError::UnsupportedScalar(owner.scalar_type()));
                }
                if owner.element_count() != count {
                    return Err(MlxError::InvalidExtent);
                }
                if !owner.same_session(&self.session) {
                    return Err(MlxError::ForeignSession);
                }
                owner.validate_access_available()?;
                (*owner).clone()
            }
        };
        if let Some(relu) = &self.relu {
            let (output, recovered) = relu.execute_encoded(staged.native())?;
            if let Some(fault) = recovered {
                return Err(MlxError::Arithmetic(fault));
            }
            if self.plan.output()
                == self.plan.relu_effect().ok_or_else(|| {
                    MlxError::InvalidRequest("checked effect provenance missing".into())
                })?
            {
                return Ok(self.session.wrap_encoded(output));
            }
            // The unused checked operation still evaluated/status-scanned above. Its private
            // completed payload drops before an immutable identity alias escapes.
            drop(output);
        }
        Ok(staged)
    }
    /// Typed one-input convenience retaining the same exact source identity and byte preflight.
    ///
    /// # Errors
    /// Returns logical type/extent, graph binding or native/checked fatal failure.
    pub fn execute_host<T: PcuScalar>(&self, input: &[T]) -> Result<MlxEncodedArray, MlxError> {
        let argument = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
        self.execute_mixed(&[(
            self.plan.input(),
            MlxCheckedProgramInput::Host {
                scalar: argument.scalar(),
                bytes: argument.bytes(),
            },
        )])
    }
}
