//! Cold exact encoded graph admission and immutable MLX owner publication.
#[rustfmt::skip]
use std::{
    rc::Rc,
    sync::Arc,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxCheckedProgramInput,
    MlxDiscovery,
    MlxEncodedArray,
    MlxError,
};
#[rustfmt::skip]
use crate::{
    PcuCostBoundary,
    PcuDeviceActivation,
    PcuDeviceIdentity,
    PcuImplementationOffer,
    PcuImplementationRequirements,
    PcuHostArgument,
    PcuScalar,
    dialect::tensor::{
        ValueId,
    },
    global::{
        PcuBackendChoice,
        PcuExecutionError,
        PcuExecutionPolicy,
        arguments::MlxSourceRoot,
        tensor::{
            PcuTensorInput,
            TensorInputKind,
        },
    },
};

pub(super) struct Prepared {
    program: program::Program,
    pub(super) root: Rc<MlxSourceRoot>,
    offer: PcuImplementationOffer,
}

#[path = "program/program.rs"]
mod program;

fn map_error(error: MlxError) -> PcuExecutionError {
    match error {
        MlxError::Arithmetic(fault) => PcuExecutionError::ArithmeticFault(fault),
        other => PcuExecutionError::MlxExecution(other),
    }
}

fn invalid(message: &str) -> PcuExecutionError {
    map_error(MlxError::InvalidRequest(message.into()))
}

// Assess the exact selected closure before deciding which MLX storage family
// owns it. Scalar type alone cannot distinguish bit transport from MatMul.
pub(super) fn assess(
    built: &super::super::capture::PcuCapturedTensorProgram,
    options: PcuExecutionPolicy,
) -> Result<PcuImplementationRequirements, PcuExecutionError> {
    let requirements = requirements(built, options)?;
    program::Plan::assess(&built.program, requirements)?;
    Ok(requirements)
}

fn requirements(
    built: &super::super::capture::PcuCapturedTensorProgram,
    options: PcuExecutionPolicy,
) -> Result<PcuImplementationRequirements, PcuExecutionError> {
    super::numerical::requirements(built, options)
}

#[cfg(test)]
#[path = "producer_tests/producer_tests.rs"]
mod producer_tests;

impl Prepared {
    #[allow(clippy::too_many_lines)] // Exact selected closure, offer and activation are one cold transaction.
    pub(super) fn prepare<T: PcuScalar, const N: usize>(
        built: &super::super::capture::PcuCapturedTensorProgram,
        inputs: &[PcuTensorInput<'_, T>; N],
        options: PcuExecutionPolicy,
    ) -> Result<Self, PcuExecutionError> {
        if !matches!(
            options.backend,
            PcuBackendChoice::Mlx | PcuBackendChoice::Automatic
        ) {
            return Err(PcuExecutionError::ResidentPolicyConflict);
        }
        // Exact backend assessment is the authority for a zero-binding producer graph.
        if built.input_indices.len() > N || built.input_indices.iter().any(|&index| index >= N) {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let requirements = assess(built, options)?;
        let plan = program::Plan::assess(&built.program, requirements)?;
        let mut affinity: Option<Rc<MlxSourceRoot>> = None;
        let mut host_count = 0;
        for &index in &built.input_indices {
            match inputs[index].kind {
                TensorInputKind::Host(_) => host_count += 1,
                TensorInputKind::Resident(owner) => {
                    let root = super::mlx::resident_root(owner)?;
                    if affinity
                        .as_ref()
                        .is_some_and(|prior| !prior.session.same_session(&root.session))
                    {
                        return Err(PcuExecutionError::Argument(
                            crate::global::PcuArgumentError::SessionMismatch,
                        ));
                    }
                    affinity = Some(Rc::clone(root));
                }
            }
        }
        let boundary = if host_count == built.input_indices.len() {
            PcuCostBoundary::HostInputsResidentOutput
        } else if host_count == 0 {
            PcuCostBoundary::Resident
        } else {
            PcuCostBoundary::MixedInputsResidentOutput
        };
        if affinity.is_none() {
            let device = usize::try_from(options.device.unwrap_or(0))
                .map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)?;
            affinity = super::mlx_roots::retained(device)?;
        }
        if affinity.as_ref().is_some_and(|root| {
            options.device.is_some_and(|ordinal| {
                usize::try_from(ordinal).ok() != Some(root.session.facts().index)
            })
        }) {
            return Err(PcuExecutionError::ResidentPolicyConflict);
        }
        let discovery = if affinity.is_some() {
            None
        } else {
            Some(MlxDiscovery::discover_default().map_err(map_error)?)
        };
        let inventory = affinity
            .as_ref()
            .map(|root| &root.discovery)
            .or(discovery.as_ref())
            .ok_or_else(|| invalid("checked MLX has no discovery root"))?;
        let index = affinity.as_ref().map_or_else(
            || {
                usize::try_from(options.device.unwrap_or(0))
                    .map_err(|_| PcuExecutionError::InvalidTensorSourcePlan)
            },
            |root| Ok(root.session.facts().index),
        )?;
        let reference = inventory.device_reference(index).map_err(map_error)?;
        let device = PcuDeviceIdentity::from_device_ref(reference)
            .ok_or_else(|| invalid("MLX discovery did not return a device identity"))?;
        let offer = plan.offer(inventory, device, boundary, requirements, &built.program)?;
        let root = if let Some(root) = affinity {
            root
        } else {
            let discovery =
                discovery.ok_or_else(|| invalid("checked MLX activation lost discovery"))?;
            let session = discovery.open_device(reference).map_err(map_error)?;
            let root = Rc::new(MlxSourceRoot { discovery, session });
            super::mlx_roots::remember(&root)?;
            root
        };
        for &index in &built.input_indices {
            if let TensorInputKind::Resident(owner) = inputs[index].kind {
                // Prime only actual selected immutable views after admission.
                let _ = super::mlx::encoded_input(owner)?;
            }
        }
        let program = plan.prepare(&root.session, Arc::clone(&built.program), requirements)?;
        if program.implementation_id() != Some(offer.implementation) {
            return Err(invalid(
                "checked MLX prepared implementation differs from offer",
            ));
        }
        Ok(Self {
            program,
            root,
            offer,
        })
    }

    pub(super) fn shape(&self) -> Rc<[usize]> {
        self.program.shape_owner()
    }

    pub(super) fn execute<T: PcuScalar, const N: usize>(
        &self,
        inputs: &[PcuTensorInput<'_, T>; N],
        indices: &[usize],
        ids: &[ValueId],
    ) -> Result<MlxEncodedArray, PcuExecutionError> {
        if self.program.implementation_id() != Some(self.offer.implementation) {
            return Err(invalid(
                "retained checked MLX implementation identity changed",
            ));
        }
        if indices.len() != ids.len()
            || indices.len() > N
            || indices.iter().any(|&index| index >= N)
        {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        // One or two unique selected inputs are frozen cold. Borrowed views and
        // host descriptors live on this stack; repeated operands bind only once.
        match (indices, ids) {
            ([], []) => self.program.execute(&[]),
            ([index], [id]) => {
                let input = bind_input(&inputs[*index])?;
                self.program.execute(&[(*id, input.as_input::<T>())])
            }
            ([left, right], [left_id, right_id]) => {
                let left = bind_input(&inputs[*left])?;
                let right = bind_input(&inputs[*right])?;
                self.program.execute(&[
                    (*left_id, left.as_input::<T>()),
                    (*right_id, right.as_input::<T>()),
                ])
            }
            _ => {
                // Bind every selected owner before borrowing descriptors. Four
                // training inputs fit inline; larger signatures may spill.
                let mut bound = smallvec::SmallVec::<[_; 4]>::new();
                for &index in indices {
                    bound.push(bind_input(&inputs[index])?);
                }
                let bindings: smallvec::SmallVec<[_; 4]> = ids
                    .iter()
                    .copied()
                    .zip(bound.iter().map(BoundInput::as_input::<T>))
                    .collect();
                self.program.execute(&bindings)
            }
        }
    }
}

enum BoundInput<'a> {
    Host(PcuHostArgument<'a>),
    Resident(super::mlx::EncodedInput<'a>),
}

impl BoundInput<'_> {
    const fn as_input<T: PcuScalar>(&self) -> MlxCheckedProgramInput<'_> {
        match self {
            Self::Host(host) => MlxCheckedProgramInput::Host {
                scalar: T::TYPE,
                bytes: host.bytes(),
            },
            Self::Resident(input) => MlxCheckedProgramInput::Resident(input.array()),
        }
    }
}

fn bind_input<'a, T: PcuScalar>(
    input: &PcuTensorInput<'a, T>,
) -> Result<BoundInput<'a>, PcuExecutionError> {
    match input.kind {
        TensorInputKind::Host(values) => Ok(BoundInput::Host(PcuHostArgument::read(
            crate::PcuBindingRef::new(0, 0),
            values,
        ))),
        TensorInputKind::Resident(owner) => {
            super::mlx::encoded_input(owner).map(BoundInput::Resident)
        }
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
