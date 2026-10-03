//! Cold neutral admission retains the exact discovery, session, offer and executable.

#[rustfmt::skip]
use super::{
    PcuArgumentError,
    PcuBackendChoice,
    PcuExecutionError,
    PcuScalar,
    PcuTensorInput,
    TensorInputKind,
};
#[rustfmt::skip]
use crate::{
    PcuCostBoundary,
    PcuDeviceActivation,
    PcuDeviceIdentity,
    PcuImplementationOffer,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    global::{PcuExecutionPolicy, arguments::MlxSourceRoot},
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MLX_EXECUTOR,
    MlxArray,
    MlxDiscovery,
    MlxError,
    MlxMatmulPlan,
    MlxMatmulRequest,
    MlxPreparedProgram,
    MlxProgramInput,
};
#[rustfmt::skip]
use std::{
    rc::Rc,
    sync::Arc,
};

pub(super) struct Prepared {
    program: MlxPreparedProgram,
    pub(super) root: Rc<MlxSourceRoot>,
    offer: PcuImplementationOffer,
}

#[path = "view/view.rs"]
mod view;
#[rustfmt::skip]
pub(super) use view::{
    EncodedInput,
    bind_input,
    encoded_input,
    resident_root,
};

fn invalid(message: &str) -> PcuExecutionError {
    PcuExecutionError::MlxExecution(MlxError::InvalidRequest(message.into()))
}

impl Prepared {
    #[allow(clippy::too_many_lines)] // Preflight, neutral offer and activation form one ordered cold transaction.
    pub(super) fn prepare<T: PcuScalar, const N: usize>(
        built: &super::super::capture::PcuCapturedTensorProgram,
        inputs: &[PcuTensorInput<'_, T>; N],
        options: PcuExecutionPolicy,
    ) -> Result<Self, PcuExecutionError> {
        MlxMatmulPlan::assess_program(&built.program)
            .map_err(|reason| PcuExecutionError::MlxExecution(MlxError::Unsupported(reason)))?;
        if !matches!(
            options.backend,
            PcuBackendChoice::Mlx | PcuBackendChoice::Automatic
        ) {
            return Err(PcuExecutionError::ResidentPolicyConflict);
        }
        let mut affinity: Option<Rc<MlxSourceRoot>> = None;
        let mut host_count = 0;
        for &index in &built.input_indices {
            match inputs[index].kind {
                TensorInputKind::Host(_) => host_count += 1,
                TensorInputKind::Resident(owner) => {
                    let root = resident_root(owner)?;
                    if affinity
                        .as_ref()
                        .is_some_and(|prior| !prior.session.same_session(&root.session))
                    {
                        return Err(PcuExecutionError::Argument(
                            PcuArgumentError::SessionMismatch,
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
            Some(MlxDiscovery::discover_default().map_err(PcuExecutionError::MlxExecution)?)
        };
        let inventory = affinity
            .as_ref()
            .map(|root| &root.discovery)
            .or(discovery.as_ref())
            .ok_or_else(|| invalid("MLX source has no retained discovery root"))?;
        let index = affinity.as_ref().map_or_else(
            || options.device.unwrap_or(0) as usize,
            |root| root.session.facts().index,
        );
        let reference = inventory.device_reference(index);
        let reference = reference.map_err(PcuExecutionError::MlxExecution)?;
        let device = PcuDeviceIdentity::from_device_ref(reference)
            .ok_or_else(|| invalid("MLX discovery did not return a device identity"))?;
        let graph = built.program.graph();
        let node = graph
            .node(built.program.output_values()[0])
            .map_err(super::super::super::tensor_build_error)?;
        let operation = MlxMatmulRequest { graph, node };
        let request = PcuImplementationRequest {
            device,
            executor: MLX_EXECUTOR,
            requirements: PcuImplementationRequirements {
                numerical_mode: node
                    .numerical_mode
                    .ok_or_else(|| invalid("MLX source lacks numerical mode"))?,
                numerical_options: node.numerical_options,
                float_underflow: node
                    .float_underflow_policy
                    .ok_or_else(|| invalid("MLX source lacks underflow policy"))?,
                range_policy: options.range_policy,
            },
            boundary,
            operation: &operation,
        };
        let mut offers = [None];
        let count = inventory
            .implementation_offers(&request, &mut offers)
            .map_err(PcuExecutionError::MlxExecution)?;
        if count != 1 {
            return Err(invalid(
                "bounded MLX source requires one exact implementation offer",
            ));
        }
        let offer = offers[0].ok_or_else(|| invalid("MLX omitted its admitted offer"))?;
        offer
            .validate_request(&request)
            .map_err(|_| invalid("MLX offer envelope changed"))?;
        let root = if let Some(root) = affinity {
            root
        } else {
            let discovery =
                discovery.ok_or_else(|| invalid("MLX activation lost its discovery root"))?;
            let session = discovery
                .open_device(reference)
                .map_err(PcuExecutionError::MlxExecution)?;
            let root = Rc::new(MlxSourceRoot { discovery, session });
            super::mlx_roots::remember(&root)?;
            root
        };
        // Prime views only after exact profile/affinity admission. The resulting
        // immutable descriptor is cached on its owner, never rebuilt per call.
        for &index in &built.input_indices {
            let _ = bind_input(&inputs[index])?;
        }
        let program = root
            .session
            .prepare_program(Arc::clone(&built.program))
            .map_err(PcuExecutionError::MlxExecution)?;
        if program.matmul().implementation_id() != Some(offer.implementation) {
            return Err(invalid(
                "prepared MLX implementation differs from its admitted offer",
            ));
        }
        Ok(Self {
            program,
            root,
            offer,
        })
    }

    pub(super) fn execute<T: PcuScalar>(
        &self,
        inputs: &[(crate::dialect::tensor::ValueId, MlxProgramInput<'_, T>)],
    ) -> Result<MlxArray, PcuExecutionError> {
        // Retained identity is checked without warm discovery or a new cost estimate.
        if self.program.matmul().implementation_id() != Some(self.offer.implementation) {
            return Err(invalid("retained MLX implementation identity changed"));
        }
        self.program
            .execute_mixed(inputs)
            .map_err(PcuExecutionError::MlxExecution)
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
