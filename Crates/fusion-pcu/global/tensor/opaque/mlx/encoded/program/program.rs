//! Concrete cold admission and retained execution for encoded MLX storage.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MLX_EXECUTOR,
    MlxCheckedProgramInput,
    MlxCheckedTensorBinaryPlan,
    MlxCheckedTensorIntegerPlan,
    MlxCheckedTensorPlan,
    MlxCheckedTensorRequest,
    MlxDiscovery,
    MlxEncodedArray,
    MlxError,
    MlxPreparedCheckedProgram,
    MlxPreparedTensorBinaryProgram,
    MlxPreparedTensorIntegerProgram,
    MlxSession,
    MlxTensorBinaryRequest,
    MlxTensorIntegerRequest,
};
#[rustfmt::skip]
use crate::{
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuImplementationId,
    PcuImplementationOffer,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    dialect::tensor::{
        TensorOwnedSelectedProgram,
        ValueId,
    },
    global::PcuExecutionError,
};
#[rustfmt::skip]
use std::{
    rc::Rc,
    sync::Arc,
};

pub(super) enum Plan {
    Leaf,
    Binary,
    Integer,
}

impl Plan {
    pub(super) fn assess(
        program: &TensorOwnedSelectedProgram,
        requirements: PcuImplementationRequirements,
    ) -> Result<Self, PcuExecutionError> {
        if MlxCheckedTensorPlan::assess_program(program, requirements).is_ok() {
            return Ok(Self::Leaf);
        }
        if MlxCheckedTensorBinaryPlan::assess_program(program, requirements).is_ok() {
            return Ok(Self::Binary);
        }
        MlxCheckedTensorIntegerPlan::assess_program(program, requirements)
            .map(|_| Self::Integer)
            .map_err(|reason| super::map_error(MlxError::Unsupported(reason)))
    }

    pub(super) fn offer(
        &self,
        inventory: &MlxDiscovery,
        device: PcuDeviceIdentity,
        boundary: PcuCostBoundary,
        requirements: PcuImplementationRequirements,
        program: &TensorOwnedSelectedProgram,
    ) -> Result<PcuImplementationOffer, PcuExecutionError> {
        match self {
            Self::Leaf => offered(
                inventory,
                device,
                boundary,
                requirements,
                &MlxCheckedTensorRequest { program },
            ),
            Self::Binary => offered(
                inventory,
                device,
                boundary,
                requirements,
                &MlxTensorBinaryRequest { program },
            ),
            Self::Integer => offered(
                inventory,
                device,
                boundary,
                requirements,
                &MlxTensorIntegerRequest { program },
            ),
        }
    }

    pub(super) fn prepare(
        self,
        session: &MlxSession,
        program: Arc<TensorOwnedSelectedProgram>,
        requirements: PcuImplementationRequirements,
    ) -> Result<Program, PcuExecutionError> {
        match self {
            Self::Leaf => session
                .prepare_checked_program(program, requirements)
                .map(Program::Leaf),
            Self::Binary => session
                .prepare_tensor_binary_program(program, requirements)
                .map(Program::Binary),
            Self::Integer => session
                .prepare_tensor_integer_program(program, requirements)
                .map(Program::Integer),
        }
        .map_err(super::map_error)
    }
}

fn offered<O>(
    inventory: &MlxDiscovery,
    device: PcuDeviceIdentity,
    boundary: PcuCostBoundary,
    requirements: PcuImplementationRequirements,
    operation: &O,
) -> Result<PcuImplementationOffer, PcuExecutionError>
where
    MlxDiscovery: PcuImplementationOffers<O, Error = MlxError>,
{
    let request = PcuImplementationRequest {
        device,
        executor: MLX_EXECUTOR,
        requirements,
        boundary,
        operation,
    };
    let mut offers = [None];
    if inventory
        .implementation_offers(&request, &mut offers)
        .map_err(super::map_error)?
        != 1
    {
        return Err(super::invalid(
            "checked MLX requires one exact implementation offer",
        ));
    }
    let offer =
        offers[0].ok_or_else(|| super::invalid("checked MLX omitted its admitted offer"))?;
    offer
        .validate_request(&request)
        .map_err(|_| super::invalid("checked MLX offer envelope changed"))?;
    Ok(offer)
}

pub(super) enum Program {
    Leaf(MlxPreparedCheckedProgram),
    Binary(MlxPreparedTensorBinaryProgram),
    Integer(MlxPreparedTensorIntegerProgram),
}

impl Program {
    pub(super) fn implementation_id(&self) -> Option<PcuImplementationId> {
        match self {
            Self::Leaf(program) => program.implementation_id(),
            Self::Binary(program) => program.implementation_id(),
            Self::Integer(program) => program.implementation_id(),
        }
    }

    pub(super) fn shape_owner(&self) -> Rc<[usize]> {
        match self {
            Self::Leaf(program) => program.plan().shape_owner(),
            Self::Binary(program) => program.plan().shape_owner(),
            Self::Integer(program) => program.plan().shape_owner(),
        }
    }

    pub(super) fn execute(
        &self,
        inputs: &[(ValueId, MlxCheckedProgramInput<'_>)],
    ) -> Result<MlxEncodedArray, PcuExecutionError> {
        match self {
            Self::Leaf(program) => program.execute_mixed(inputs),
            Self::Binary(program) => program.execute_mixed(inputs),
            Self::Integer(program) => program.execute_mixed(inputs),
        }
        .map_err(super::map_error)
    }
}
