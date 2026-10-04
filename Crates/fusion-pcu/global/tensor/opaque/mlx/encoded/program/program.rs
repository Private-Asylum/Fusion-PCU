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
    MlxSelectedNumericalTensorPlan,
    MlxSelectedNumericalTensorRequest,
    MlxPreparedSelectedNumericalTensorProgram,
    MlxSelectedTensorGraphPlan,
    MlxPreparedSelectedTensorGraph,
    MlxSelectedTensorGraphRequest,
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
    Numerical(MlxSelectedNumericalTensorPlan),
    Graph(MlxSelectedTensorGraphPlan),
}

impl Plan {
    pub(super) fn assess(
        program: &Arc<TensorOwnedSelectedProgram>,
        requirements: PcuImplementationRequirements,
    ) -> Result<Self, PcuExecutionError> {
        if MlxCheckedTensorPlan::assess_program(program, requirements).is_ok() {
            return Ok(Self::Leaf);
        }
        if MlxCheckedTensorBinaryPlan::assess_program(program, requirements).is_ok() {
            return Ok(Self::Binary);
        }
        if MlxCheckedTensorIntegerPlan::assess_program(program, requirements).is_ok() {
            return Ok(Self::Integer);
        }
        if let Ok(plan) =
            MlxSelectedNumericalTensorPlan::assess_program(Arc::clone(program), requirements)
        {
            return Ok(Self::Numerical(plan));
        }
        MlxSelectedTensorGraphPlan::assess_program(Arc::clone(program), requirements)
            .map(Self::Graph)
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
            Self::Graph(plan) => offered(
                inventory,
                device,
                boundary,
                requirements,
                &MlxSelectedTensorGraphRequest { plan },
            ),
            Self::Numerical(plan) => offered(
                inventory,
                device,
                boundary,
                requirements,
                &MlxSelectedNumericalTensorRequest { plan },
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
            Self::Graph(plan) => {
                if plan.requirements() != requirements
                    || !Arc::ptr_eq(&plan.program_owner(), &program)
                {
                    return Err(PcuExecutionError::InvalidTensorSourcePlan);
                }
                let shape = Rc::from(plan.shape());
                session
                    .prepare_selected_tensor_graph(plan)
                    .map(|program| Program::Graph { program, shape })
            }
            Self::Numerical(plan) => {
                if plan.requirements() != requirements
                    || !Arc::ptr_eq(&plan.program_owner(), &program)
                {
                    return Err(PcuExecutionError::InvalidTensorSourcePlan);
                }
                let shape = Rc::from(plan.shape());
                session
                    .prepare_selected_numerical_tensor_program(plan)
                    .map(|program| Program::Numerical { program, shape })
            }
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

#[allow(clippy::large_enum_variant)]
// Retained cold cache entries are borrowed on warm calls. Boxing would add a
// separate allocation and pointer indirection to the numerical execution path.
pub(super) enum Program {
    Leaf(MlxPreparedCheckedProgram),
    Binary(MlxPreparedTensorBinaryProgram),
    Integer(MlxPreparedTensorIntegerProgram),
    Graph {
        program: MlxPreparedSelectedTensorGraph,
        shape: Rc<[usize]>,
    },
    Numerical {
        program: MlxPreparedSelectedNumericalTensorProgram,
        shape: Rc<[usize]>,
    },
}

impl Program {
    pub(super) fn implementation_id(&self) -> Option<PcuImplementationId> {
        match self {
            Self::Leaf(program) => program.implementation_id(),
            Self::Binary(program) => program.implementation_id(),
            Self::Integer(program) => program.implementation_id(),
            Self::Numerical { program, .. } => program.implementation_id(),
            Self::Graph { program, .. } => program.implementation_id(),
        }
    }

    pub(super) fn shape_owner(&self) -> Rc<[usize]> {
        match self {
            Self::Leaf(program) => program.plan().shape_owner(),
            Self::Binary(program) => program.plan().shape_owner(),
            Self::Integer(program) => program.plan().shape_owner(),
            Self::Numerical { shape, .. } | Self::Graph { shape, .. } => Rc::clone(shape),
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
            Self::Graph { program, .. } => {
                return program.execute_mixed(inputs).map_err(|error| {
                    match (error.effect, error.cause) {
                        (Some(effect), MlxError::Arithmetic(fault)) => {
                            let plan = program.plan();
                            super::super::fault::numerical(
                                plan.program(),
                                effect,
                                plan.requirements(),
                                fault,
                            )
                        }
                        (_, other) => super::map_error(other),
                    }
                });
            }
            Self::Numerical { program, .. } => {
                return program.execute_mixed(inputs).map_err(|error| match error {
                    MlxError::Arithmetic(fault) => {
                        let plan = program.plan();
                        super::super::fault::numerical(
                            plan.program(),
                            plan.effect(),
                            plan.requirements(),
                            fault,
                        )
                    }
                    other => super::map_error(other),
                });
            }
        }
        .map_err(super::map_error)
    }
}
