//! Bounded leaf/ReLU/binary Metal publication through the existing initialized device owner.
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalError,
    MetalTensorInput,
};
#[rustfmt::skip]
use crate::{
    PcuBindingRef,
    PcuDeviceBuffer,
    PcuDeviceTensor,
    PcuHostArgument,
    PcuScalar,
};
#[rustfmt::skip]
use crate::global::{
    PcuArgumentError,
    PcuBackendChoice,
    PcuExecutionError,
    PcuExecutionPolicy,
    PcuTensor,
    arguments::TensorBacking,
    resident::{
        DeviceTensor,
        Session,
    },
    tensor::{
        capture::PcuCapturedTensorProgram,
        PcuTensorInput,
        TensorInputKind,
    },
};
#[path = "program/program.rs"]
mod program;
#[path = "roots/roots.rs"]
mod roots;
use program::assess;

pub(super) struct Prepared {
    pub(super) root: Rc<Session>,
    program: program::Program,
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

pub(super) fn map_error(error: MetalError) -> PcuExecutionError {
    match error {
        MetalError::Arithmetic(fault) => PcuExecutionError::ArithmeticFault(fault),
        other => PcuExecutionError::BackendFailure(format!("Metal tensor execution: {other}")),
    }
}

impl Prepared {
    pub(super) fn prepare<T: PcuScalar, const N: usize>(
        built: &PcuCapturedTensorProgram,
        inputs: &[PcuTensorInput<'_, T>; N],
        options: PcuExecutionPolicy,
    ) -> Result<Self, PcuExecutionError> {
        // Assess the entire selected effect closure before discovery, uploads or native work.
        let plan = assess(built, options)?;
        if plan.scalar_type() != T::TYPE
            || !matches!(
                options.backend,
                PcuBackendChoice::Metal | PcuBackendChoice::Automatic
            )
        {
            return Err(PcuExecutionError::ResidentPolicyConflict);
        }
        if !(1..=2).contains(&built.input_indices.len()) {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let mut root: Option<Rc<Session>> = None;
        // Inspect every actually used resident owner before opening a host root.
        // Mixed host/resident and reordered source parameters retain native affinity.
        for &index in &built.input_indices {
            let input = inputs
                .get(index)
                .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
            if let TensorInputKind::Resident(owner) = input.kind {
                owner
                    .validate_initialized()
                    .map_err(PcuExecutionError::Argument)?;
                #[allow(irrefutable_let_patterns)]
                // Metal-only has exactly one backing; mixed builds reject opaque owners.
                let TensorBacking::Device { session, .. } = &owner.backing else {
                    return Err(PcuExecutionError::Argument(
                        PcuArgumentError::UnsupportedResidentBorrow,
                    ));
                };
                if session.metal_backend().is_none()
                    || options
                        .device
                        .is_some_and(|device| session.device_id() != device)
                {
                    return Err(PcuExecutionError::ResidentPolicyConflict);
                }
                if root
                    .as_ref()
                    .is_some_and(|previous| !previous.shares_metal_session(session))
                {
                    return Err(PcuExecutionError::Argument(
                        PcuArgumentError::SessionMismatch,
                    ));
                }
                root = Some(Rc::clone(session));
            }
        }
        let root = if let Some(session) = root {
            // Preserve physical affinity while newly escaped owners carry the current hint.
            if session.block_size() == options.block_size {
                session
            } else {
                Rc::new(Session::from_metal_backend(
                    session
                        .metal_backend()
                        .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?
                        .clone(),
                    options.block_size,
                ))
            }
        } else {
            roots::get(options.device.unwrap_or(0), options.block_size)?
        };
        let backend = root
            .metal_backend()
            .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
        let program = plan.prepare(backend.session())?;
        Ok(Self { root, program })
    }

    pub(super) fn execute<T: PcuScalar, const N: usize>(
        &self,
        inputs: &[PcuTensorInput<'_, T>; N],
        indices: &[usize],
    ) -> Result<PcuTensor<T>, PcuExecutionError> {
        if !(1..=2).contains(&indices.len()) || indices.iter().any(|&index| index >= N) {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        // Caller-owned stack argument spans live until terminal backend completion.
        // Repeated operands share one unique input and are projected by the frozen plan.
        let hosts: [Option<PcuHostArgument<'_>>; 2] = core::array::from_fn(|slot| {
            indices
                .get(slot)
                .and_then(|&index| match inputs[index].kind {
                    TensorInputKind::Host(values) => {
                        Some(PcuHostArgument::read(PcuBindingRef::new(0, 0), values))
                    }
                    TensorInputKind::Resident(_) => None,
                })
        });
        let first = self.bind(inputs[indices[0]], hosts[0].as_ref())?;
        let output = if let Some(&second) = indices.get(1) {
            let second = self.bind(inputs[second], hosts[1].as_ref())?;
            self.program.execute(&[first, second])?
        } else {
            self.program.execute(&[first])?
        };
        let shape = crate::PcuOwnedShape::from_slice(output.shape());
        let count = output.element_count();
        // This is the backend's actual terminal initialized resource, never a pointer import.
        let (_, resource) = output.into_resource();
        let tensor = PcuDeviceTensor::new(shape, PcuDeviceBuffer::<T, _>::new(resource, count))
            .map_err(PcuExecutionError::TensorStorage)?;
        Ok(PcuTensor::from_successful_output(
            DeviceTensor::Metal(tensor),
            Rc::clone(&self.root),
        ))
    }

    fn bind<'a, T: PcuScalar>(
        &self,
        input: PcuTensorInput<'a, T>,
        host: Option<&'a PcuHostArgument<'a>>,
    ) -> Result<MetalTensorInput<'a>, PcuExecutionError> {
        match input.kind {
            TensorInputKind::Host(values) => Ok(MetalTensorInput::HostBytes {
                scalar: T::TYPE,
                elements: values.len(),
                bytes: host
                    .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?
                    .bytes(),
            }),
            TensorInputKind::Resident(owner) => {
                #[allow(irrefutable_let_patterns)]
                // Metal-only has one backing; mixed builds must reject foreign providers.
                let TensorBacking::Device {
                    tensor: DeviceTensor::Metal(tensor),
                    session,
                    ..
                } = &owner.backing
                else {
                    return Err(PcuExecutionError::Argument(
                        PcuArgumentError::UnsupportedResidentBorrow,
                    ));
                };
                if !self.root.shares_metal_session(session) {
                    return Err(PcuExecutionError::Argument(
                        PcuArgumentError::SessionMismatch,
                    ));
                }
                Ok(MetalTensorInput::Resident {
                    scalar: T::TYPE,
                    elements: tensor.buffer().len(),
                    resource: tensor.buffer().resource(),
                })
            }
        }
    }
}
