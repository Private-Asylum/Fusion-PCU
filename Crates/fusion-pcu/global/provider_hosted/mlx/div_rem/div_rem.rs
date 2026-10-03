//! Joint quotient/remainder publication for host and immutable resident borrows.
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxBinaryInput,
    MlxEncodedArray,
    MlxPreparedEncodedPrefix,
    MlxSession,
};
#[rustfmt::skip]
use crate::{
    global::arguments::{
        PcuCallArgument,
        PcuCallArgumentKind,
    },
    PcuBindingAccess,
    PcuBindingRef,
    PcuExecutionError,
    PcuHostDispatchError,
    PcuScalarType,
};
#[rustfmt::skip]
use super::{
    map_mlx_error,
    map_mlx_execution,
    validate_host,
    validate_output,
};

#[path = "kernel/kernel.rs"]
mod kernel;
use kernel::JointKernel;

type ArgumentPair<'a> = [PcuCallArgumentKind<'a>; 2];
type Inputs<'a> = [Option<PcuCallArgumentKind<'a>>; 2];

pub(super) fn call<K: JointKernel, const N: usize>(
    prepared: &mut K,
    declarations: &[PcuBindingRef],
    prefixes: &[Option<MlxPreparedEncodedPrefix>; 2],
    readback: &mut [Vec<u8>; 2],
    arguments: [PcuCallArgument<'_>; N],
) -> Result<(), PcuExecutionError> {
    let (inputs, outputs) = collect(
        prepared.input_bindings(),
        declarations,
        prepared.output_bindings(),
        prepared.scalar_type(),
        arguments,
    )?;
    let (inputs, mut outputs) = match (inputs, outputs) {
        (
            [
                Some(PcuCallArgumentKind::Host(left)),
                Some(PcuCallArgumentKind::Host(right)),
            ],
            [PcuCallArgumentKind::Host(q), PcuCallArgumentKind::Host(r)],
        ) => {
            return prepared
                .call(&mut [left, right, q, r])
                .map_err(map_mlx_error);
        }
        (
            [Some(PcuCallArgumentKind::Host(input)), None],
            [PcuCallArgumentKind::Host(q), PcuCallArgumentKind::Host(r)],
        ) => {
            return prepared.call(&mut [input, q, r]).map_err(map_mlx_error);
        }
        pair => pair,
    };
    let scalar = prepared.scalar_type();
    let bytes = prepared.output_byte_lengths();
    // Obtain both exclusive publication destinations before staging either input.
    let [q, r] = &mut outputs;
    let destinations = [
        Destination::prepare(
            q,
            scalar,
            bytes[0],
            prepared.output_bindings()[0],
            prefixes[0].as_ref(),
        )?,
        Destination::prepare(
            r,
            scalar,
            bytes[1],
            prepared.output_bindings()[1],
            prefixes[1].as_ref(),
        )?,
    ];
    let mut borrowed = [None; 2];
    for (slot, kind) in inputs
        .iter()
        .enumerate()
        .take(prepared.input_bindings().len())
    {
        borrowed[slot] = Some(input(
            kind.as_ref()
                .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?,
            scalar,
            prepared.input_byte_lengths()[slot],
            prepared.input_bindings()[slot],
        )?);
    }
    let first = borrowed[0].ok_or(PcuExecutionError::InvalidTensorSourcePlan)?;
    let [quotient, remainder] = if let Some(second) = borrowed[1] {
        prepared.execute_inputs(&[first, second])
    } else {
        prepared.execute_inputs(&[first])
    }
    .map_err(map_mlx_execution)?;
    let [q, r] = destinations;
    let [q_read, r_read] = readback;
    // Both terminal prefix merges/private host reads must succeed. Publication
    // objects have no fallible work left and do not inspect or replan the graph.
    let q = complete(q, quotient, scalar, bytes[0], prefixes[0].as_ref(), q_read)?;
    let r = complete(r, remainder, scalar, bytes[1], prefixes[1].as_ref(), r_read)?;
    let q = q.ready()?;
    let r = r.ready()?;
    q.publish();
    r.publish();
    Ok(())
}

fn input<'a>(
    kind: &'a PcuCallArgumentKind<'_>,
    scalar: PcuScalarType,
    bytes: usize,
    target: PcuBindingRef,
) -> Result<MlxBinaryInput<'a>, PcuExecutionError> {
    match kind {
        PcuCallArgumentKind::Host(argument) => {
            validate_host(argument, scalar, PcuBindingAccess::ReadOnly, bytes)?;
            Ok(MlxBinaryInput::HostBytes {
                target,
                scalar,
                bytes: argument.bytes(),
            })
        }
        PcuCallArgumentKind::MlxRead(argument) => Ok(MlxBinaryInput::Resident {
            target,
            array: argument.array,
        }),
        PcuCallArgumentKind::MlxWrite(_) => {
            Err(map_mlx_error(PcuHostDispatchError::AccessMismatch(target)))
        }
        #[cfg(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            all(feature = "cpu", feature = "tensor"),
            all(feature = "vulkan", feature = "tensor")
        ))]
        _ => Err(PcuExecutionError::Argument(
            crate::global::PcuArgumentError::SessionMismatch,
        )),
    }
}

enum Destination<'a> {
    Host(&'a mut [u8]),
    Owner {
        array: &'a mut MlxEncodedArray,
        session: &'a MlxSession,
    },
}
impl<'a> Destination<'a> {
    fn prepare(
        kind: &'a mut PcuCallArgumentKind<'_>,
        scalar: PcuScalarType,
        bytes: usize,
        binding: PcuBindingRef,
        prefix: Option<&MlxPreparedEncodedPrefix>,
    ) -> Result<Self, PcuExecutionError> {
        validate_output(kind, scalar, bytes, binding, prefix)?;
        match kind {
            PcuCallArgumentKind::Host(argument) => Ok(Self::Host(
                &mut argument
                    .bytes_mut()
                    .ok_or(PcuExecutionError::InvalidTensorSourcePlan)?[..bytes],
            )),
            PcuCallArgumentKind::MlxWrite(argument) => Ok(Self::Owner {
                array: argument.array,
                session: &argument.root.session,
            }),
            PcuCallArgumentKind::MlxRead(_) => Err(PcuExecutionError::InvalidTensorSourcePlan),
            #[cfg(all(feature = "cpu", feature = "tensor"))]
            PcuCallArgumentKind::CpuOwner(_) => Err(PcuExecutionError::InvalidTensorSourcePlan),
            #[cfg(all(feature = "vulkan", feature = "tensor"))]
            PcuCallArgumentKind::VulkanRead(_) | PcuCallArgumentKind::VulkanWrite(_) => {
                Err(PcuExecutionError::InvalidTensorSourcePlan)
            }
            #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
            PcuCallArgumentKind::ResidentRead(_) | PcuCallArgumentKind::ResidentWrite(_) => {
                Err(PcuExecutionError::InvalidTensorSourcePlan)
            }
        }
    }
}

enum Publication<'a> {
    Host {
        destination: &'a mut [u8],
        private: &'a [u8],
        completed: MlxEncodedArray,
    },
    Owner {
        destination: &'a mut MlxEncodedArray,
        completed: MlxEncodedArray,
    },
}
impl<'a> Publication<'a> {
    fn ready(self) -> Result<ReadyPublication<'a>, PcuExecutionError> {
        match self {
            Self::Host {
                destination,
                private,
                completed,
            } => {
                completed.release().map_err(map_mlx_execution)?;
                Ok(ReadyPublication::Host {
                    destination,
                    private,
                })
            }
            Self::Owner {
                destination,
                completed,
            } => Ok(ReadyPublication::Owner {
                destination,
                completed,
            }),
        }
    }
}

enum ReadyPublication<'a> {
    Host {
        destination: &'a mut [u8],
        private: &'a [u8],
    },
    Owner {
        destination: &'a mut MlxEncodedArray,
        completed: MlxEncodedArray,
    },
}
impl ReadyPublication<'_> {
    fn publish(self) {
        match self {
            Self::Host {
                destination,
                private,
            } => destination.copy_from_slice(private),
            Self::Owner {
                destination,
                completed,
            } => *destination = completed,
        }
    }
}

fn complete<'a>(
    destination: Destination<'a>,
    completed: MlxEncodedArray,
    scalar: PcuScalarType,
    bytes: usize,
    prefix: Option<&MlxPreparedEncodedPrefix>,
    readback: &'a mut [u8],
) -> Result<Publication<'a>, PcuExecutionError> {
    if completed.scalar_type() != scalar || completed.byte_len() != bytes {
        return Err(PcuExecutionError::InvalidTensorSourcePlan);
    }
    match destination {
        Destination::Host(destination) => {
            if readback.len() != bytes {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            }
            completed
                .read_bytes_into(readback)
                .map_err(map_mlx_execution)?;
            Ok(Publication::Host {
                destination,
                private: readback,
                completed,
            })
        }
        Destination::Owner { array, session } => {
            if !completed.same_session(session) {
                return Err(PcuExecutionError::Argument(
                    crate::global::PcuArgumentError::SessionMismatch,
                ));
            }
            let completed = if let Some(prefix) = prefix {
                let merged = prefix
                    .execute(&completed, array)
                    .map_err(map_mlx_execution)?;
                completed.release().map_err(map_mlx_execution)?;
                merged
            } else {
                completed
            };
            Ok(Publication::Owner {
                destination: array,
                completed,
            })
        }
    }
}

fn collect<'a, const N: usize>(
    inputs: &[PcuBindingRef],
    declarations: &[PcuBindingRef],
    outputs: [PcuBindingRef; 2],
    scalar: PcuScalarType,
    arguments: [PcuCallArgument<'a>; N],
) -> Result<(Inputs<'a>, ArgumentPair<'a>), PcuExecutionError> {
    if N != declarations.len() + 2 || N > 4 || !(1..=2).contains(&inputs.len()) {
        return Err(PcuExecutionError::InvalidTensorSourcePlan);
    }
    let mut reads = [None, None];
    let mut writes = [None, None];
    let mut seen = [outputs[0]; 4];
    for (position, argument) in arguments.into_iter().enumerate() {
        let kind = argument.into_parts().1;
        let target = match &kind {
            PcuCallArgumentKind::Host(argument) => argument.target(),
            PcuCallArgumentKind::MlxRead(argument) => argument.target,
            PcuCallArgumentKind::MlxWrite(argument) => argument.target,
            #[cfg(any(
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                all(feature = "cpu", feature = "tensor"),
                all(feature = "vulkan", feature = "tensor")
            ))]
            _ => {
                return Err(PcuExecutionError::Argument(
                    crate::global::PcuArgumentError::SessionMismatch,
                ));
            }
        };
        if seen[..position].contains(&target) {
            return Err(map_mlx_error(PcuHostDispatchError::Duplicate(target)));
        }
        seen[position] = target;
        if let Some(slot) = outputs.iter().position(|binding| *binding == target) {
            writes[slot] = Some(kind);
        } else if let Some(slot) = inputs.iter().position(|binding| *binding == target) {
            reads[slot] = Some(kind);
        } else if declarations.contains(&target) {
            // Validated source lowering supplies a typed empty declaration for an
            // unread parameter; no second borrow, upload or owner affinity exists.
            match &kind {
                PcuCallArgumentKind::Host(argument) => {
                    validate_host(argument, scalar, PcuBindingAccess::ReadOnly, 0)?;
                }
                PcuCallArgumentKind::MlxRead(argument)
                    if argument.array.scalar_type() == scalar => {}
                _ => return Err(map_mlx_error(PcuHostDispatchError::AccessMismatch(target))),
            }
        } else {
            return Err(map_mlx_error(PcuHostDispatchError::Unexpected(target)));
        }
    }
    for (slot, target) in inputs.iter().enumerate() {
        if reads[slot].is_none() {
            return Err(map_mlx_error(PcuHostDispatchError::Missing(*target)));
        }
    }
    let [q, r] = writes;
    let required = |kind: Option<PcuCallArgumentKind<'a>>, target| {
        kind.ok_or_else(|| map_mlx_error(PcuHostDispatchError::Missing(target)))
    };
    Ok((reads, [required(q, outputs[0])?, required(r, outputs[1])?]))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
