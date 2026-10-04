//! Static opaque ownership and frozen ordinary source preparation.

#[rustfmt::skip]
use super::{
    PcuExecutionError,
    PcuHostCallSite,
    PcuScalar,
    PcuSourceShape,
    PcuTensor,
    PcuTensorGraphCapture,
    PcuTensorGraphValue,
    PcuTensorInput,
    PcuTensorShapeWitness,
    TensorInputKind,
};
#[rustfmt::skip]
use crate::global::{
    PcuArgumentError,
    PcuBackendChoice,
    arguments::TensorBacking,
    policy,
};
#[rustfmt::skip]
use std::{
    any::TypeId,
    cell::RefCell,
};
#[cfg(feature = "cpu")]
#[path = "cpu/cpu.rs"]
mod cpu;
#[cfg(feature = "metal")]
#[path = "metal/metal.rs"]
mod metal;
#[cfg(feature = "mlx")]
#[path = "mlx/mlx.rs"]
mod mlx;
#[cfg(feature = "mlx")]
#[path = "mlx/encoded/encoded.rs"]
mod mlx_encoded;
#[cfg(feature = "mlx")]
#[path = "mlx/roots/roots.rs"]
mod mlx_roots;
#[cfg(feature = "vulkan")]
#[path = "vulkan/vulkan.rs"]
mod vulkan;

#[cfg(any(feature = "metal", feature = "mlx"))]
#[path = "fault/fault.rs"]
mod fault;

#[cfg(any(feature = "metal", feature = "mlx"))]
#[path = "numerical/numerical.rs"]
mod numerical;

#[derive(PartialEq, Eq)]
enum ShapeKey {
    Static(PcuSourceShape),
    Dynamic(Vec<usize>),
}
impl ShapeKey {
    fn new(shape: PcuTensorShapeWitness<'_>) -> Self {
        match shape {
            PcuTensorShapeWitness::Static(shape) => Self::Static(shape),
            PcuTensorShapeWitness::Dynamic(shape) => Self::Dynamic(shape.to_vec()),
        }
    }
    fn matches(&self, shape: PcuTensorShapeWitness<'_>) -> bool {
        match (self, shape) {
            (Self::Static(left), PcuTensorShapeWitness::Static(right)) => *left == right,
            (Self::Dynamic(left), PcuTensorShapeWitness::Dynamic(right)) => left == right,
            _ => false,
        }
    }
}

#[allow(clippy::large_enum_variant)]
// Concrete prepared providers are retained once in the cold cache and borrowed
// for replay. Boxing the encoded numerical entry adds an allocation and another
// pointer indirection merely to shrink other compiled-provider feature subsets.
enum Prepared {
    #[cfg(feature = "metal")]
    Metal(metal::Prepared),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::Prepared),
    #[cfg(feature = "mlx")]
    Mlx(mlx::Prepared),
    #[cfg(feature = "mlx")]
    MlxEncoded(mlx_encoded::Prepared),
    #[cfg(feature = "cpu")]
    Cpu {
        program: cpu::CpuPrepared,
        shape: std::rc::Rc<[usize]>,
    },
    #[cfg(feature = "cpu")]
    CpuOutputs {
        program: cpu::CpuPrepared,
        shapes: Vec<std::rc::Rc<[usize]>>,
    },
}
struct Entry {
    site: usize,
    specialization: TypeId,
    factory: TypeId,
    scalar: crate::PcuScalarType,
    shapes: Vec<ShapeKey>,
    indices: Vec<usize>,
    #[cfg(feature = "mlx")]
    resident_roles: Vec<bool>,
    #[cfg(feature = "mlx")]
    ids: Vec<crate::dialect::tensor::ValueId>,
    prepared: Prepared,
}
#[derive(Default)]
struct State {
    generation: u64,
    entries: Vec<Entry>,
}
std::thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

pub(super) fn selected<T: PcuScalar, const N: usize>(inputs: &[PcuTensorInput<'_, T>; N]) -> bool {
    selected_for_backend(policy::route().backend, inputs)
}

fn selected_for_backend<T: PcuScalar, const N: usize>(
    backend: PcuBackendChoice,
    inputs: &[PcuTensorInput<'_, T>; N],
) -> bool {
    // Explicit device routes inspect only captured semantic inputs later. Unused opaque
    // owners remain shape/type witnesses; used foreign owners still fail device preflight.
    #[cfg(feature = "cuda")]
    if backend == PcuBackendChoice::Cuda {
        return false;
    }
    #[cfg(feature = "rocm")]
    if backend == PcuBackendChoice::Rocm {
        return false;
    }
    #[cfg(feature = "metal")]
    if backend == PcuBackendChoice::Metal {
        return true;
    }
    #[cfg(feature = "vulkan")]
    if backend == PcuBackendChoice::Vulkan {
        return true;
    }
    #[cfg(feature = "cpu")]
    if backend == PcuBackendChoice::Cpu {
        return true;
    }
    #[cfg(feature = "mlx")]
    if backend == PcuBackendChoice::Mlx {
        return true;
    }
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    let mut device_affinity = false;
    for input in inputs {
        if let TensorInputKind::Resident(owner) = input.kind {
            match &owner.backing {
                #[cfg(feature = "vulkan")]
                TensorBacking::Vulkan { .. } => return true,
                #[cfg(feature = "mlx")]
                TensorBacking::Mlx { .. } | TensorBacking::MlxEncoded { .. } => return true,
                #[cfg(feature = "cpu")]
                TensorBacking::Cpu { .. } => return true,
                #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
                TensorBacking::Device { session, .. } => {
                    #[cfg(not(feature = "metal"))]
                    let _ = session;
                    #[cfg(feature = "metal")]
                    if session.metal_backend().is_some() {
                        return true;
                    }
                    device_affinity = true;
                }
            }
        }
    }
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    if device_affinity {
        return false;
    }
    (automatic_vulkan_only(backend)
        || cfg!(all(
            feature = "metal",
            not(any(
                feature = "mlx",
                feature = "rocm",
                feature = "cuda",
                feature = "vulkan",
                feature = "cpu"
            ))
        ))
        || cfg!(all(
            feature = "mlx",
            not(any(feature = "rocm", feature = "cuda", feature = "vulkan"))
        )))
        && backend == PcuBackendChoice::Automatic
}

// Sole-provider eligibility is a compile-time fact, not a runtime fallback. Preserve the
// existing default-device opening and retained cold root; mixed-provider ranking is separate.
const fn automatic_vulkan_only(backend: PcuBackendChoice) -> bool {
    cfg!(all(
        feature = "vulkan",
        not(any(
            feature = "cpu",
            feature = "metal",
            feature = "mlx",
            feature = "rocm",
            feature = "cuda"
        ))
    )) && matches!(backend, PcuBackendChoice::Automatic)
}

fn roles_match<T: PcuScalar, const N: usize>(
    entry: &Entry,
    inputs: &[PcuTensorInput<'_, T>; N],
) -> bool {
    #[cfg(feature = "mlx")]
    {
        entry
            .resident_roles
            .iter()
            .zip(inputs)
            .all(|(&resident, input)| {
                resident == matches!(input.kind, TensorInputKind::Resident(_))
            })
    }
    #[cfg(not(feature = "mlx"))]
    {
        let _ = (entry, inputs);
        true
    }
}

fn affinity_matches<T: PcuScalar, const N: usize>(
    entry: &Entry,
    inputs: &[PcuTensorInput<'_, T>; N],
) -> bool {
    entry.indices.iter().all(|&index| match inputs[index].kind {
        TensorInputKind::Host(_) => true,
        TensorInputKind::Resident(owner) => match (&entry.prepared, &owner.backing) {
            #[cfg(feature = "metal")]
            (Prepared::Metal(prepared), TensorBacking::Device { session, .. }) => {
                prepared.root.shares_metal_session(session)
            }
            #[cfg(feature = "vulkan")]
            (Prepared::Vulkan(prepared), TensorBacking::Vulkan { buffer, .. }) => {
                prepared.root.owns_buffer(buffer)
            }
            #[cfg(feature = "mlx")]
            (Prepared::Mlx(prepared), TensorBacking::Mlx { array, .. }) => {
                prepared.root.session.same_session(array.session())
            }
            #[cfg(feature = "mlx")]
            (Prepared::Mlx(prepared), TensorBacking::MlxEncoded { array, .. }) => {
                array.same_session(&prepared.root.session)
            }
            #[cfg(feature = "mlx")]
            (Prepared::MlxEncoded(prepared), TensorBacking::MlxEncoded { array, .. }) => {
                array.same_session(&prepared.root.session)
            }
            #[cfg(feature = "mlx")]
            (Prepared::MlxEncoded(prepared), TensorBacking::Mlx { array, .. }) => {
                prepared.root.session.same_session(array.session())
            }
            #[cfg(feature = "cpu")]
            (Prepared::Cpu { .. } | Prepared::CpuOutputs { .. }, TensorBacking::Cpu { .. }) => true,
            #[cfg(any(
                feature = "rocm",
                feature = "cuda",
                all(
                    feature = "metal",
                    any(feature = "mlx", feature = "cpu", feature = "vulkan")
                )
            ))]
            (_, TensorBacking::Device { .. }) => false,
            #[cfg(any(
                all(feature = "mlx", any(feature = "cpu", feature = "vulkan",)),
                all(feature = "vulkan", feature = "cpu"),
                all(
                    feature = "metal",
                    any(feature = "mlx", feature = "cpu", feature = "vulkan")
                )
            ))]
            _ => false,
        },
    })
}

fn prepare<T: PcuScalar, const N: usize>(
    built: &super::capture::PcuCapturedTensorProgram,
    inputs: &[PcuTensorInput<'_, T>; N],
    options: crate::global::PcuExecutionPolicy,
) -> Result<Prepared, PcuExecutionError> {
    if options.range_policy != crate::PcuRangePolicy::Reject {
        return Err(PcuExecutionError::UnsupportedRangePolicy);
    }
    #[cfg(feature = "metal")]
    {
        let resident_metal = built.input_indices.iter().any(|&index| matches!(inputs[index].kind, TensorInputKind::Resident(owner) if matches!(&owner.backing, TensorBacking::Device { session, .. } if session.metal_backend().is_some())));
        if options.backend == PcuBackendChoice::Metal
            || resident_metal
            || cfg!(not(any(
                feature = "mlx",
                feature = "rocm",
                feature = "cuda",
                feature = "vulkan",
                feature = "cpu"
            ))) && options.backend == PcuBackendChoice::Automatic
        {
            return metal::Prepared::prepare(built, inputs, options).map(Prepared::Metal);
        }
    }
    #[cfg(feature = "vulkan")]
    {
        let resident_vulkan = built.input_indices.iter().any(|&index| matches!(inputs[index].kind, TensorInputKind::Resident(owner) if matches!(owner.backing, TensorBacking::Vulkan { .. })));
        if options.backend == PcuBackendChoice::Vulkan
            || resident_vulkan
            || automatic_vulkan_only(options.backend)
        {
            return vulkan::Prepared::prepare(built, inputs, options).map(Prepared::Vulkan);
        }
    }
    #[cfg(feature = "cpu")]
    {
        let resident_cpu = built.input_indices.iter().any(|&index| matches!(inputs[index].kind, TensorInputKind::Resident(owner) if matches!(owner.backing, TensorBacking::Cpu { .. })));
        if options.backend == PcuBackendChoice::Cpu || resident_cpu {
            if resident_cpu
                && !matches!(
                    options.backend,
                    PcuBackendChoice::Cpu | PcuBackendChoice::Automatic
                )
            {
                return Err(PcuExecutionError::ResidentPolicyConflict);
            }
            if options.device.is_some_and(|ordinal| ordinal != 0) {
                return Err(PcuExecutionError::ResidentPolicyConflict);
            }
            let program = cpu::CpuPrepared::prepare(&built.program, T::TYPE)?;
            let shape = std::rc::Rc::from(program.output_shape());
            return Ok(Prepared::Cpu { program, shape });
        }
    }
    #[cfg(feature = "mlx")]
    {
        if matches!(
            T::TYPE,
            crate::PcuScalarType::F16
                | crate::PcuScalarType::BF16
                | crate::PcuScalarType::F8E4M3FN
                | crate::PcuScalarType::F8E5M2
        ) || mlx_encoded::assess(built, options).is_ok()
        {
            mlx_encoded::Prepared::prepare(built, inputs, options).map(Prepared::MlxEncoded)
        } else {
            mlx::Prepared::prepare(built, inputs, options).map(Prepared::Mlx)
        }
    }
    #[cfg(not(feature = "mlx"))]
    Err(PcuExecutionError::TensorExecutionUnavailable)
}

fn execute<T: PcuScalar, const N: usize>(
    entry: &mut Entry,
    inputs: &[PcuTensorInput<'_, T>; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    if !affinity_matches(entry, inputs) {
        return Err(PcuExecutionError::Argument(
            PcuArgumentError::SessionMismatch,
        ));
    }
    for &index in &entry.indices {
        if let TensorInputKind::Resident(owner) = inputs[index].kind {
            match inputs[index].shape {
                PcuTensorShapeWitness::Static(shape) => owner.validate_read(shape),
                PcuTensorShapeWitness::Dynamic(_) => owner.validate_initialized(),
            }
            .map_err(PcuExecutionError::Argument)?;
        }
    }
    match &mut entry.prepared {
        #[cfg(feature = "metal")]
        Prepared::Metal(prepared) => prepared.execute(inputs, &entry.indices),
        #[cfg(feature = "vulkan")]
        Prepared::Vulkan(prepared) => prepared.execute(inputs, &entry.indices),
        #[cfg(feature = "mlx")]
        Prepared::Mlx(prepared) => {
            let [left, right] = entry.indices.as_slice() else {
                return Err(PcuExecutionError::InvalidTensorSourcePlan);
            };
            let left_input = mlx::bind_input(&inputs[*left])?;
            let right_input = mlx::bind_input(&inputs[*right])?;
            let array = prepared.execute(&[
                (entry.ids[0], left_input.as_input()),
                (entry.ids[1], right_input.as_input()),
            ])?;
            let shape = array.shape();
            Ok(PcuTensor {
                backing: TensorBacking::Mlx {
                    array,
                    shape,
                    root: std::rc::Rc::clone(&prepared.root),
                    marker: core::marker::PhantomData,
                },
            })
        }
        #[cfg(feature = "mlx")]
        Prepared::MlxEncoded(prepared) => {
            let array = prepared.execute(inputs, &entry.indices, &entry.ids)?;
            Ok(PcuTensor {
                backing: TensorBacking::MlxEncoded {
                    array,
                    shape: prepared.shape(),
                    root: std::rc::Rc::clone(&prepared.root),
                    marker: core::marker::PhantomData,
                },
            })
        }
        #[cfg(feature = "cpu")]
        Prepared::CpuOutputs { .. } => Err(PcuExecutionError::InvalidTensorSourcePlan),
        #[cfg(feature = "cpu")]
        Prepared::Cpu { program, shape } => {
            let mut values = [&[][..]; N];
            for (binding, &index) in entry.indices.iter().enumerate() {
                values[binding] = match inputs[index].kind {
                    TensorInputKind::Host(values) => values,
                    TensorInputKind::Resident(owner) => match &owner.backing {
                        TensorBacking::Cpu { values, .. } => values,
                        #[cfg(any(
                            feature = "mlx",
                            feature = "vulkan",
                            feature = "rocm",
                            feature = "cuda",
                            feature = "metal"
                        ))]
                        _ => {
                            return Err(PcuExecutionError::Argument(
                                PcuArgumentError::UnsupportedResidentBorrow,
                            ));
                        }
                    },
                };
            }
            let values = program.execute(&values[..entry.indices.len()])?;
            Ok(PcuTensor {
                backing: TensorBacking::Cpu {
                    values,
                    shape: std::rc::Rc::clone(shape),
                },
            })
        }
    }
}

pub(super) fn call<T: PcuScalar, const N: usize, F>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    inputs: &[PcuTensorInput<'_, T>; N],
    capture_function: F,
) -> Result<PcuTensor<T>, PcuExecutionError>
where
    F: FnOnce(
            &mut PcuTensorGraphCapture,
            [PcuTensorGraphValue<T>; N],
        ) -> Result<PcuTensorGraphValue<T>, PcuExecutionError>
        + 'static,
{
    let generation = policy::route().generation;
    STATE
        .try_with(|state| {
            let mut state = state
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            if state.generation != generation {
                state.entries.clear();
                state.generation = generation;
            }
            let site = core::ptr::from_ref(site).addr();
            if let Some(entry) = state.entries.iter_mut().find(|entry| {
                entry.site == site
                    && entry.specialization == specialization
                    && entry.factory == TypeId::of::<F>()
                    && entry.scalar == T::TYPE
                    && entry.shapes.len() == N
                    && entry
                        .shapes
                        .iter()
                        .zip(inputs)
                        .all(|(shape, input)| shape.matches(input.shape))
                    && roles_match(entry, inputs)
                    && affinity_matches(entry, inputs)
            }) {
                return execute(entry, inputs);
            }
            let snapshot = policy::snapshot()?;
            let shapes = inputs.map(|input| input.shape);
            let built = super::capture::build(
                shapes,
                snapshot.policy.float_underflow,
                snapshot.policy.numerical_mode,
                snapshot.policy.numerical_options,
                capture_function,
            )?;
            let prepared = prepare(&built, inputs, snapshot.policy)?;
            let mut entry = Entry {
                site,
                specialization,
                factory: TypeId::of::<F>(),
                scalar: T::TYPE,
                shapes: shapes.into_iter().map(ShapeKey::new).collect(),
                #[cfg(feature = "mlx")]
                resident_roles: inputs
                    .iter()
                    .map(|input| matches!(input.kind, TensorInputKind::Resident(_)))
                    .collect(),
                indices: built.input_indices,
                #[cfg(feature = "mlx")]
                ids: built.input_ids,
                prepared,
            };
            let result = execute(&mut entry, inputs)?;
            if snapshot.policy.cache_capacity != 0 {
                if state.entries.len() >= snapshot.policy.cache_capacity {
                    state.entries.remove(0);
                }
                state.entries.push(entry);
            }
            Ok(result)
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

pub(super) fn call_outputs<T: PcuScalar, const N: usize, const M: usize, F>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    inputs: &[PcuTensorInput<'_, T>; N],
    capture_function: F,
) -> Result<[PcuTensor<T>; M], PcuExecutionError>
where
    F: FnOnce(
            &mut PcuTensorGraphCapture,
            [PcuTensorGraphValue<T>; N],
        ) -> Result<[PcuTensorGraphValue<T>; M], PcuExecutionError>
        + 'static,
{
    let generation = policy::route().generation;
    STATE
        .try_with(|state| {
            let mut state = state
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?;
            if state.generation != generation {
                state.entries.clear();
                state.generation = generation;
            }
            let site = core::ptr::from_ref(site).addr();
            if let Some(entry) = state.entries.iter_mut().find(|entry| {
                entry.site == site
                    && entry.specialization == specialization
                    && entry.factory == TypeId::of::<F>()
                    && entry.scalar == T::TYPE
                    && entry.shapes.len() == N
                    && entry
                        .shapes
                        .iter()
                        .zip(inputs)
                        .all(|(shape, input)| shape.matches(input.shape))
                    && roles_match(entry, inputs)
                    && affinity_matches(entry, inputs)
            }) {
                return execute_outputs(entry, inputs);
            }
            let snapshot = policy::snapshot()?;
            let shapes = inputs.map(|input| input.shape);
            let built = super::capture::build_outputs(
                shapes,
                snapshot.policy.float_underflow,
                snapshot.policy.numerical_mode,
                snapshot.policy.numerical_options,
                capture_function,
            )?;
            let prepared = prepare_outputs(&built, inputs, snapshot.policy)?;
            let mut entry = Entry {
                site,
                specialization,
                factory: TypeId::of::<F>(),
                scalar: T::TYPE,
                shapes: shapes.into_iter().map(ShapeKey::new).collect(),
                #[cfg(feature = "mlx")]
                resident_roles: inputs
                    .iter()
                    .map(|input| matches!(input.kind, TensorInputKind::Resident(_)))
                    .collect(),
                indices: built.input_indices,
                #[cfg(feature = "mlx")]
                ids: built.input_ids,
                prepared,
            };
            let result = execute_outputs(&mut entry, inputs)?;
            if snapshot.policy.cache_capacity != 0 {
                if state.entries.len() >= snapshot.policy.cache_capacity {
                    state.entries.remove(0);
                }
                state.entries.push(entry);
            }
            Ok(result)
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

pub(super) fn clear() -> Result<(), PcuExecutionError> {
    STATE
        .try_with(|state| {
            state
                .try_borrow_mut()
                .map_err(|_| PcuExecutionError::ReentrantCall)?
                .entries
                .clear();
            Ok(())
        })
        .map_err(|_| PcuExecutionError::ThreadUnavailable)?
}

#[cfg(all(test, feature = "cpu", any(feature = "cuda", feature = "rocm")))]
#[path = "routing/tests/tests.rs"]
mod routing_tests;

#[cfg(all(
    test,
    feature = "vulkan",
    not(any(
        feature = "cpu",
        feature = "metal",
        feature = "mlx",
        feature = "rocm",
        feature = "cuda"
    ))
))]
#[path = "routing/vulkan_only/vulkan_only.rs"]
mod vulkan_only_routing_tests;

fn prepare_outputs<T: PcuScalar, const N: usize>(
    built: &super::capture::PcuCapturedTensorProgram,
    inputs: &[PcuTensorInput<'_, T>; N],
    options: crate::global::PcuExecutionPolicy,
) -> Result<Prepared, PcuExecutionError> {
    if options.range_policy != crate::PcuRangePolicy::Reject {
        return Err(PcuExecutionError::UnsupportedRangePolicy);
    }
    #[cfg(feature = "cpu")]
    {
        let resident_cpu = built.input_indices.iter().any(|&index| matches!(inputs[index].kind, TensorInputKind::Resident(owner) if matches!(owner.backing, TensorBacking::Cpu { .. })));
        if options.backend == PcuBackendChoice::Cpu || resident_cpu {
            if resident_cpu
                && !matches!(
                    options.backend,
                    PcuBackendChoice::Cpu | PcuBackendChoice::Automatic
                )
                || options.device.is_some_and(|ordinal| ordinal != 0)
            {
                return Err(PcuExecutionError::ResidentPolicyConflict);
            }
            let program = cpu::CpuPrepared::prepare_outputs(&built.program, T::TYPE)?;
            let shapes = program.output_shapes();
            return Ok(Prepared::CpuOutputs { program, shapes });
        }
    }
    let _ = (built, inputs);
    Err(PcuExecutionError::TensorExecutionUnavailable)
}

fn execute_outputs<T: PcuScalar, const N: usize, const M: usize>(
    entry: &mut Entry,
    inputs: &[PcuTensorInput<'_, T>; N],
) -> Result<[PcuTensor<T>; M], PcuExecutionError> {
    if !affinity_matches(entry, inputs) {
        return Err(PcuExecutionError::Argument(
            PcuArgumentError::SessionMismatch,
        ));
    }
    for &index in &entry.indices {
        if let TensorInputKind::Resident(owner) = inputs[index].kind {
            match inputs[index].shape {
                PcuTensorShapeWitness::Static(shape) => owner.validate_read(shape),
                PcuTensorShapeWitness::Dynamic(_) => owner.validate_initialized(),
            }
            .map_err(PcuExecutionError::Argument)?;
        }
    }
    #[cfg(feature = "cpu")]
    if let Prepared::CpuOutputs { program, shapes } = &mut entry.prepared {
        if shapes.len() != M {
            return Err(PcuExecutionError::InvalidTensorSourcePlan);
        }
        let mut values = [&[][..]; N];
        for (binding, &index) in entry.indices.iter().enumerate() {
            values[binding] = match inputs[index].kind {
                TensorInputKind::Host(values) => values,
                TensorInputKind::Resident(owner) => match &owner.backing {
                    TensorBacking::Cpu { values, .. } => values,
                    #[cfg(any(
                        feature = "mlx",
                        feature = "vulkan",
                        feature = "rocm",
                        feature = "cuda",
                        feature = "metal"
                    ))]
                    _ => {
                        return Err(PcuExecutionError::Argument(
                            PcuArgumentError::UnsupportedResidentBorrow,
                        ));
                    }
                },
            };
        }
        let outputs = program.execute_outputs::<T, M>(&values[..entry.indices.len()])?;
        let mut index = 0;
        return Ok(outputs.map(|values| {
            let shape = std::rc::Rc::clone(&shapes[index]);
            index += 1;
            PcuTensor {
                backing: TensorBacking::Cpu { values, shape },
            }
        }));
    }
    Err(PcuExecutionError::TensorExecutionUnavailable)
}
