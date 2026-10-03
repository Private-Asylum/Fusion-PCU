//! Matched copied-host inputs, fresh device output/status, terminal check and readback.
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuCheckedFloat,
    PcuScalar,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuKernelId,
    PcuRangePolicy,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalBuffer,
    MetalError,
    MetalPreparedFloatBinary,
    MetalPreparedFloatBinaryKernel,
    MetalSession,
};
pub fn graph<const N: usize, T: PcuCheckedFloat>(
    session: &MetalSession,
    op: PcuDispatchFloatBinaryOp,
) -> MetalPreparedFloatBinaryKernel {
    graph_with_reproducibility::<N, T>(session, op, pcu_facade::PcuReproducibility::Unspecified)
}
pub fn graph_with_reproducibility<const N: usize, T: PcuCheckedFloat>(
    session: &MetalSession,
    op: PcuDispatchFloatBinaryOp,
    reproducibility: pcu_facade::PcuReproducibility,
) -> MetalPreparedFloatBinaryKernel {
    let bindings = [
        PcuBinding::scalar::<T>(
            Some("left"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("right"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<T>(
            Some("output"),
            0,
            2,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
    ];
    let index = PcuDispatchIndex::InvocationId;
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: bindings[0].reference(),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(2),
            binding: bindings[1].reference(),
            index,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            value_type: PcuValueType::Scalar(T::TYPE),
            op,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: bindings[2].reference(),
            index,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    session
        .prepare_float_binary_kernel(&PcuDispatchKernelIr {
            numerical_requirements: pcu_facade::PcuImplementationRequirements {
                // The genuine source Sub workload explicitly chooses Strict; one checked
                // primitive retains identical arithmetic, with the same frozen admission tuple.
                numerical_mode: if matches!(op, PcuDispatchFloatBinaryOp::Sub) {
                    pcu_facade::PcuNumericalMode::Strict
                } else {
                    pcu_facade::PcuNumericalMode::Boundary
                },
                numerical_options: pcu_facade::PcuNumericalOptions {
                    reproducibility,
                    ..pcu_facade::PcuNumericalOptions::default()
                },
                ..PcuDispatchKernelIr::DEFAULT_REQUIREMENTS
            },
            id: PcuKernelId(71),
            entry: PcuDispatchEntryPoint {
                name: "signed_map",
                logical_shape: [u32::try_from(N).unwrap(), 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::for_value_type(PcuValueType::Scalar(T::TYPE)),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        })
        .unwrap()
}

pub trait DeviceMap {
    fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError>;
}
impl DeviceMap for MetalPreparedFloatBinaryKernel {
    fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        Self::execute(self, inputs)
    }
}
impl DeviceMap for MetalPreparedFloatBinary {
    fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        Self::execute(self, inputs[0], inputs[1])
    }
}
pub fn host_call<T: PcuScalar>(
    session: &MetalSession,
    executable: &impl DeviceMap,
    left: &[T],
    right: &[T],
    output: &mut [T],
) -> Result<(), MetalError> {
    if left.len() != right.len() || output.len() < left.len() {
        return Err(MetalError::InvalidExtent);
    }
    let left_argument = PcuHostArgument::read(PcuBindingRef::new(0, 0), left);
    let right_argument = PcuHostArgument::read(PcuBindingRef::new(0, 1), right);
    let a = session.upload_bytes(left_argument.bytes())?;
    let b = session.upload_bytes(right_argument.bytes())?;
    let result = executable.execute([&a, &b])?;
    let mut destination =
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output[..left.len()]);
    result.read_into_bytes(destination.bytes_mut().expect("mutable host destination"))
}
pub fn expected<T: PcuCheckedFloat>(
    op: PcuDispatchFloatBinaryOp,
    left: &[T],
    right: &[T],
    sentinel: T,
) -> Vec<u8> {
    let expected: Vec<T> = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| {
            match op {
                PcuDispatchFloatBinaryOp::Add => a.pcu_checked_add(b),
                PcuDispatchFloatBinaryOp::Sub => a.pcu_checked_sub(b),
                PcuDispatchFloatBinaryOp::Mul => a.pcu_checked_mul(b),
                PcuDispatchFloatBinaryOp::Div => a.pcu_checked_div(b),
            }
            .unwrap()
        })
        .chain(core::iter::once(sentinel))
        .collect();
    PcuHostArgument::read(PcuBindingRef::new(0, 0), &expected)
        .bytes()
        .to_vec()
}
