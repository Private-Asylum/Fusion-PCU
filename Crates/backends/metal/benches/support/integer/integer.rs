//! Matched graph/native staging and whole-output oracles, separated from Criterion scheduling.
#[rustfmt::skip]
use fusion_pcu_metal::{
    MetalBuffer,
    MetalError,
    MetalPreparedIntegerKernel,
    MetalPreparedIntegerMap,
    MetalSession,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuCheckedInteger,
    PcuScalar,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuHostArgument,
    PcuKernelId,
    PcuValueType,
    PcuValueTypeCaps,
};

pub fn graph<const N: usize, T: PcuScalar>(
    session: &MetalSession,
    op: PcuDispatchIntegerBinaryOp,
) -> MetalPreparedIntegerKernel {
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
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
            range_policy: pcu_facade::PcuRangePolicy::Reject,
            value_type: PcuValueType::Scalar(T::TYPE),
            op,
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
        .prepare_integer_kernel(&PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
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
impl DeviceMap for MetalPreparedIntegerKernel {
    fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        Self::execute(self, inputs)
    }
}
impl DeviceMap for MetalPreparedIntegerMap {
    fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        Self::execute(self, inputs[0], inputs[1])
    }
}
pub fn host_call<T: PcuScalar>(
    session: &MetalSession,
    map: &impl DeviceMap,
    left: &[T],
    right: &[T],
    output: &mut [T],
) -> Result<(), MetalError> {
    if left.len() != right.len() || output.len() < left.len() {
        return Err(MetalError::InvalidExtent);
    }
    let stage = |values| {
        let host = PcuHostArgument::read(PcuBindingRef::new(0, 0), values);
        session.upload_bytes(host.bytes())
    };
    let lhs = stage(left)?;
    let rhs = stage(right)?;
    let result = map.execute([&lhs, &rhs])?;
    let mut destination =
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output[..left.len()]);
    result.read_into_bytes(destination.bytes_mut().unwrap())
}
#[path = "../support.rs"]
mod shared;
pub use shared::guard;

/// Complete preflight oracles and transactional failures, outside all timed samples.
pub fn validate<const N: usize, T: PcuCheckedInteger + Eq + std::fmt::Debug>(
    session: &MetalSession,
    graph: &MetalPreparedIntegerKernel,
    native: &MetalPreparedIntegerMap,
    op: PcuDispatchIntegerBinaryOp,
    source: &mut impl FnMut(&[T], &[T], &mut [T]) -> Result<(), fusion_pcu_metal::MetalHostKernelError>,
    fixture: Fixture<T>,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut left = vec![fixture.phases[0]; N];
    let right = vec![fixture.right; N];
    let mut output = vec![fixture.sentinel; N + 1];
    for phase in fixture.phases {
        left.fill(phase);
        let expected = match op {
            PcuDispatchIntegerBinaryOp::Add => phase.pcu_checked_add(fixture.right),
            PcuDispatchIntegerBinaryOp::Sub => phase.pcu_checked_sub(fixture.right),
            PcuDispatchIntegerBinaryOp::Mul => phase.pcu_checked_mul(fixture.right),
        }
        .unwrap();
        source(&left, &right, &mut output).unwrap();
        assert!(output[..N].iter().all(|&value| value == expected));
        assert_eq!(output[N], fixture.sentinel);
        for result in [
            host_call(session, graph, &left, &right, &mut output),
            host_call(session, native, &left, &right, &mut output),
        ] {
            result.unwrap();
            assert!(output[..N].iter().all(|&value| value == expected));
            assert_eq!(output[N], fixture.sentinel);
        }
    }
    left[N / 2] = fixture.fault;
    left[N - 1] = fixture.fault;
    let before = output.clone();
    let Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(expected))) =
        source(&left, &right, &mut output)
    else {
        panic!("missing source range fault");
    };
    assert_eq!(expected.invocation_id, u64::try_from(N / 2).unwrap());
    assert!(!expected.recovered);
    assert_eq!(output, before);
    for result in [
        host_call(session, graph, &left, &right, &mut output),
        host_call(session, native, &left, &right, &mut output),
    ] {
        let Err(MetalError::Arithmetic(fault)) = result else {
            panic!("missing control range fault");
        };
        assert_eq!(fault, expected);
        assert_eq!(output, before);
    }
    left.fill(fixture.phases[1]);
    source(&left, &right, &mut output).unwrap();
    (left, right, output)
}

/// Type-specific borders for matching checked source/graph/native measurement preflight.
#[derive(Clone, Copy)]
pub struct Fixture<T> {
    pub phases: [T; 2],
    pub right: T,
    pub fault: T,
    pub sentinel: T,
}
