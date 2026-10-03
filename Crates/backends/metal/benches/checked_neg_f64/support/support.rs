//! Matching F64 source/graph/native ownership and publication boundary.
#[rustfmt::skip]
use pcu_facade::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatUnaryOp,
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
    MetalPreparedFloatKernel,
    MetalPreparedFloatUnary,
    MetalSession,
};

pub trait DeviceMap {
    fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError>;
}
impl DeviceMap for MetalPreparedFloatKernel {
    fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        Self::execute(self, input)
    }
}
impl DeviceMap for MetalPreparedFloatUnary {
    fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        Self::execute(self, input)
    }
}
pub fn host_call(
    session: &MetalSession,
    executable: &impl DeviceMap,
    input: &[f64],
    output: &mut [f64],
) -> Result<(), MetalError> {
    if output.len() < input.len() {
        return Err(MetalError::InvalidExtent);
    }
    let length = input.len();
    let input = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
    let input = session.upload_bytes(input.bytes())?;
    let result = executable.execute(&input)?;
    let mut destination =
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output[..length]);
    result.read_into_bytes(destination.bytes_mut().expect("mutable host destination"))
}
pub fn graph<const N: usize>(session: &MetalSession) -> MetalPreparedFloatKernel {
    let bindings = [
        PcuBinding::scalar::<f64>(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
        ),
        PcuBinding::scalar::<f64>(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
        ),
    ];
    let ops = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: PcuValueType::f64(),
            op: PcuDispatchFloatUnaryOp::Neg,
            underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(2),
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(2),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    session
        .prepare_float_unary_kernel(&PcuDispatchKernelIr {
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            id: PcuKernelId(1),
            entry: PcuDispatchEntryPoint {
                name: "checked_neg",
                logical_shape: [u32::try_from(N).unwrap(), 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::FLOAT64,
            feature_caps: PcuDispatchFeatureCaps::empty(),
        })
        .unwrap()
}

pub fn validate<const N: usize>(
    session: &MetalSession,
    graph: &MetalPreparedFloatKernel,
    native: &MetalPreparedFloatUnary,
    source: &mut impl FnMut(&[f64], &mut [f64]) -> Result<(), fusion_pcu_metal::MetalHostKernelError>,
) {
    let mut input = vec![1.0_f64; N];
    let special = [
        0,
        0x8000_0000_0000_0000,
        1,
        0x8000_0000_0000_0001,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
    ];
    for (value, bits) in input.iter_mut().zip(special) {
        *value = f64::from_bits(bits);
    }
    let expected: Vec<_> = input
        .iter()
        .map(|value| value.to_bits() ^ 0x8000_0000_0000_0000)
        .chain([91.0_f64.to_bits()])
        .collect();
    let mut output = vec![91.0_f64; N + 1];
    source(&input, &mut output).unwrap();
    assert_eq!(
        output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    for result in [
        host_call(session, graph, &input, &mut output),
        host_call(session, native, &input, &mut output),
    ] {
        result.unwrap();
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
    input[N / 2] = f64::from_bits(0x7ff0_0000_0000_0001);
    input[N - 1] = f64::INFINITY;
    let Err(pcu_facade::PcuHostDispatchError::Backend(MetalError::Arithmetic(expected_fault))) =
        source(&input, &mut output)
    else {
        panic!("missing F64 source fault");
    };
    assert_eq!(expected_fault.invocation_id, u64::try_from(N / 2).unwrap());
    assert_eq!(
        output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    for result in [
        host_call(session, graph, &input, &mut output),
        host_call(session, native, &input, &mut output),
    ] {
        let Err(MetalError::Arithmetic(fault)) = result else {
            panic!("missing F64 control fault");
        };
        assert_eq!(fault, expected_fault);
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
    input.fill(2.0);
    source(&input, &mut output).unwrap();
    assert!(
        output[..N]
            .iter()
            .all(|value| value.to_bits() == (-2.0_f64).to_bits())
    );
}
