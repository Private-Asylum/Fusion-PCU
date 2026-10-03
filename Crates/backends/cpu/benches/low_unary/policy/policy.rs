//! Complete request assertions occur only during cold benchmark preparation/scoring.
use std::sync::Mutex;

use fusion_pcu_cpu::PcuCpuHostBackend;
use pcu_facade::{
    PcuDispatchDataOp, PcuDispatchFloatUnaryOp, PcuDispatchKernelIr, PcuDispatchOp,
    PcuFloatUnderflowPolicy, PcuHostKernelBackend, PcuImplementationRequirements, PcuNumericalMode,
    PcuRangePolicy,
};

static EXPECTED: Mutex<Option<Policy>> = Mutex::new(None);

#[derive(Clone, Copy)]
pub struct Policy {
    pub requirements: PcuImplementationRequirements,
    operation: PcuDispatchFloatUnaryOp,
}

impl Policy {
    pub fn new(
        operation: PcuDispatchFloatUnaryOp,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
    ) -> Self {
        let mut requirements = PcuImplementationRequirements {
            float_underflow: underflow,
            range_policy: range,
            ..PcuImplementationRequirements::default()
        };
        if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
            requirements.numerical_mode = PcuNumericalMode::Strict;
        }
        Self {
            requirements,
            operation,
        }
    }

    pub fn expect(self) {
        *EXPECTED.lock().unwrap() = Some(self);
    }

    fn assert(self, kernel: &PcuDispatchKernelIr<'_>) {
        assert_eq!(kernel.numerical_requirements, self.requirements);
        let mut count = 0;
        for operation in kernel.ops {
            if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                op,
                underflow_policy,
                range_policy,
                ..
            }) = operation
            {
                assert_eq!(*op, self.operation);
                assert_eq!(*underflow_policy, self.requirements.float_underflow);
                assert_eq!(*range_policy, self.requirements.range_policy);
                count += 1;
            }
        }
        assert_eq!(count, 1);
    }
}

pub fn assert_expected(kernel: &PcuDispatchKernelIr<'_>) {
    EXPECTED
        .lock()
        .unwrap()
        .expect("cold unary benchmark policy")
        .assert(kernel);
}

pub struct Verified(pub PcuCpuHostBackend, pub Policy);

impl PcuHostKernelBackend for Verified {
    type Prepared = <PcuCpuHostBackend as PcuHostKernelBackend>::Prepared;
    type Error = <PcuCpuHostBackend as PcuHostKernelBackend>::Error;

    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        self.1.assert(kernel);
        self.0.prepare_host_kernel(kernel)
    }
}
