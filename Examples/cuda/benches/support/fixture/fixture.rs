//! First-call, post-fault reset and exact output proofs, outside Criterion timing.
#[rustfmt::skip]
use fusion_pcu::{
    PcuTensor,
};
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaHostKernelError,
    CudaOwnedDispatchBackend,
};
use super::native::Native;
use super::source::transform;
use super as support;

pub fn host<const N: usize>(
    backend: &CudaOwnedDispatchBackend,
    prepared: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), CudaHostKernelError>,
) -> Native {
    let mut native = support::cold("native_prepare_and_device_allocations", || {
        Native::new::<N>(backend)
    });
    let mut output = support::cold("host_allocations", || vec![0.0; N]);
    let mut bad = support::oracle::input(N, 0);
    bad[0] = f32::INFINITY;
    // A first fault, followed by success, proves status initialization and post-fault reset.
    support::oracle::verify_fault(&support::cold("ordinary_first_call_fault", || {
        transform::<N>(&bad, &mut output).unwrap_err()
    }));
    assert!(matches!(prepared(&bad, &mut output).unwrap_err(),
        CudaHostKernelError::CheckedExecutionFault(actual) if actual == support::oracle::fault()));
    native.upload(&support::oracle::bytes(&bad));
    assert_eq!(native.submit(), 5); // CUDA checked ABI: invocation zero, InvalidFloatingOperand tag.
    for phase in [1, 17] {
        let input = support::oracle::input(N, phase);
        transform::<N>(&input, &mut output).unwrap();
        support::oracle::verify(&input, &output);
        prepared(&input, &mut output).unwrap();
        support::oracle::verify(&input, &output);
        native.upload(&support::oracle::bytes(&input));
        assert_eq!(native.submit(), u64::MAX);
        support::oracle::verify_bytes(&input, native.readback());
    }
    #[cfg(feature = "allocation-census")]
    {
        let input = support::oracle::input(N, 1);
        let bytes = support::oracle::bytes(&input);
        support::allocations::census(&format!("host/ordinary/{N}"), || {
            transform::<N>(&input, &mut output).unwrap();
        });
        support::oracle::verify(&input, &output);
        support::allocations::census(&format!("host/prepared/{N}"), || {
            prepared(&input, &mut output).unwrap();
        });
        support::oracle::verify(&input, &output);
        support::allocations::census(&format!("host/native/{N}"), || native.host(&bytes));
        support::oracle::verify_bytes(&input, native.result());
    }
    native
}

pub struct Resident {
    pub banks: [PcuTensor<f32>; 2],
    pub destination: PcuTensor<f32>,
    pub native: Native,
    inputs: [Vec<f32>; 2],
    output: Vec<f32>,
}

impl Resident {
    pub fn new<const N: usize>(backend: &CudaOwnedDispatchBackend) -> Self {
        let mut output = vec![0.0; N];
        let mut bad = support::oracle::input(N, 0);
        bad[0] = f32::INFINITY;
        let inputs = [support::oracle::input(N, 1), support::oracle::input(N, 17)];
        let resident_inputs = inputs
            .each_ref()
            .map(|input| support::source::identity(input).unwrap());
        let resident_bad = support::source::identity(&bad).unwrap();
        let mut resident_output = support::source::identity(&vec![0.0; N]).unwrap();
        support::oracle::verify_fault(
            &transform::<N>(&resident_bad, &mut resident_output).unwrap_err(),
        );
        // Fatal mutable resident faults invalidate the destination; reconstruct it outside timing.
        resident_output = support::source::identity(&vec![0.0; N]).unwrap();
        let mut native_resident = Native::new::<N>(backend);
        native_resident.upload(&support::oracle::bytes(&bad));
        assert_eq!(native_resident.submit(), 5);
        for (bank, input) in inputs.iter().enumerate() {
            native_resident.upload_bank(bank, &support::oracle::bytes(input));
            assert_eq!(native_resident.submit_bank(bank), u64::MAX);
            support::oracle::verify_bytes(input, native_resident.readback());
        }
        for (input, resident) in inputs.iter().zip(&resident_inputs) {
            transform::<N>(resident, &mut resident_output).unwrap();
            support::cold("terminal_resident_output_readback", || {
                resident_output.read_into(&mut output).unwrap();
            });
            support::oracle::verify(input, &output);
        }
        #[cfg(feature = "allocation-census")]
        {
            support::allocations::census(&format!("resident/ordinary/{N}"), || {
                transform::<N>(&resident_inputs[0], &mut resident_output).unwrap();
            });
            resident_output.read_into(&mut output).unwrap();
            support::oracle::verify(&inputs[0], &output);
            support::allocations::census(&format!("resident/native/{N}"), || {
                assert_eq!(native_resident.submit_bank(0), u64::MAX);
            });
            support::oracle::verify_bytes(&inputs[0], native_resident.readback());
        }
        Self {
            banks: resident_inputs,
            destination: resident_output,
            native: native_resident,
            inputs,
            output,
        }
    }

    pub fn verify(&mut self, last_pcu_bank: usize, last_native_bank: usize) {
        self.destination.read_into(&mut self.output).unwrap();
        support::oracle::verify(&self.inputs[last_pcu_bank], &self.output);
        support::oracle::verify_bytes(&self.inputs[last_native_bank], self.native.readback());
    }
}

/// Criterion measures only the full host call; every actual result is checked after its timer.
pub fn measure_host<const N: usize>(
    iterations: u64,
    mut call: impl FnMut(&[f32], &mut [f32]),
) -> std::time::Duration {
    let mut input = support::oracle::input(N, 1);
    let mut output = vec![0.0; N];
    let mut elapsed = std::time::Duration::ZERO;
    for iteration in 0..iterations {
        input[0] = f32::from(u16::try_from(iteration % 31).unwrap());
        let start = std::time::Instant::now();
        call(&input, &mut output);
        elapsed += start.elapsed();
        support::oracle::verify(&input, &output);
    }
    elapsed
}

/// Native byte preparation and its oracle are outside the same full-host boundary.
pub fn measure_native<const N: usize>(iterations: u64, native: &mut Native) -> std::time::Duration {
    let mut input = support::oracle::input(N, 1);
    let mut bytes = support::oracle::bytes(&input);
    let mut elapsed = std::time::Duration::ZERO;
    for iteration in 0..iterations {
        input[0] = f32::from(u16::try_from(iteration % 31).unwrap());
        bytes[..4].copy_from_slice(&input[0].to_le_bytes());
        let start = std::time::Instant::now();
        native.host(&bytes);
        elapsed += start.elapsed();
        support::oracle::verify_bytes(&input, native.result());
    }
    elapsed
}
