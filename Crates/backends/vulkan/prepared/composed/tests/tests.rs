//! Pure declaration/span tests: the capture backend never initializes a Vulkan device.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuHostKernelBackend,
};
#[path = "graph/graph.rs"]
mod graph;

struct AdmissionCase {
    portable: bool,
    write_only: bool,
}
struct Admission(Option<PcuSpirvComposedProfile>);
impl PcuHostKernelBackend for AdmissionCase {
    type Prepared = Admission;
    type Error = PcuVulkanError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Admission, Self::Error> {
        let mut candidate = *kernel;
        let mut bindings = kernel.bindings.to_vec();
        if self.write_only {
            for binding in &mut bindings {
                if binding.access == PcuBindingAccess::ReadWrite {
                    binding.access = PcuBindingAccess::WriteOnly;
                }
            }
        }
        candidate.bindings = &bindings;
        if self.portable {
            candidate
                .numerical_requirements
                .numerical_options
                .reproducibility = fusion_pcu::PcuReproducibility::PortableV1;
        }
        Ok(Admission(PcuVulkanPreparedComposed::admitted_profile(
            &candidate,
        )))
    }
}
impl PcuPreparedHostKernel for Admission {
    type Error = PcuVulkanError;
    fn call(&mut self, _: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        Err(PcuVulkanError::UnsupportedPreparedProfile)
    }
}

#[test]
fn ordinary_cold_admission_preserves_access_and_reproducibility() {
    macro_rules! width {
        ($ty:ty) => {{
            fn case() {
                for portable in [false, true] {
                    for write_only in [false, true] {
                        let captured = graph::prepare::<$ty, 65, _>(
                            &AdmissionCase {
                                portable,
                                write_only,
                            },
                            PcuFloatUnderflowPolicy::IeeeAfterRounding,
                            PcuRangePolicy::Clamp,
                        );
                        assert_eq!(captured.0.is_some(), !portable && !write_only);
                        if let Some(profile) = captured.0 {
                            assert_eq!(profile.requirements().range_policy, PcuRangePolicy::Clamp);
                            assert_eq!(profile.declarations().len(), 2);
                            assert_eq!(profile.resources().len(), 2);
                        }
                    }
                }
            }
            case();
        }};
    }
    width!(fusion_pcu::PcuF16Bits);
    width!(fusion_pcu::PcuBf16Bits);
    width!(fusion_pcu::PcuF8E4M3FnBits);
    width!(fusion_pcu::PcuF8E5M2Bits);
    width!(f32);
    width!(f64);
}

struct Capture;
struct Captured(PcuSpirvComposedProfile);
impl PcuHostKernelBackend for Capture {
    type Prepared = Captured;
    type Error = fusion_pcu_spirv::PcuSpirvError;
    fn prepare_host_kernel(&self, ir: &PcuDispatchKernelIr<'_>) -> Result<Captured, Self::Error> {
        fusion_pcu_spirv::validate_composed_float_map(ir).map(Captured)
    }
}
impl PcuPreparedHostKernel for Captured {
    type Error = PcuVulkanError;
    fn call(&mut self, args: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        schema::validate(args, &self.0).map(|_| ())
    }
}

#[test]
fn declaration_order_prefixes_and_access_are_frozen_before_native_work() {
    let mut captured = graph::prepare::<f32, 65, _>(
        &Capture,
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    );
    let input = [1_f32; 65];
    let mut output = [17_f32; 68];
    let a = PcuBindingRef::new(0, 1);
    let c = PcuBindingRef::new(0, 0);
    assert_eq!(
        schema::validate(
            &[
                PcuHostArgument::read(a, &input),
                PcuHostArgument::read_write(c, &mut output)
            ],
            &captured.0
        )
        .unwrap()[..2],
        [0, 1]
    );
    assert!(
        captured
            .call(&mut [
                PcuHostArgument::read(a, &input[..64]),
                PcuHostArgument::read_write(c, &mut output)
            ])
            .is_err()
    );
    assert!(
        captured
            .call(&mut [
                PcuHostArgument::read(a, &input),
                PcuHostArgument::read_write(c, &mut output[..64])
            ])
            .is_err()
    );
    assert!(
        captured
            .call(&mut [
                PcuHostArgument::read(a, &input),
                PcuHostArgument::read(c, &input)
            ])
            .is_err()
    );
    assert!(
        captured
            .call(&mut [
                PcuHostArgument::read(a, &input),
                PcuHostArgument::read_write(a, &mut output)
            ])
            .is_err()
    );
    assert!(
        captured
            .call(&mut [
                PcuHostArgument::read(a, &[1_f64; 65]),
                PcuHostArgument::read_write(c, &mut output)
            ])
            .is_err()
    );
    assert!(
        output
            .into_iter()
            .all(|value| value.to_bits() == 17_f32.to_bits())
    );
}
