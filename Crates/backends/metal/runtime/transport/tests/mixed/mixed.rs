//! Typed real resident capacities, ignored foreign mutable owner and dual publication.

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDeviceArgument,
    PcuDeviceBuffer,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuDispatchKernelIr,
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryHostAccess,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuScalar,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[rustfmt::skip]
use crate::{
    MetalError,
    MetalMemoryProvider,
    MetalMemoryResource,
    MetalMixedHostArgument,
    MetalPreparedTransportHostKernel,
    MetalSession,
};
use super::fixture;

trait Sample: PcuScalar {
    fn raw(lane: usize, phase: u8) -> Self;
}
macro_rules! samples {($($ty:ty=>$width:literal),+)=>{$(
    impl Sample for $ty {
        fn raw(lane:usize,phase:u8)->Self {
            let mut bytes=[0_u8;$width];
            for (index,byte) in bytes.iter_mut().enumerate() {
                *byte=match lane%7 {
                    0=>phase,
                    1=>0xff,
                    2=>if index==$width-1 {0x80} else {0},
                    3=>if index==0 {1} else {0},
                    _=>u8::try_from((lane*13+index*37)%256).unwrap().wrapping_add(phase),
                };
            }
            Self::decode_le(bytes)
        }
    }
)+};}
samples!(i8=>1,u8=>1,i16=>2,u16=>2,i32=>4,u32=>4,i64=>8,u64=>8,i128=>16,u128=>16,PcuI256=>32,PcuU256=>32,PcuI512=>64,PcuU512=>64,f32=>4,f64=>8,PcuF16Bits=>2,PcuBf16Bits=>2,PcuF8E4M3FnBits=>1,PcuF8E5M2Bits=>1,PcuF128Bits=>16,PcuF256Bits=>32);

fn bytes<T: PcuScalar>(values: &[T]) -> Vec<u8> {
    PcuHostArgument::read(PcuBindingRef::new(0, 0), values)
        .bytes()
        .to_vec()
}
fn owner<T: PcuScalar>(
    provider: &mut MetalMemoryProvider,
    values: &[T],
) -> PcuDeviceBuffer<T, MetalMemoryResource> {
    let raw = bytes(values);
    let mut resource = provider
        .allocate(PcuMemoryAllocationRequest {
            pool: PcuMemoryPoolId(186),
            size_bytes: u64::try_from(raw.len()).unwrap(),
            alignment_bytes: 1,
            access: PcuMemoryAccess::ReadWrite,
            host_access: PcuMemoryHostAccess::TransferOnly,
            require_device_local: false,
        })
        .unwrap();
    provider.transfer_to(&mut resource, 0, &raw).unwrap();
    PcuDeviceBuffer::new(resource, values.len())
}
fn read<T: PcuScalar>(
    provider: &mut MetalMemoryProvider,
    value: &PcuDeviceBuffer<T, MetalMemoryResource>,
) -> Vec<u8> {
    let mut raw = vec![0; value.len() * T::HOST_SIZE];
    provider
        .transfer_from(value.resource(), 0, &mut raw)
        .unwrap();
    raw
}

struct Case<'a, T: Sample> {
    prepared: &'a mut MetalPreparedTransportHostKernel,
    provider: &'a mut MetalMemoryProvider,
    foreign: &'a mut MetalMemoryProvider,
    phase: u8,
    mode: usize,
    input: Vec<T>,
    seed: Vec<T>,
}
impl<T: Sample> Case<'_, T> {
    fn run(&mut self, short: bool, foreign_actual: bool) {
        let input = owner(self.provider, &self.input);
        let seed = owner(self.provider, &self.seed);
        let mut ghost = owner(self.foreign, &[T::raw(5, self.phase)]);
        let foreign_input = owner(self.foreign, &self.input);
        let mut stage = vec![T::raw(9, self.phase); 20];
        let mut output = vec![T::raw(11, self.phase); if short { 16 } else { 22 }];
        let mut stage_owner = owner(self.provider, &stage);
        let mut output_owner = owner(self.provider, &output);
        let before_stage = bytes(&stage);
        let before_output = bytes(&output);
        let result = self.call(
            &input,
            &seed,
            &mut ghost,
            &foreign_input,
            &mut stage,
            &mut output,
            &mut stage_owner,
            &mut output_owner,
            foreign_actual,
        );
        if short || foreign_actual {
            let expected = if foreign_actual {
                PcuHostDispatchError::Backend(MetalError::ForeignSession)
            } else {
                PcuHostDispatchError::BufferTooSmall(fixture::OUTPUT)
            };
            assert_eq!(result, Err(expected));
            assert!(!self.prepared.last_call_may_have_written());
            assert_eq!(bytes(&stage), before_stage);
            assert_eq!(bytes(&output), before_output);
            assert_eq!(read(self.provider, &stage_owner), before_stage);
            assert_eq!(read(self.provider, &output_owner), before_output);
        } else {
            result.unwrap();
            assert!(self.prepared.last_call_may_have_written());
            let mut expected_stage = bytes(&self.seed[..1]).repeat(17);
            expected_stage.extend_from_slice(&before_stage[17 * T::HOST_SIZE..]);
            let mut expected_output = bytes(&self.input[..17]);
            expected_output.extend_from_slice(&before_output[17 * T::HOST_SIZE..]);
            let actual_stage = if self.mode & 1 != 0 {
                read(self.provider, &stage_owner)
            } else {
                bytes(&stage)
            };
            let actual_output = if self.mode & 2 != 0 {
                read(self.provider, &output_owner)
            } else {
                bytes(&output)
            };
            assert_eq!(actual_stage, expected_stage);
            assert_eq!(actual_output, expected_output);
        }
        assert!(!self.prepared.last_call_completion_uncertain());
        assert_eq!(read(self.provider, &input), bytes(&self.input));
        assert_eq!(read(self.provider, &seed), bytes(&self.seed));
        assert_eq!(read(self.foreign, &ghost), bytes(&[T::raw(5, self.phase)]));
    }

    #[allow(clippy::too_many_arguments)] // Explicit original five-role schema plus authentic retained test owners.
    fn call(
        &mut self,
        input: &PcuDeviceBuffer<T, MetalMemoryResource>,
        seed: &PcuDeviceBuffer<T, MetalMemoryResource>,
        ghost: &mut PcuDeviceBuffer<T, MetalMemoryResource>,
        foreign_input: &PcuDeviceBuffer<T, MetalMemoryResource>,
        stage: &mut [T],
        output: &mut [T],
        stage_owner: &mut PcuDeviceBuffer<T, MetalMemoryResource>,
        output_owner: &mut PcuDeviceBuffer<T, MetalMemoryResource>,
        foreign_actual: bool,
    ) -> Result<(), crate::MetalHostKernelError> {
        let input = if foreign_actual {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(fixture::INPUT, foreign_input))
        } else if self.mode & 4 != 0 {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(fixture::INPUT, input))
        } else {
            MetalMixedHostArgument::Host(PcuHostArgument::read(fixture::INPUT, &self.input[..17]))
        };
        let seed = if self.mode & 8 != 0 {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(fixture::SEED, seed))
        } else {
            MetalMixedHostArgument::Host(PcuHostArgument::read(fixture::SEED, &self.seed[..1]))
        };
        let stage = if self.mode & 1 != 0 {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                fixture::STAGE,
                stage_owner,
            ))
        } else {
            MetalMixedHostArgument::Host(PcuHostArgument::read_write(fixture::STAGE, stage))
        };
        let output = if self.mode & 2 != 0 {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                fixture::OUTPUT,
                output_owner,
            ))
        } else {
            MetalMixedHostArgument::Host(PcuHostArgument::read_write(fixture::OUTPUT, output))
        };
        // Deliberately reordered full declarations; ignored foreign mutable owner is real.
        self.prepared.call_mixed(&mut [
            output,
            seed,
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(fixture::GHOST, ghost)),
            stage,
            input,
        ])
    }
}

fn qualify<T: Sample>(session: &MetalSession, foreign: &MetalSession) {
    for grid in [false, true] {
        fixture::visit(
            T::TYPE,
            grid,
            PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
            |ir| {
                let mut prepared = session
                    .transport_host_backend()
                    .prepare_host_kernel(ir)
                    .unwrap();
                let mut provider = session.memory_provider(PcuMemoryPoolId(186));
                let mut foreign = foreign.memory_provider(PcuMemoryPoolId(186));
                for phase in 0..3 {
                    for mode in 0..16 {
                        let mut case = Case {
                            prepared: &mut prepared,
                            provider: &mut provider,
                            foreign: &mut foreign,
                            phase,
                            mode,
                            input: (0..24).map(|lane| T::raw(lane, phase)).collect(),
                            seed: (0..5).map(|lane| T::raw(lane + 1, phase)).collect(),
                        };
                        case.run(false, false);
                        case.run(true, false);
                        case.run(false, true);
                        case.run(false, false);
                    }
                }
            },
        );
    }
}

#[test]
#[ignore = "Requires actual Metal; all22 typed owners and sixteen host/resident layouts."]
fn all_twenty_two_real_mixed_resources_saved_ssa_atomic_hosts_and_foreign_ghosts() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    macro_rules! run {($($ty:ty),+)=>{$(qualify::<$ty>(&session,&foreign);)+};}
    run!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        PcuI256,
        PcuU256,
        PcuI512,
        PcuU512,
        f32,
        f64,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        PcuF128Bits,
        PcuF256Bits
    );
}
