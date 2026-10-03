//! Minimum native CPU work owns its RAM directly; safe-endpoint diagnostics remain separate.
#[path = "../../helper_ordered_float_maps/raw/ffi/ffi.rs"]
mod ffi;
mod independent;
#[cfg(feature="allocation-census")]
#[rustfmt::skip]
use std::{
    cell::Cell,
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuDispatchSubmission,
    PcuHostArgument,
    PcuInvocationShape,
    PcuPreparedOwnedDispatch,
    PcuOwnedDispatchBackend,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmOwnedDispatchBackend,
    HipKernel,
    HipStreamHandle,
    HipRuntime,
    compile_hip_source_for_device,
};
use super::oracle::Format;
#[cfg(feature = "allocation-census")]
#[derive(Debug, Default, Clone, Copy)]
pub struct Api {
    pub host_to_device_copies: u64,
    pub device_to_host_copies: u64,
    pub launches: u64,
    pub event_records: u64,
    pub event_waits: u64,
}
fn initialized_ram<T: Format, const N: usize>(inputs: [&[T]; 2]) -> ([Vec<u8>; 2], [Vec<u8>; 2]) {
    let bytes = N * size_of::<T>();
    let banks = inputs.map(|input| {
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input)
            .bytes()
            .to_vec()
    });
    assert!(banks.iter().all(|input| input.len() == bytes));
    let initial = vec![T::sentinel(); N + 2];
    let readbacks = std::array::from_fn(|_| {
        PcuHostArgument::read(PcuBindingRef::new(0, 0), &initial)
            .bytes()
            .to_vec()
    });
    (banks, readbacks)
}

#[allow(clippy::large_enum_variant)] // Entire execution owner already resides in one cold pinned Box.
enum Execution {
    Lowered {
        kernel: HipKernel,
        stream: HipStreamHandle,
    },
    Handwritten {
        program: ffi::Program,
        ram: ffi::Endpoints,
        _runtime: HipRuntime,
    },
}
static OWNER_DROPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
struct State {
    sdk: ffi::Sdk,
    allocations: [ffi::Pointer; 4],
    execution: Execution,
    grid: [u32; 3],
    block: [u32; 3],
    roles: [u8; 3],
    banks: [Vec<u8>; 2],
    readbacks: [Vec<u8>; 2],
    status: [u8; 8],
    prefix_bytes: usize,
}
impl State {
    fn bank(&self, bank: usize) -> &[u8] {
        match &self.execution {
            Execution::Handwritten { ram, .. } => ram.bank(bank),
            Execution::Lowered { .. } => &self.banks[bank],
        }
    }
    fn replace_bank(&mut self, bank: usize, bytes: &[u8]) {
        match &mut self.execution {
            Execution::Handwritten { ram, .. } => ram.replace_bank(bank, bytes),
            Execution::Lowered { .. } => self.banks[bank].copy_from_slice(bytes),
        }
    }
    fn readbacks(&self) -> [&[u8]; 2] {
        match &self.execution {
            Execution::Handwritten { ram, .. } => ram.readbacks(),
            Execution::Lowered { .. } => self.readbacks.each_ref().map(Vec::as_slice),
        }
    }
    const fn status_address(&self) -> *const u8 {
        match &self.execution {
            Execution::Handwritten { ram, .. } => ram.status_address(),
            Execution::Lowered { .. } => self.status.as_ptr(),
        }
    }
}
impl Drop for State {
    fn drop(&mut self) {
        OWNER_DROPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for pointer in self.allocations {
            self.sdk.release(pointer);
        }
    }
}
pub struct Raw {
    state: Option<Box<State>>,
    #[cfg(feature = "allocation-census")]
    api: Rc<Cell<Api>>,
}
impl Raw {
    pub fn new<T: Format, const N: usize>(
        backend: &RocmOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
        banks: [&[T]; 2],
    ) -> Self {
        Self::create::<T, N>(backend, ir, banks, false)
    }
    pub fn handwritten<T: Format, const N: usize>(
        backend: &RocmOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
        banks: [&[T]; 2],
    ) -> Self {
        Self::create::<T, N>(backend, ir, banks, true)
    }
    fn create<T: Format, const N: usize>(
        backend: &RocmOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
        banks: [&[T]; 2],
        handwritten: bool,
    ) -> Self {
        let prepared = backend
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: ir,
                shape: PcuInvocationShape::invocations(
                    std::num::NonZeroU32::new(ir.entry.logical_shape[0]).unwrap(),
                ),
            })
            .unwrap();
        assert_eq!(prepared.binding_schema().len(), 3);
        let roles = std::array::from_fn(|slot| {
            u8::try_from(prepared.binding_schema()[slot].target.binding).unwrap()
        });
        assert_eq!(
            roles
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            [0, 1, 2].into_iter().collect()
        );
        let sdk = ffi::Sdk::new(backend.device_identity().device_id());
        let bytes = N * size_of::<T>();
        let allocations = [
            sdk.allocate(bytes),
            sdk.allocate(bytes),
            sdk.allocate(bytes),
            sdk.allocate(8),
        ];
        let (banks, readbacks) = initialized_ram::<T, N>(banks);
        let (grid, block) = prepared.launch_geometry();
        let state = State {
            sdk,
            allocations,
            execution: if handwritten {
                let runtime = HipRuntime::new(backend.device_identity().device_id()).unwrap();
                let source = independent::source(
                    T::SIGNED,
                    N,
                    ir.entry.logical_shape[0],
                    ir.numerical_requirements.range_policy,
                    ir.numerical_requirements.float_underflow,
                );
                let image = compile_hip_source_for_device(&runtime, &source).unwrap();
                let program = ffi::Program::new(&image);
                let ram = ffi::Endpoints::new(&program, [&banks[0], &banks[1]], &readbacks[0]);
                Execution::Handwritten {
                    program,
                    ram,
                    _runtime: runtime,
                }
            } else {
                Execution::Lowered {
                    kernel: prepared.hip_kernel(),
                    stream: prepared.stream_handle(),
                }
            },
            grid,
            block,
            roles: if handwritten { [0, 1, 2] } else { roles },
            banks: if handwritten {
                std::array::from_fn(|_| Vec::new())
            } else {
                banks
            },
            readbacks: if handwritten {
                std::array::from_fn(|_| Vec::new())
            } else {
                readbacks
            },
            status: u64::MAX.to_le_bytes(),
            prefix_bytes: bytes,
        };
        let mut owner = Self {
            state: Some(Box::new(state)),
            #[cfg(feature = "allocation-census")]
            api: Rc::new(Cell::new(Api::default())),
        };
        let state = owner.state.as_ref().unwrap();
        let code = match &state.execution {
            Execution::Handwritten { program, ram, .. } => {
                program.initialize_status(ram, state.allocations[3])
            }
            Execution::Lowered { .. } => state.sdk.upload(state.allocations[3], &state.status),
        };
        if code != 0 {
            owner.retire();
        }
        println!(
            "cold/owned_sdk_control: handwritten={handwritten} pinned_owner_bytes={} retained_input_bytes={} retained_readback_bytes={} status_bytes=8 SDK_allocations=4 SDK_device_bytes={} pinned_host_allocations={}",
            size_of::<State>(),
            2 * bytes,
            2 * (bytes + 2 * size_of::<T>()),
            3 * bytes + 8,
            if handwritten { 5 } else { 0 }
        );
        owner
    }
    fn retire(&mut self) -> ! {
        // Error-only conservative retirement retains actual RAM, raw allocation, module,
        // stream/context and SDK library roots. The boxed inline faultword keeps its submitted RAM
        // address stable when ownership moves into retirement. No caller RAM is submitted.
        self.retain_complete_state();
        panic!("native owned minimum retired after unproven SDK completion");
    }
    fn retain_complete_state(&mut self) {
        let address = self.state.as_ref().unwrap().status_address();
        let state = self.state.take().unwrap();
        assert_eq!(
            address,
            state.status_address(),
            "retirement must preserve submitted inline RAM address"
        );
        std::mem::forget(state);
    }
    pub fn known_terminal_retirement_witness(mut self) {
        // This uses the real conservative retirement path AFTER proven healthy quiescence.
        // It proves owner address/drop retention; it does not simulate driver loss.
        self.call(0);
        let before = OWNER_DROPS.load(std::sync::atomic::Ordering::Relaxed);
        self.retain_complete_state();
        assert_eq!(
            OWNER_DROPS.load(std::sync::atomic::Ordering::Relaxed),
            before
        );
        eprintln!(
            "known-terminal-retirement: stable boxed faultword RAM and full allocation/code/event/stream/runtime/SDK owner retained without destructor"
        );
    }
    pub fn call(&mut self, bank: usize) {
        assert_eq!(
            self.submit(bank),
            u64::MAX,
            "owned SDK healthy workload faulted"
        );
    }
    pub fn probe<T: Format>(&mut self, input: &[T]) -> u64 {
        let replacement = PcuHostArgument::read(PcuBindingRef::new(0, 0), input)
            .bytes()
            .to_vec();
        let previous = self.state.as_ref().unwrap().bank(0).to_vec();
        self.state.as_mut().unwrap().replace_bank(0, &replacement);
        self.reset_status();
        let word = self.submit(0);
        self.state.as_mut().unwrap().replace_bank(0, &previous);
        self.reset_status();
        word
    }
    fn reset_status(&mut self) {
        let state = self.state.as_mut().unwrap();
        state.status = u64::MAX.to_le_bytes();
        let code = match &mut state.execution {
            Execution::Handwritten { program, ram, .. } => {
                ram.reset();
                program.initialize_status(ram, state.allocations[3])
            }
            Execution::Lowered { .. } => state.sdk.upload(state.allocations[3], &state.status),
        };
        if code != 0 {
            self.retire();
        }
    }
    fn submit(&mut self, bank: usize) -> u64 {
        let direct = {
            let state = self.state.as_mut().unwrap();
            match &mut state.execution {
                Execution::Handwritten { program, ram, .. } => {
                    #[cfg(feature = "allocation-census")]
                    self.api.set(Api {
                        host_to_device_copies: self.api.get().host_to_device_copies + 1,
                        device_to_host_copies: self.api.get().device_to_host_copies + 3,
                        launches: self.api.get().launches + 1,
                        event_records: self.api.get().event_records + 1,
                        event_waits: self.api.get().event_waits + 1,
                    });
                    let code = program.submit(
                        ram,
                        bank,
                        state.allocations,
                        state.grid,
                        state.block,
                        state.prefix_bytes,
                    );
                    // Do not inspect pinned RAM on failure: native writes might still run.
                    Some((code, if code == 0 { ram.word() } else { 0 }))
                }
                Execution::Lowered { .. } => None,
            }
        };
        if let Some((code, word)) = direct {
            if code != 0 {
                self.retire();
            }
            return word;
        }
        let state = self.state.as_mut().unwrap();
        #[cfg(feature = "allocation-census")]
        self.api.set(Api {
            host_to_device_copies: self.api.get().host_to_device_copies + 1,
            ..self.api.get()
        });
        if state.sdk.upload(state.allocations[2], &state.banks[bank]) != 0 {
            self.retire();
        }
        let state = self.state.as_ref().unwrap();
        let pointers = state.roles.map(|role| state.allocations[usize::from(role)]);
        let pointers = [pointers[0], pointers[1], pointers[2], state.allocations[3]];
        let succeeded = match &state.execution {
            Execution::Lowered { kernel, stream } => {
                ffi::launch(kernel, stream, state.grid, state.block, pointers)
                    .and_then(|mut completion| completion.wait())
                    .is_ok()
            }
            Execution::Handwritten { .. } => unreachable!("direct owned stream returns above"),
        };
        if !succeeded {
            self.retire();
        }
        let state = self.state.as_mut().unwrap();
        #[cfg(feature = "allocation-census")]
        self.api.set(Api {
            device_to_host_copies: self.api.get().device_to_host_copies + 1,
            ..self.api.get()
        });
        if state.sdk.read(&mut state.status, state.allocations[3]) != 0 {
            self.retire();
        }
        let word = u64::from_le_bytes(state.status);
        // Native results remain private on every fault. Probe explicitly resets before reuse.
        if word != u64::MAX && word & (1 << 63) == 0 {
            return word;
        }
        for slot in 0..2 {
            let state = self.state.as_mut().unwrap();
            #[cfg(feature = "allocation-census")]
            self.api.set(Api {
                device_to_host_copies: self.api.get().device_to_host_copies + 1,
                ..self.api.get()
            });
            if state.sdk.read(
                &mut state.readbacks[slot][..state.prefix_bytes],
                state.allocations[slot],
            ) != 0
            {
                self.retire();
            }
        }
        word
    }
    pub fn verify<T: Format>(&self, expected_stage: &[T], output: &[T]) {
        let state = self.state.as_ref().unwrap();
        for (actual, expected) in state.readbacks().into_iter().zip([expected_stage, output]) {
            assert_eq!(
                &actual[..state.prefix_bytes],
                PcuHostArgument::read(PcuBindingRef::new(0, 0), expected).bytes()
            );
        }
        let sentinels = [T::sentinel(); 2];
        let tail = PcuHostArgument::read(PcuBindingRef::new(0, 0), &sentinels);
        for bytes in state.readbacks() {
            assert_eq!(&bytes[state.prefix_bytes..], tail.bytes());
        }
    }
    #[cfg(feature = "allocation-census")]
    pub fn counter(&self) -> Rc<Cell<Api>> {
        self.api.clone()
    }
}
