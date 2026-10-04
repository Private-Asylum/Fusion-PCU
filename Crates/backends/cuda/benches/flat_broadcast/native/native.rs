//! Independent raw SDK scalar broadcast with retained pinned RAM and one explicit stream.
#[path = "ffi/ffi.rs"]
mod ffi;
#[cfg(feature = "allocation-census")]
pub use ffi::Api;
#[rustfmt::skip]
use fusion_pcu::{PcuDispatchKernelIr,PcuBindingRef,PcuHostArgument,PcuDispatchSubmission,PcuInvocationShape,PcuPreparedOwnedDispatch,PcuOwnedDispatchBackend};
#[rustfmt::skip]
use fusion_pcu_cuda::{CudaOwnedDispatchBackend,CudaRuntime,compile_cuda_source_for_device};
use super::oracle::Format;
pub struct Native {
    owner: ffi::Owner,
}
impl Native {
    pub fn new<T: Format, const N: usize>(
        backend: &CudaOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
        banks: [&[T]; 2],
    ) -> Self {
        assert_scalar_broadcast::<N>(ir);
        let descriptor = fusion_pcu::describe_scalar_transport_map::<4>(ir, T::TYPE).unwrap();
        let schema = descriptor.resource(PcuBindingRef::new(0, 0)).unwrap();
        assert_eq!(schema.minimum_read_elements, 1);
        let prepared = backend
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: ir,
                shape: PcuInvocationShape::invocations(
                    std::num::NonZeroU32::new(ir.entry.logical_shape[0]).unwrap(),
                ),
            })
            .unwrap();
        let (grid, block) = prepared.launch_geometry();
        assert_eq!(prepared.binding_schema().len(), 2);
        let runtime = CudaRuntime::new(backend.device_identity().device_id()).unwrap();
        let code = independent_source(size_of::<T>(), N, ir.entry.logical_shape[0]);
        let image = compile_cuda_source_for_device(&runtime, &code).unwrap();
        let banks = banks.map(|values| {
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &values[..1])
                .bytes()
                .to_vec()
        });
        let initial = vec![T::pattern(251); N + 2];
        let bytes = PcuHostArgument::read(PcuBindingRef::new(0, 0), &initial);
        let owner = ffi::Owner::new(
            runtime,
            &image,
            [&banks[0], &banks[1]],
            bytes.bytes(),
            N * size_of::<T>(),
            grid,
            block,
        );
        eprintln!(
            "cold/native_flat: {} N={N} scalar_bytes={} device_bytes={} SDK_pinned_endpoints=3 host_bytes={} runtime/stream/module/event retained",
            T::LABEL,
            size_of::<T>(),
            (N + 1) * size_of::<T>(),
            (N + 4) * size_of::<T>()
        );
        Self { owner }
    }
    pub fn call(&mut self, bank: usize) {
        self.owner.call(bank);
    }
    #[allow(clippy::chunks_exact_to_as_chunks)] // Generic size_of::<T>() is not a stable const-generic array length.
    pub fn verify<T: Format>(&self, value: T) {
        let expected = value.encode_le();
        let sentinel = T::pattern(251).encode_le();
        let actual = self.owner.output();
        let prefix = actual.len() - 2 * size_of::<T>();
        for scalar in actual[..prefix].chunks_exact(size_of::<T>()) {
            assert_eq!(scalar, expected.as_ref());
        }
        for scalar in actual[prefix..].chunks_exact(size_of::<T>()) {
            assert_eq!(scalar, sentinel.as_ref());
        }
    }
    #[cfg(feature = "allocation-census")]
    pub fn counter(&self) -> std::rc::Rc<std::cell::Cell<ffi::Api>> {
        self.owner.counter()
    }
    pub fn known_terminal_retirement_witness(self) {
        self.owner.known_terminal_retirement_witness();
    }
}
fn assert_scalar_broadcast<const N: usize>(ir: &PcuDispatchKernelIr<'_>) {
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuDispatchControlOp,
        PcuDispatchDataOp,
        PcuDispatchIndex,
        PcuDispatchOp,
    };
    let (body, index) = match ir.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => {
            assert_eq!(*extent, u32::try_from(N).unwrap());
            (*body, PcuDispatchIndex::GridStrideId)
        }
        [
            load,
            store,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => {
            assert_eq!(ir.entry.logical_shape[0], u32::try_from(N).unwrap());
            let _ = (load, store);
            (&ir.ops[..2], PcuDispatchIndex::InvocationId)
        }
        _ => panic!("independent peer requires exact scalar broadcast body"),
    };
    match body {
        [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index: PcuDispatchIndex::BindingElementZero,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: output,
                index: actual_index,
                value,
            }),
        ] => {
            assert_eq!(*binding, PcuBindingRef::new(0, 0));
            assert_eq!(*output, PcuBindingRef::new(0, 1));
            assert_eq!(*actual_index, index);
            assert_eq!(value, result);
        }
        _ => panic!("independent peer refuses changed source operation sequence"),
    }
}
fn independent_source(bytes: usize, extent: usize, invocations: u32) -> String {
    let carrier = match bytes {
        1 => "using T=unsigned char;".to_owned(),
        2 => "using T=unsigned short;".to_owned(),
        4 => "using T=unsigned int;".to_owned(),
        8 => "using T=unsigned long long;".to_owned(),
        16 | 32 | 64 => format!("struct T {{ unsigned long long limbs[{}]; }};", bytes / 8),
        _ => panic!("unqualified carrier size"),
    };
    format!(
        "{carrier}\nstatic_assert(sizeof(T)=={bytes},\"exact transport size\");\nextern \"C\" __global__ void native_flat(const T* input,T* output) {{ unsigned start=blockIdx.x*blockDim.x+threadIdx.x; if(start>={invocations}u)return; T scalar=input[0]; for(unsigned long long lane=start;lane<{extent}ull;lane+={invocations}u) output[lane]=scalar; }}\n"
    )
}
