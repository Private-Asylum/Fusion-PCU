//! Direct native launch of the same admitted kernel with matched retained status and storage.
#[path = "ffi/ffi.rs"]
mod ffi;
#[rustfmt::skip]
use std::num::NonZeroU32;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuDispatchSubmission,
    PcuHostArgument,
    PcuInvocationShape,
    PcuPreparedOwnedDispatch,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    RocmOwnedDispatchBackend,
    HipKernel,
    HipStreamHandle,
    DeviceBuffer,
};
use super::oracle::Format;
pub struct Native {
    kernel: HipKernel,
    stream: HipStreamHandle,
    grid: [u32; 3],
    block: [u32; 3],
    inputs: Vec<Vec<DeviceBuffer>>,
    input_counts: Vec<usize>,
    input_slots: Vec<usize>,
    roles: [u8; 4],
    role_count: usize,
    output: [DeviceBuffer; 2],
    status: DeviceBuffer,
    sentinel: bool,
    prefix_bytes: usize,
}
impl Native {
    pub fn new<T: Format, const N: usize>(
        backend: &RocmOwnedDispatchBackend,
        ir: &PcuDispatchKernelIr<'_>,
        bank_count: usize,
    ) -> Self {
        assert!((3..=4).contains(&ir.bindings.len()));
        let prepared = backend
            .prepare_dispatch(PcuDispatchSubmission {
                kernel: ir,
                shape: PcuInvocationShape::invocations(
                    NonZeroU32::new(ir.entry.logical_shape[0]).unwrap(),
                ),
            })
            .unwrap();
        let schema = prepared.binding_schema();
        let input_bindings = schema
            .iter()
            .filter(|binding| binding.access == fusion_pcu::PcuBindingAccess::ReadOnly)
            .map(|binding| binding.target)
            .collect::<Vec<_>>();
        let fusion_pcu::PcuBindingType::Value(value_type) = ir.bindings[0].binding_type else {
            unreachable!()
        };
        let description = fusion_pcu::assess_checked_integer_div_rem_operands(
            ir,
            value_type,
            fusion_pcu::PcuValueTypeCaps::for_scalar(value_type.scalar_type()),
        )
        .unwrap();
        let logical_counts = description.input_element_counts(N);
        let input_counts = input_bindings
            .iter()
            .map(|binding| {
                let position = description
                    .input_bindings()
                    .iter()
                    .position(|input| input == binding)
                    .unwrap();
                logical_counts[position]
            })
            .collect::<Vec<_>>();
        let input_slots = input_bindings
            .iter()
            .map(|binding| {
                description
                    .input_bindings()
                    .iter()
                    .position(|input| input == binding)
                    .unwrap()
            })
            .collect();
        let mut roles = [2; 4];
        for (role, binding) in roles.iter_mut().zip(schema) {
            *role = if binding.access == fusion_pcu::PcuBindingAccess::ReadOnly {
                u8::try_from(
                    input_bindings
                        .iter()
                        .position(|input| *input == binding.target)
                        .unwrap(),
                )
                .unwrap()
            } else {
                2 + u8::try_from(
                    description
                        .output_bindings()
                        .iter()
                        .position(|output| *output == binding.target)
                        .unwrap(),
                )
                .unwrap()
            };
        }
        let role_count = schema.len();
        let (grid, block) = prepared.launch_geometry();
        let mut output =
            std::array::from_fn(|_| backend.allocate((N + 2) * size_of::<T>()).unwrap());
        let initial = vec![T::SENTINEL; N + 2];
        for buffer in &mut output {
            buffer
                .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), &initial).bytes())
                .unwrap();
        }
        Self {
            kernel: prepared.hip_kernel(),
            stream: prepared.stream_handle(),
            grid,
            block,
            inputs: (0..bank_count)
                .map(|_| {
                    input_counts
                        .iter()
                        .map(|count| backend.allocate(count * size_of::<T>()).unwrap())
                        .collect()
                })
                .collect(),
            input_counts,
            input_slots,
            roles,
            role_count,
            output,
            status: backend.allocate(size_of::<u64>()).unwrap(),
            sentinel: false,
            prefix_bytes: N * size_of::<T>(),
        }
    }
    pub fn upload<T: Format>(&mut self, bank: usize, lhs: &[T], rhs: &[T]) {
        for ((buffer, slot), count) in self.inputs[bank]
            .iter_mut()
            .zip(&self.input_slots)
            .zip(&self.input_counts)
        {
            let values = [lhs, rhs][*slot];
            buffer
                .copy_from(
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &values[..*count]).bytes(),
                )
                .unwrap();
        }
    }
    pub fn submit(&mut self, bank: usize) -> u64 {
        if !self.sentinel {
            self.status.copy_from(&u64::MAX.to_le_bytes()).unwrap();
        }
        let mut buffers = [&self.status; 5];
        for (buffer, role) in buffers.iter_mut().zip(&self.roles[..self.role_count]) {
            *buffer = if *role >= 2 {
                &self.output[usize::from(*role - 2)]
            } else {
                &self.inputs[bank][usize::from(*role)]
            };
        }
        buffers[self.role_count] = &self.status;
        self.sentinel = false;
        ffi::launch(
            &self.kernel,
            &self.stream,
            self.grid,
            self.block,
            buffers,
            self.role_count + 1,
        )
        .unwrap()
        .wait()
        .unwrap();
        let mut word = [0; 8];
        self.status.copy_to(&mut word).unwrap();
        let word = u64::from_le_bytes(word);
        self.sentinel = word == u64::MAX;
        word
    }
    pub fn read<T: Format>(&self, q: &mut [T], r: &mut [T]) {
        for (buffer, values) in self.output.iter().zip([q, r]) {
            let mut argument = PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values);
            buffer
                .copy_to(&mut argument.bytes_mut().unwrap()[..self.prefix_bytes])
                .unwrap();
        }
    }
    pub fn host<T: Format>(&mut self, lhs: &[T], rhs: &[T], q: &mut [T], r: &mut [T]) {
        self.upload(0, lhs, rhs);
        assert_eq!(self.submit(0), u64::MAX);
        self.read(q, r);
    }
}
