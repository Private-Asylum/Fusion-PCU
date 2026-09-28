use core::cell::Cell;
use fusion_pcu_macros::pcu;
use pcu_alias::{PcuHostArgument, PcuHostKernelBackend, PcuPreparedHostKernel};

extern crate pcu_alias;

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn matrix_scale<const R: usize, const C: usize>(
    seed: &f32,
    input: &[[f32; C]; R],
    output: &mut [[f32; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    output[id / C][id % C] = input[id / C][id % C] * *seed;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TestError;

struct TestBackend {
    prepares: Cell<usize>,
    input_pointer: Cell<usize>,
    output_pointer: Cell<usize>,
}

struct TestPrepared {
    input_pointer: usize,
    output_pointer: usize,
}

impl PcuHostKernelBackend for TestBackend {
    type Prepared = TestPrepared;
    type Error = TestError;

    fn prepare_host_kernel(
        &self,
        kernel: &pcu_alias::PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        self.prepares.set(self.prepares.get() + 1);
        assert_eq!(kernel.entry.logical_shape, [6, 1, 1]);
        assert!(kernel.ops.iter().any(|operation| matches!(
            operation,
            pcu_alias::PcuDispatchOp::Data(pcu_alias::PcuDispatchDataOp::BindingLoad {
                binding: pcu_alias::PcuBindingRef { binding: 0, .. },
                index: pcu_alias::PcuDispatchIndex::BindingElementZero,
                ..
            })
        )));
        Ok(TestPrepared {
            input_pointer: self.input_pointer.get(),
            output_pointer: self.output_pointer.get(),
        })
    }
}

impl PcuPreparedHostKernel for TestPrepared {
    type Error = TestError;

    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        assert_eq!(arguments.len(), 3);
        assert_eq!(arguments[1].bytes().len(), 6 * core::mem::size_of::<f32>());
        assert_eq!(arguments[2].bytes().len(), 6 * core::mem::size_of::<f32>());
        assert_eq!(arguments[1].access(), pcu_alias::PcuBindingAccess::ReadOnly);
        assert_eq!(
            arguments[2].access(),
            pcu_alias::PcuBindingAccess::ReadWrite
        );
        assert_eq!(arguments[1].bytes().as_ptr() as usize, self.input_pointer);
        assert_eq!(arguments[2].bytes().as_ptr() as usize, self.output_pointer);
        let (prefix, suffix) = arguments.split_at_mut(2);
        suffix[0]
            .bytes_mut()
            .expect("mutable matrix")
            .copy_from_slice(prefix[1].bytes());
        Ok(())
    }
}

#[test]
fn prepared_host_call_flattens_matrices_as_zero_copy_views() {
    let seed = 1.5_f32;
    let input = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let mut output = [[0.0_f32; 3]; 2];
    let backend = TestBackend {
        prepares: Cell::new(0),
        input_pointer: Cell::new(input[0].as_ptr() as usize),
        output_pointer: Cell::new(output[0].as_ptr() as usize),
    };
    let mut call = matrix_scale_prepare::<2, 3, _>(&backend).expect("matrix preparation");
    assert_eq!(backend.prepares.get(), 1);
    call(&seed, &input, &mut output).expect("host matrix execution");
    assert_eq!(
        output.map(|row| row.map(f32::to_bits)),
        input.map(|row| row.map(f32::to_bits))
    );
    assert_eq!(backend.prepares.get(), 1);
}
