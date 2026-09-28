use core::cell::Cell;
use fusion_pcu_macros::pcu;
use pcu_alias::{PcuHostArgument, PcuHostKernelBackend, PcuPreparedHostKernel, PcuScalar};

extern crate pcu_alias;

#[pcu(invocations = R * C, crate_path = ::pcu_alias)]
fn generic_matrix_copy<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    output[id / C][id % C] = input[id / C][id % C];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TestError;

struct TestBackend {
    prepares: Cell<usize>,
    expected_scalar: pcu_alias::PcuScalarType,
    input_pointer: usize,
    output_pointer: usize,
    byte_len: usize,
}

struct TestPrepared {
    input_pointer: usize,
    output_pointer: usize,
    byte_len: usize,
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
        assert_eq!(kernel.bindings.len(), 2);
        assert!(kernel.bindings.iter().all(|binding| {
            binding.binding_type
                == pcu_alias::PcuBindingType::Value(pcu_alias::PcuValueType::Scalar(
                    self.expected_scalar,
                ))
        }));
        assert!(matches!(
            kernel.ops,
            [
                pcu_alias::PcuDispatchOp::Data(pcu_alias::PcuDispatchDataOp::BindingLoad {
                    index: pcu_alias::PcuDispatchIndex::InvocationId,
                    ..
                }),
                pcu_alias::PcuDispatchOp::Data(pcu_alias::PcuDispatchDataOp::BindingStore {
                    index: pcu_alias::PcuDispatchIndex::InvocationId,
                    ..
                }),
                pcu_alias::PcuDispatchOp::Control(pcu_alias::PcuDispatchControlOp::Return),
            ]
        ));
        Ok(TestPrepared {
            input_pointer: self.input_pointer,
            output_pointer: self.output_pointer,
            byte_len: self.byte_len,
        })
    }
}

impl PcuPreparedHostKernel for TestPrepared {
    type Error = TestError;

    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        assert_eq!(arguments.len(), 2);
        assert_eq!(arguments[0].bytes().len(), self.byte_len);
        assert_eq!(arguments[1].bytes().len(), self.byte_len);
        assert_eq!(arguments[0].access(), pcu_alias::PcuBindingAccess::ReadOnly);
        assert_eq!(
            arguments[1].access(),
            pcu_alias::PcuBindingAccess::ReadWrite
        );
        assert_eq!(arguments[0].bytes().as_ptr() as usize, self.input_pointer);
        assert_eq!(arguments[1].bytes().as_ptr() as usize, self.output_pointer);
        let (input, output) = arguments.split_at_mut(1);
        output[0]
            .bytes_mut()
            .expect("mutable generic matrix")
            .copy_from_slice(input[0].bytes());
        Ok(())
    }
}

fn assert_generic_matrix_copy<T: PcuScalar>(input: [[T; 3]; 2]) {
    let mut output = [[input[0][0]; 3]; 2];
    let expected_bytes =
        PcuHostArgument::read(pcu_alias::PcuBindingRef::new(0, 0), input.as_flattened())
            .bytes()
            .to_vec();
    let backend = TestBackend {
        prepares: Cell::new(0),
        expected_scalar: T::TYPE,
        input_pointer: input[0].as_ptr() as usize,
        output_pointer: output[0].as_ptr() as usize,
        byte_len: 6 * T::HOST_SIZE,
    };
    let mut call = generic_matrix_copy_prepare::<T, 2, 3, _>(&backend)
        .expect("generic matrix identity preparation");
    assert_eq!(backend.prepares.get(), 1);
    call(&input, &mut output).expect("generic matrix identity call");
    let output_argument =
        PcuHostArgument::read(pcu_alias::PcuBindingRef::new(0, 0), output.as_flattened());
    let output_bytes = output_argument.bytes();
    assert_eq!(output_bytes, expected_bytes);
    assert_eq!(backend.prepares.get(), 1);
}

#[test]
fn generic_rank_two_identity_flattens_without_copy_for_distinct_scalar_types() {
    assert_generic_matrix_copy([[1.0_f64, -0.0, f64::INFINITY], [f64::NAN, 4.0, 5.0]]);
    assert_generic_matrix_copy([[1_u32, u32::MAX, 3], [4, 5, 6]]);
}
