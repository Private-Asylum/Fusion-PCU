use core::cell::Cell;
use fusion_pcu_macros::pcu;
use pcu_alias::{
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};

extern crate pcu_alias;

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn copy(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}

#[pcu(invocations = N, crate_path = ::pcu_alias)]
fn generic_copy<T: pcu_alias::PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FakeError;

struct FakeBackend<'a> {
    prepares: &'a Cell<usize>,
    fail_calls: bool,
}

struct FakePrepared {
    scalar: pcu_alias::PcuScalarType,
    fail_calls: bool,
}

struct DetachedBackend;

fn call_through_host_helper<F>(
    mut call: F,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), FakeError>
where
    F: FnMut(&[f32], &mut [f32]) -> Result<(), FakeError>,
{
    call(input, output)
}

fn call_through_device_helper<F>(
    mut call: F,
    input: &pcu_alias::PcuDeviceBuffer<f32, core::cell::RefCell<Vec<f32>>>,
    output: &mut pcu_alias::PcuDeviceBuffer<f32, core::cell::RefCell<Vec<f32>>>,
) -> Result<(), FakeError>
where
    F: for<'input, 'output> FnMut(
        &'input pcu_alias::PcuDeviceBuffer<f32, core::cell::RefCell<Vec<f32>>>,
        &'output mut pcu_alias::PcuDeviceBuffer<f32, core::cell::RefCell<Vec<f32>>>,
    ) -> Result<(), FakeError>,
{
    call(input, output)
}

impl PcuHostKernelBackend for DetachedBackend {
    type Prepared = FakePrepared;
    type Error = FakeError;

    fn prepare_host_kernel(
        &self,
        kernel: &pcu_alias::PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let pcu_alias::PcuValueType::Scalar(scalar) = kernel.bindings[0]
            .binding_type
            .value_type()
            .expect("scalar binding")
        else {
            panic!("scalar binding expected");
        };
        Ok(FakePrepared {
            scalar,
            fail_calls: false,
        })
    }
}

impl PcuHostKernelBackend for FakeBackend<'_> {
    type Prepared = FakePrepared;
    type Error = FakeError;

    fn prepare_host_kernel(
        &self,
        kernel: &pcu_alias::PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        self.prepares.set(self.prepares.get() + 1);
        let pcu_alias::PcuValueType::Scalar(scalar) = kernel.bindings[0]
            .binding_type
            .value_type()
            .expect("scalar binding")
        else {
            panic!("scalar binding expected");
        };
        Ok(FakePrepared {
            scalar,
            fail_calls: self.fail_calls,
        })
    }
}

impl PcuPreparedHostKernel for FakePrepared {
    type Error = FakeError;

    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        assert_eq!(arguments[0].scalar(), self.scalar);
        assert_eq!(arguments[1].scalar(), self.scalar);
        if self.fail_calls {
            return Err(FakeError);
        }
        let (input_arg, output_args) = arguments.split_at_mut(1);
        let input = input_arg[0].bytes();
        let output = output_args[0].bytes_mut().expect("mutable output");
        output.copy_from_slice(input);
        Ok(())
    }
}

#[test]
fn prepared_typed_call_reuses_preparation_and_preserves_mutable_output() {
    let prepares = Cell::new(0);
    let backend = FakeBackend {
        prepares: &prepares,
        fail_calls: false,
    };
    let mut call = copy_prepare(&backend).expect("cold preparation");
    assert_eq!(prepares.get(), 1);

    let input = [1.0_f32, -0.0, f32::from_bits(0x7fc0_1234)];
    let mut output = [8.0_f32; 3];
    call(&input, &mut output).expect("first warm call");
    assert_eq!(output.map(f32::to_bits), input.map(f32::to_bits));

    let next = [4.0_f32, 5.0, 6.0];
    call(&next, &mut output).expect("second warm call");
    assert_eq!(output.map(f32::to_bits), next.map(f32::to_bits));
    assert_eq!(prepares.get(), 1);
}

#[test]
fn generic_prepare_specializes_scalar_and_call_errors_propagate() {
    let prepares = Cell::new(0);
    let backend = FakeBackend {
        prepares: &prepares,
        fail_calls: true,
    };
    let mut call = generic_copy_prepare::<u16, 3, _>(&backend).expect("generic preparation");
    let input = [1_u16, 0x8000, 0xffff];
    let mut output = [0_u16; 3];
    assert_eq!(call(&input, &mut output), Err(FakeError));
    assert_eq!(prepares.get(), 1);

    let mut wrapping_call =
        generic_wrapping_prepare::<u16, 2, _>(&backend).expect("wrapping preparation");
    let left = [u16::MAX, 2];
    let right = [1_u16, 3];
    let mut wrapped = [0_u16; 2];
    assert_eq!(wrapping_call(&left, &right, &mut wrapped), Err(FakeError));
    assert_eq!(prepares.get(), 2);
}

#[test]
fn returned_host_call_does_not_borrow_backend() {
    let call = {
        let backend = DetachedBackend;
        copy_prepare(&backend).expect("preparation owns executable")
    };
    let input = [7.0_f32, 8.0, 9.0];
    let mut output = [0.0_f32; 3];
    call_through_host_helper(call, &input, &mut output).expect("call after backend drop");
    assert_eq!(output.map(f32::to_bits), input.map(f32::to_bits));
}

#[test]
fn source_argument_names_may_match_generated_locals() {
    let backend = DetachedBackend;
    let mut call = resident_copy_prepare(&backend).expect("preparation");
    let prepared = [3.0_f32, 4.0, 5.0];
    let mut arguments = [0.0_f32; 3];
    call(&prepared, &mut arguments).expect("typed call");
    assert_eq!(arguments.map(f32::to_bits), prepared.map(f32::to_bits));
}

#[pcu(invocations = N + 1, crate_path = ::pcu_alias)]
fn generic_wrapping<T: pcu_alias::PcuWrappingInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    output[id] = left[id].wrapping_add(right[id]);
}

#[pcu(invocations = 3, crate_path = ::pcu_alias)]
fn resident_copy(prepared: &[f32], arguments: &mut [f32]) {
    let id = context.global_invocation_id;
    arguments[id] = prepared[id];
}

struct FakeDeviceBackend;
struct FakeDevicePrepared;

impl pcu_alias::PcuDeviceKernelBackend for FakeDeviceBackend {
    type Resource = core::cell::RefCell<Vec<f32>>;
    type Prepared = FakeDevicePrepared;
    type Error = FakeError;

    fn prepare_device_kernel(
        &self,
        _kernel: &pcu_alias::PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        Ok(FakeDevicePrepared)
    }
}

impl pcu_alias::PcuPreparedDeviceKernel for FakeDevicePrepared {
    type Resource = core::cell::RefCell<Vec<f32>>;
    type Error = FakeError;

    fn call(
        &mut self,
        arguments: &mut [pcu_alias::PcuDeviceArgument<'_, Self::Resource>],
    ) -> Result<(), Self::Error> {
        let input = arguments[0].resource().borrow().clone();
        let mut output = arguments[1].resource().borrow_mut();
        output.copy_from_slice(&input);
        Ok(())
    }
}

#[test]
fn device_preparation_keeps_typed_mutable_resident_call() {
    let backend = FakeDeviceBackend;
    let call = resident_copy_prepare_device(&backend).expect("device preparation");
    let input = pcu_alias::PcuDeviceBuffer::new(core::cell::RefCell::new(vec![2.0, 3.0, 4.0]), 3);
    let mut output = pcu_alias::PcuDeviceBuffer::new(core::cell::RefCell::new(vec![0.0; 3]), 3);
    call_through_device_helper(call, &input, &mut output).expect("resident call");
    assert_eq!(*output.resource().borrow(), [2.0, 3.0, 4.0]);
}
