//! Cold shape checks and actual retained algorithm/completion acceptance.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuHostArgument,
    PcuPrecisionPolicy,
    PcuScalar,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    CublasEnvironmentSnapshot,
    CublasNumericalConfig,
    CudaCompletionBatch,
};
use super::*;

fn descriptor(
    left: [usize; 2],
    right: [usize; 2],
    transpose: [bool; 2],
    scalar: PcuScalarType,
    precision: PcuPrecisionPolicy,
) -> Result<CublasLtMatmulDescriptor, CublasError> {
    CublasLtMatmulDescriptor::new(
        left,
        right,
        transpose,
        CublasNumericalConfig::new(scalar, precision, CublasEnvironmentSnapshot::capture())?,
    )
}
#[test]
fn physical_row_major_descriptor_checks_all_transposes_and_extents() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            for transpose in [[false, false], [true, false], [false, true], [true, true]] {
                let left = if transpose[0] { [9, 17] } else { [17, 9] };
                let right = if transpose[1] { [11, 9] } else { [9, 11] };
                let value = descriptor(left, right, transpose, scalar, precision).unwrap();
                assert_eq!(value.output, [17, 11]);
                assert_eq!(value.left, left);
                assert_eq!(value.right, right);
                let width = if scalar == PcuScalarType::F32 { 4 } else { 8 };
                assert_eq!(
                    value.bytes,
                    [17 * 9 * width, 9 * 11 * width, 17 * 11 * width]
                );
            }
        }
    }
    for (left, right) in [
        ([0, 3], [3, 2]),
        ([2, 3], [2, 4]),
        ([2, 3], [3, 0]),
        ([usize::MAX, 1], [1, 1]),
        ([1 << 32, 1 << 32], [1 << 32, 1]),
    ] {
        assert!(
            descriptor(
                left,
                right,
                [false; 2],
                PcuScalarType::F64,
                PcuPrecisionPolicy::Preserve
            )
            .is_err()
        );
    }
}
trait Real: PcuScalar + PartialEq + std::fmt::Debug {
    fn integer(value: i32) -> Self;
}
impl Real for f32 {
    #[allow(clippy::cast_precision_loss)] // Fixture integers and exact sums stay far below 2^24.
    fn integer(value: i32) -> Self {
        value as Self
    }
}
impl Real for f64 {
    fn integer(value: i32) -> Self {
        Self::from(value)
    }
}
fn upload<T: PcuScalar>(buffer: &mut DeviceBuffer, values: &[T]) {
    buffer
        .copy_from(PcuHostArgument::read(PcuBindingRef::new(0, 0), values).bytes())
        .unwrap();
}
fn download<T: PcuScalar>(buffer: &DeviceBuffer, values: &mut [T]) {
    buffer
        .copy_to(
            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values)
                .bytes_mut()
                .unwrap(),
        )
        .unwrap();
}
fn exercise<T: Real>(
    runtime: &CudaRuntime,
    rows: usize,
    inner: usize,
    columns: usize,
    transpose: [bool; 2],
    precision: PcuPrecisionPolicy,
) {
    let stream = runtime.create_stream().unwrap();
    let left_shape = if transpose[0] {
        [inner, rows]
    } else {
        [rows, inner]
    };
    let right_shape = if transpose[1] {
        [columns, inner]
    } else {
        [inner, columns]
    };
    let description = descriptor(left_shape, right_shape, transpose, T::TYPE, precision).unwrap();
    let plan = CublasLtMatmulPlan::prepare(runtime, &stream, description.clone()).unwrap();
    let identity = plan.identity();
    assert!(identity.algorithm_id >= 0 && identity.workspace_bytes <= WORKSPACE_LIMIT);
    assert_eq!(
        identity.compute_type,
        if T::TYPE == PcuScalarType::F64 {
            70
        } else if precision == PcuPrecisionPolicy::BackendOptimized {
            77
        } else {
            68
        }
    );
    let mut left = runtime.allocate(description.bytes[0]).unwrap();
    let mut right = runtime.allocate(description.bytes[1]).unwrap();
    let output = runtime.allocate(description.bytes[2]).unwrap();
    for phase in 0..3 {
        let mut left_values = vec![T::integer(0); rows * inner];
        let mut right_values = vec![T::integer(0); inner * columns];
        let left_integer = |row, depth| i32::try_from((row + 3 * depth + phase) % 7).unwrap() - 3;
        let right_integer =
            |depth, column| i32::try_from((2 * depth + column + phase) % 7).unwrap() - 3;
        for row in 0..rows {
            for depth in 0..inner {
                left_values[if transpose[0] {
                    depth * rows + row
                } else {
                    row * inner + depth
                }] = T::integer(left_integer(row, depth));
            }
        }
        for depth in 0..inner {
            for column in 0..columns {
                right_values[if transpose[1] {
                    column * inner + depth
                } else {
                    depth * columns + column
                }] = T::integer(right_integer(depth, column));
            }
        }
        let expected: Vec<_> = (0..rows * columns)
            .map(|cell| {
                T::integer(
                    (0..inner)
                        .map(|depth| {
                            left_integer(cell / columns, depth)
                                * right_integer(depth, cell % columns)
                        })
                        .sum(),
                )
            })
            .collect();
        upload(&mut left, &left_values);
        upload(&mut right, &right_values);
        plan.execute(&left, &right, &output).unwrap();
        let mut actual = vec![T::integer(0); rows * columns];
        download(&output, &mut actual);
        assert_eq!(actual, expected);
        assert_eq!(identity, plan.identity());
    }
}
#[test]
#[ignore = "requires authorized idle CUDA hardware and stable cuBLASLt; run serially"]
fn real_lt_f32_f64_all_transposes_odd_rectangular_changing_inputs() {
    let runtime = CudaRuntime::new(0).unwrap();
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        for (rows, inner, columns) in [(2, 3, 2), (17, 9, 11), (64, 96, 32)] {
            for transpose in [[false, false], [true, false], [false, true], [true, true]] {
                exercise::<f32>(&runtime, rows, inner, columns, transpose, precision);
                exercise::<f64>(&runtime, rows, inner, columns, transpose, precision);
            }
        }
    }
}
#[test]
#[ignore = "requires authorized idle CUDA hardware and stable cuBLASLt; run serially"]
fn retained_plan_workspace_and_operands_obey_terminal_batch_and_queue_identity() {
    let runtime = CudaRuntime::new(0).unwrap();
    let stream = runtime.create_stream().unwrap();
    let plan = CublasLtMatmulPlan::prepare(
        &runtime,
        &stream,
        descriptor(
            [2, 2],
            [2, 2],
            [false; 2],
            PcuScalarType::F32,
            PcuPrecisionPolicy::Preserve,
        )
        .unwrap(),
    )
    .unwrap();
    let mut left = runtime.allocate(16).unwrap();
    let mut right = runtime.allocate(16).unwrap();
    let output = runtime.allocate(16).unwrap();
    upload(&mut left, &[1.0_f32, 2.0, 3.0, 4.0]);
    upload(&mut right, &[1.0_f32, 0.0, 0.0, 1.0]);
    let small = runtime.allocate(4).unwrap();
    let mut batch = CudaCompletionBatch::new(&stream);
    assert!(
        plan.submit_into_batch(&mut batch, &small, &right, &output)
            .is_err()
    );
    assert!(plan.is_usable());
    assert!(
        plan.submit_into_batch(&mut batch, &left, &right, &left)
            .is_err()
    );
    assert!(plan.is_usable());
    plan.submit_into_batch(&mut batch, &left, &right, &output)
        .unwrap();
    assert!(!plan.is_usable());
    let mut unrelated = CudaCompletionBatch::new(&stream);
    assert_eq!(
        plan.submit_into_batch(&mut unrelated, &left, &right, &output),
        Err(CublasError::Busy)
    );
    let other_stream = runtime.create_stream().unwrap();
    assert_eq!(
        plan.submit_into_batch(
            &mut CudaCompletionBatch::new(&other_stream),
            &left,
            &right,
            &output
        ),
        Err(CublasError::DifferentStream)
    );
    plan.submit_into_batch(&mut batch, &left, &right, &output)
        .unwrap();
    assert!(output.copy_to(&mut [0; 16]).is_err());
    let retained = Rc::downgrade(&plan.root);
    let mut completion = batch.finish().unwrap();
    drop(batch);
    drop(plan);
    drop(left);
    drop(right);
    assert!(retained.upgrade().is_some());
    completion.wait().unwrap();
    assert!(retained.upgrade().is_none());
    let mut actual = [0.0_f32; 4];
    download(&output, &mut actual);
    assert_eq!(
        actual.map(f32::to_bits),
        [1.0_f32, 2.0, 3.0, 4.0].map(f32::to_bits)
    );
}
