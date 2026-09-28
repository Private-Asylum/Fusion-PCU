//! Native nested-array signatures using proven row-major invocation decomposition.
#![cfg(feature = "rocm")]

use fusion_pcu::pcu;

mod arithmetic {
    use fusion_pcu::pcu;

    #[pcu]
    pub fn multiply(value: f32, factor: f32) -> f32 {
        value * factor
    }
}

#[pcu(invocations: R * C)]
fn matrix_scale_with_helper<const R: usize, const C: usize>(
    seed: &f32,
    input: &[[f32; C]; R],
    output: &mut [[f32; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    output[id / C][id % C] = arithmetic::multiply(input[id / C][id % C], *seed);
}

#[pcu(invocations: R * C)]
fn matrix_scale<const R: usize, const C: usize>(
    seed: &f32,
    input: &[[f32; C]; R],
    output: &mut [[f32; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    output[id / C][id % C] = input[id / C][id % C] * *seed;
}

#[pcu(invocations: R * C)]
fn matrix_shift<const R: usize, const C: usize>(
    seed: &f64,
    input: &[[f64; C]; R],
    output: &mut [[f64; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    output[id / C][id % C] = input[id / C][id % C] + *seed;
}

#[test]
#[ignore = "requires a working ROCm device"]
fn nested_arrays_preserve_shape_and_current_scalar_values() {
    fusion_pcu::global::use_defaults().unwrap();
    let input = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let mut output = [[-9.0_f32; 3]; 2];
    for seed in [0.5_f32, -2.0, 4.0] {
        matrix_scale(&seed, &input, &mut output).unwrap();
        let expected = input.map(|row| row.map(|value| (value * seed).to_bits()));
        assert_eq!(output.map(|row| row.map(f32::to_bits)), expected);
        matrix_scale_with_helper(&seed, &input, &mut output).unwrap();
        assert_eq!(output.map(|row| row.map(f32::to_bits)), expected);
    }
    // A second specialization shares no assumptions about the previous matrix's row width.
    let input = [[1.0_f32, 2.0, 3.0, 4.0, 5.0]];
    let mut output = [[0.0_f32; 5]];
    matrix_scale(&2.0, &input, &mut output).unwrap();
    assert_eq!(
        output.map(|row| row.map(f32::to_bits)),
        [[2.0_f32, 4.0, 6.0, 8.0, 10.0].map(f32::to_bits)]
    );
    let input = [[0.5_f64, 2.0], [4.0, -8.0], [16.0, 32.0]];
    let mut output = [[0.0_f64; 2]; 3];
    for seed in [0.25_f64, -4.0] {
        matrix_shift(&seed, &input, &mut output).unwrap();
        let expected = input.map(|row| row.map(|value| (value + seed).to_bits()));
        assert_eq!(output.map(|row| row.map(f64::to_bits)), expected);
    }
    fusion_pcu::global::clear_thread_cache().unwrap();
}

#[pcu(invocations: R * C)]
fn matrix_copy<T: fusion_pcu::PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    output[id / C][id % C] = input[id / C][id % C];
}

#[test]
#[ignore = "requires a working ROCm device"]
fn generic_nested_arrays_transport_values_without_numeric_conversion() {
    fusion_pcu::global::use_defaults().unwrap();
    let input = [[0_u32, u32::MAX, 0x8000_0001], [7, 42, 1024]];
    let mut output = [[99_u32; 3]; 2];
    matrix_copy(&input, &mut output).unwrap();
    assert_eq!(output, input);
    let input = [[i64::MIN, i64::MAX], [-1, 0], [1, 0x1234_5678_9abc_def0]];
    let mut output = [[99_i64; 2]; 3];
    matrix_copy(&input, &mut output).unwrap();
    assert_eq!(output, input);
    let input = [
        [f64::from_bits(1), -0.0],
        [f64::INFINITY, f64::from_bits(0x7ff8_1234_5678_9abc)],
    ];
    let mut output = [[99.0_f64; 2]; 2];
    matrix_copy(&input, &mut output).unwrap();
    assert_eq!(
        output.map(|row| row.map(f64::to_bits)),
        input.map(|row| row.map(f64::to_bits))
    );
    fusion_pcu::global::clear_thread_cache().unwrap();
}

#[pcu(invocations: R * C)]
fn matrix_scale_locals<const R: usize, const C: usize>(
    seed: &f32,
    input: &[[f32; C]; R],
    output: &mut [[f32; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    let row = id / C;
    let col = id % C;
    output[row][col] = arithmetic::multiply(input[row][col], *seed);
}

#[pcu(invocations: 3)]
fn matrix_copy_grid<T: fusion_pcu::PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
    output: &mut [[T; C]; R],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < R * C {
        let row = id / C;
        let col = id % C;
        output[row][col] = input[row][col];
        id += stride;
    }
}

#[pcu(invocations: 3)]
fn matrix_shift_grid<const R: usize, const C: usize>(
    seed: &f64,
    input: &[[f64; C]; R],
    output: &mut [[f64; C]; R],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < R * C {
        let row = id / C;
        let col = id % C;
        output[row][col] = input[row][col] + *seed;
        id += stride;
    }
}

#[test]
#[ignore = "requires a working ROCm device"]
fn canonical_matrix_locals_and_grid_stride_cover_each_element() {
    fusion_pcu::global::use_defaults().unwrap();
    let input = [[1.0_f32, -2.0, 3.0, -4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]];
    let mut output = [[-99.0_f32; 5]; 2];
    for seed in [0.5_f32, -2.0, 4.0] {
        matrix_scale_locals(&seed, &input, &mut output).unwrap();
        assert_eq!(
            output.map(|row| row.map(f32::to_bits)),
            input.map(|row| row.map(|value| (value * seed).to_bits()))
        );
    }
    let mut input = [[i64::MIN, i64::MAX, -1, 0, 1], [2, 3, 4, 5, 6]];
    let mut output = [[99_i64; 5]; 2];
    for iteration in 0..3 {
        input[1][0] = iteration;
        matrix_copy_grid(&input, &mut output).unwrap();
        assert_eq!(output, input);
    }
    let input = [
        [-0.0_f64, f64::from_bits(1), f64::INFINITY, 1.0, 2.0],
        [f64::from_bits(0x7ff8_1234_5678_9abc), -1.0, 3.0, 4.0, 5.0],
    ];
    let mut output = [[99.0_f64; 5]; 2];
    matrix_copy_grid(&input, &mut output).unwrap();
    assert_eq!(
        output.map(|row| row.map(f64::to_bits)),
        input.map(|row| row.map(f64::to_bits))
    );
    let input = [
        [0.5_f64, 2.0, 4.0, 8.0, 16.0],
        [-1.0, -2.0, -4.0, -8.0, -16.0],
    ];
    for seed in [0.25_f64, -4.0] {
        matrix_shift_grid(&seed, &input, &mut output).unwrap();
        assert_eq!(
            output.map(|row| row.map(f64::to_bits)),
            input.map(|row| row.map(|value| (value + seed).to_bits()))
        );
    }
    fusion_pcu::global::clear_thread_cache().unwrap();
}
