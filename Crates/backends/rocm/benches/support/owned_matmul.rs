//! f32 profile for the shared fresh-output `MatMul` driver.

#[rustfmt::skip]
use std::{
    error::Error,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuTensor,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    Rocblas,
};

#[path = "alloc.rs"]
#[allow(dead_code)]
pub mod alloc;

use super::owned_matmul_common::{MatMulProfile, run_case, select_backend};

const SMALL: usize = 4;
const LARGE: usize = 256;

#[fusion_pcu::pcu(invocations = N)]
fn matrix_refresh<const R: usize, const C: usize, const N: usize>(
    input: &[[f32; C]; R],
    output: &mut [[f32; C]; R],
) {
    let id = pcu::context::global_invocation_id();
    let row = id / C;
    let column = id % C;
    output[row][column] = input[row][column];
}

pub struct F32Profile;

impl MatMulProfile for F32Profile {
    type Scalar = f32;

    const GROUP: &'static str = "owned_matmul_fresh_output";
    const CENSUS_LABEL: &'static str = "Owned MatMul warm Rust heap census";
    const DIAGNOSTIC_LABEL: &'static str = "Owned MatMul";
    const PREPARATION_LABEL: &'static str = "owned MatMul graph preparation";
    const NATIVE_ROUTE: &'static str = "native_rocblas";
    const WARM_SOURCE_REFRESH: bool = true;

    fn fill<const R: usize, const K: usize, const C: usize>(
        lhs: &mut [f32],
        rhs: &mut [f32],
        job: u64,
    ) {
        for (index, value) in lhs.iter_mut().enumerate() {
            let index = u64::try_from(index).expect("matrix extent fits u64");
            let signed = i16::try_from((index * 17 + job * 3) % 23).expect("bounded") - 11;
            *value = f32::from(signed) / 17.0;
        }
        for (index, value) in rhs.iter_mut().enumerate() {
            let index = u64::try_from(index).expect("matrix extent fits u64");
            let signed = i16::try_from((index * 13 + job * 5 + 7) % 29).expect("bounded") - 14;
            *value = f32::from(signed) / 19.0;
        }
        let _ = (R, K, C);
    }

    #[allow(clippy::cast_possible_truncation)] // The reference result is intentionally rounded to f32.
    fn oracle<const R: usize, const K: usize, const C: usize>(
        lhs: &[f32],
        rhs: &[f32],
        output: &mut [f32],
    ) {
        for row in 0..R {
            for column in 0..C {
                let mut sum = 0.0_f64;
                for depth in 0..K {
                    sum = f64::from(lhs[row * K + depth])
                        .mul_add(f64::from(rhs[depth * C + column]), sum);
                }
                output[row * C + column] = sum as f32;
            }
        }
    }

    fn verify<const R: usize, const K: usize, const C: usize>(expected: &[f32], actual: &[f32]) {
        assert_eq!(expected.len(), R * C);
        assert_eq!(actual.len(), R * C);
        let tolerance = 2.0e-6_f32 * f32::from(u16::try_from(K).expect("depth fits u16"));
        assert!(
            expected
                .iter()
                .zip(actual)
                .all(|(&e, &a)| (e - a).abs() <= tolerance * e.abs().max(1.0)),
            "{R}x{K}x{C} f32 MatMul output differs from CPU oracle"
        );
    }

    fn source_identity<const R: usize, const C: usize>(
        input: &[[f32; C]; R],
    ) -> Result<PcuTensor<f32>, Box<dyn Error>> {
        Ok(super::owned_matmul_common::source_identity::<f32, R, C>(
            input,
        )?)
    }

    fn source_matmul<const R: usize, const K: usize, const C: usize>(
        lhs: &PcuTensor<f32>,
        rhs: &PcuTensor<f32>,
    ) -> Result<PcuTensor<f32>, Box<dyn Error>> {
        Ok(super::owned_matmul_common::source_matmul::<f32, R, K, C>(
            lhs, rhs,
        )?)
    }

    fn refresh_source_inputs<
        const R: usize,
        const K: usize,
        const C: usize,
        const NL: usize,
        const NR: usize,
    >(
        host: &super::owned_matmul_common::HostInputs<Self, R, K, C>,
        lhs: &mut PcuTensor<f32>,
        rhs: &mut PcuTensor<f32>,
    ) -> Result<(), Box<dyn Error>> {
        matrix_refresh::<R, K, NL>(&host.source_lhs, lhs)?;
        matrix_refresh::<K, C, NR>(&host.source_rhs, rhs)?;
        Ok(())
    }

    fn warm_source_refresh<
        const R: usize,
        const K: usize,
        const C: usize,
        const NL: usize,
        const NR: usize,
    >(
        host: &super::owned_matmul_common::HostInputs<Self, R, K, C>,
        lhs: &mut PcuTensor<f32>,
        rhs: &mut PcuTensor<f32>,
    ) -> Result<(), Box<dyn Error>> {
        Self::refresh_source_inputs::<R, K, C, NL, NR>(host, lhs, rhs)
    }

    fn require_native_support(_: &Rocblas) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn native_call<const R: usize, const K: usize, const C: usize>(
        runtime: &fusion_pcu_rocm::HipRuntime,
        blas: &Rocblas,
        lhs: &DeviceBuffer,
        rhs: &DeviceBuffer,
        output_bytes: usize,
    ) -> Result<DeviceBuffer, Box<dyn Error>> {
        let output = runtime.allocate(output_bytes)?;
        blas.sgemm(false, false, C, R, K, 1.0, rhs, C, lhs, K, 0.0, &output, C)?;
        Ok(output)
    }
}

pub fn run(criterion: &mut criterion::Criterion) -> Result<(), Box<dyn Error>> {
    let (backend, runtime, pool, name) = select_backend()?;
    println!("Owned source MatMul benchmark device: {name}");
    run_case::<F32Profile, SMALL, 2, SMALL, 8, 8>(criterion, &backend, &runtime, pool)?;
    run_case::<F32Profile, LARGE, LARGE, LARGE, 65_536, 65_536>(criterion, &backend, &runtime, pool)
}
