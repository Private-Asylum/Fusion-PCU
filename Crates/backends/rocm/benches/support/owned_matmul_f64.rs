//! f64 profile for the shared fresh-output `MatMul` driver.

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

use super::owned_matmul_common::{MatMulProfile, run_case, select_backend};

const SMALL: usize = 4;
const LARGE: usize = 256;

pub struct F64Profile;

impl MatMulProfile for F64Profile {
    type Scalar = f64;

    const GROUP: &'static str = "owned_matmul_f64_fresh_output";
    const CENSUS_LABEL: &'static str = "Owned f64 MatMul heap census";
    const DIAGNOSTIC_LABEL: &'static str = "Owned f64 MatMul";
    const PREPARATION_LABEL: &'static str = "owned f64 MatMul graph preparation";
    const NATIVE_ROUTE: &'static str = "native_rocblas_dgemm";
    const WARM_SOURCE_REFRESH: bool = false;

    fn fill<const R: usize, const K: usize, const C: usize>(
        lhs: &mut [f64],
        rhs: &mut [f64],
        job: u64,
    ) {
        for (index, value) in lhs.iter_mut().enumerate() {
            let index = u64::try_from(index).expect("matrix extent fits u64");
            let signed = i16::try_from((index * 17 + job * 3) % 23).expect("bounded") - 11;
            *value = f64::from(signed) / 17.0;
        }
        for (index, value) in rhs.iter_mut().enumerate() {
            let index = u64::try_from(index).expect("matrix extent fits u64");
            let signed = i16::try_from((index * 13 + job * 5 + 7) % 29).expect("bounded") - 14;
            *value = f64::from(signed) / 19.0;
        }
        if R > 0 && K > 0 && C > 0 {
            lhs[..K].fill(0.0);
            rhs[..C].fill(0.0);
            lhs[0] = 16_777_217.0;
            rhs[0] = 1.0;
        }
    }

    fn oracle<const R: usize, const K: usize, const C: usize>(
        lhs: &[f64],
        rhs: &[f64],
        output: &mut [f64],
    ) {
        for row in 0..R {
            for column in 0..C {
                let mut sum = 0.0_f64;
                for depth in 0..K {
                    sum = lhs[row * K + depth].mul_add(rhs[depth * C + column], sum);
                }
                output[row * C + column] = sum;
            }
        }
        if R > 0 && C > 0 {
            assert_eq!(
                output[0].to_bits(),
                16_777_217.0_f64.to_bits(),
                "f64 precision witness"
            );
        }
    }

    fn verify<const R: usize, const K: usize, const C: usize>(expected: &[f64], actual: &[f64]) {
        assert_eq!(expected.len(), R * C);
        assert_eq!(actual.len(), R * C);
        let tolerance =
            2.0e-12_f64 * f64::from(u16::try_from(K).expect("benchmark depth fits u16"));
        assert!(
            expected
                .iter()
                .zip(actual)
                .all(|(&e, &a)| (e - a).abs() <= tolerance * e.abs().max(1.0)),
            "{R}x{K}x{C} f64 MatMul output differs from CPU oracle"
        );
    }

    fn source_identity<const R: usize, const C: usize>(
        input: &[[f64; C]; R],
    ) -> Result<PcuTensor<f64>, Box<dyn Error>> {
        Ok(super::owned_matmul_common::source_identity::<f64, R, C>(
            input,
        )?)
    }

    fn source_matmul<const R: usize, const K: usize, const C: usize>(
        lhs: &PcuTensor<f64>,
        rhs: &PcuTensor<f64>,
    ) -> Result<PcuTensor<f64>, Box<dyn Error>> {
        Ok(super::owned_matmul_common::source_matmul::<f64, R, K, C>(
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
        lhs: &mut PcuTensor<f64>,
        rhs: &mut PcuTensor<f64>,
    ) -> Result<(), Box<dyn Error>> {
        *lhs = super::owned_matmul_common::source_identity::<f64, R, K>(&host.source_lhs)?;
        *rhs = super::owned_matmul_common::source_identity::<f64, K, C>(&host.source_rhs)?;
        let _ = (NL, NR);
        Ok(())
    }

    fn require_native_support(blas: &Rocblas) -> Result<(), Box<dyn Error>> {
        blas.require_dgemm_support()?;
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
        // Row-major lhs*rhs is column-major rhs^T*lhs^T in the same storage.
        blas.dgemm(false, false, C, R, K, 1.0, rhs, C, lhs, K, 0.0, &output, C)?;
        Ok(output)
    }
}

pub fn run(criterion: &mut criterion::Criterion) -> Result<(), Box<dyn Error>> {
    let (backend, runtime, pool, name) = select_backend()?;
    println!("Owned source f64 MatMul benchmark device: {name}");
    run_case::<F64Profile, SMALL, 2, SMALL, 0, 0>(criterion, &backend, &runtime, pool)?;
    run_case::<F64Profile, LARGE, LARGE, LARGE, 0, 0>(criterion, &backend, &runtime, pool)
}
