//! Exact dyadic witnesses cover the entire bounded forward/loss/backward/SGD spine.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

macro_rules! sources {
    ($scalar:ty, $loss:ident, $train:ident, $retain:ident, $sum:ident, $consume:ident) => {
        #[pcu(flag(strict))]
        pub fn $loss(
            input: &[[$scalar; 2]; 2],
            weights: &[[$scalar; 1]; 2],
            target: &[[$scalar; 1]; 2],
        ) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
            let prediction = pcu::relu(pcu::matmul(input, weights)?)?;
            pcu::mean_squared_error(&prediction, target)
        }

        #[pcu(flag(strict))]
        pub fn $train(
            input: &[[$scalar; 2]; 2],
            transpose: &[[$scalar; 2]; 2],
            weights: &[[$scalar; 1]; 2],
            target: &[[$scalar; 1]; 2],
        ) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
            let preactivation = pcu::matmul(input, weights)?;
            let prediction = pcu::relu(&preactivation)?;
            // An unused checked numerical result still has observable error semantics.
            let _loss = pcu::mean_squared_error(&prediction, target)?;
            let difference = pcu::sub(&prediction, target)?;
            let derivative = pcu::relu_backward(&preactivation, &difference)?;
            let gradient = pcu::matmul(transpose, &derivative)?;
            pcu::sgd_update(weights, &gradient, 0.5)
        }

        #[pcu]
        pub fn $retain(input: &[$scalar]) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
            pcu::identity(input)
        }

        #[pcu]
        pub fn $sum(
            left: &[$scalar],
            right: &[$scalar],
        ) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
            Ok(left + right)
        }

        #[pcu]
        pub fn $consume(
            input: PcuTensor<$scalar>,
        ) -> Result<PcuTensor<$scalar>, PcuExecutionError> {
            pcu::relu(input)
        }
    };
}

sources!(f32, loss_f32, train_f32, retain_f32, sum_f32, consume_f32);
sources!(f64, loss_f64, train_f64, retain_f64, sum_f64, consume_f64);
