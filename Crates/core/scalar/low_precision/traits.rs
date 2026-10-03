//! Generic checked-float bridges; all numerical work remains in the inherent references.
#[rustfmt::skip]
use crate::{
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuCheckedFloat,
    PcuClampedFloat,
    PcuClampedError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
};

macro_rules! checked_binary {
    ($ty:ty, $method:ident) => {
        fn $method(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
            <$ty>::$method(self, rhs)
        }
    };
}
macro_rules! checked_binary_policy {
    ($ty:ty, $method:ident) => {
        fn $method(
            self,
            rhs: Self,
            policy: PcuFloatUnderflowPolicy,
        ) -> Result<Self, PcuExecutionFaultKind> {
            <$ty>::$method(self, rhs, policy)
        }
    };
}
macro_rules! clamped_binary {
    ($ty:ty, $method:ident) => {
        fn $method(self, rhs: Self) -> Result<Self, PcuClampedError<Self>> {
            <$ty>::$method(self, rhs)
        }
    };
}
macro_rules! clamped_binary_policy {
    ($ty:ty, $method:ident) => {
        fn $method(
            self,
            rhs: Self,
            policy: PcuFloatUnderflowPolicy,
        ) -> Result<Self, PcuClampedError<Self>> {
            <$ty>::$method(self, rhs, policy)
        }
    };
}
macro_rules! bridges {
    ($ty:ty) => {
        impl PcuCheckedFloat for $ty {
            checked_binary!($ty, pcu_checked_add);
            checked_binary!($ty, pcu_checked_sub);
            checked_binary!($ty, pcu_checked_mul);
            checked_binary!($ty, pcu_checked_div);
            checked_binary_policy!($ty, pcu_checked_add_with_policy);
            checked_binary_policy!($ty, pcu_checked_sub_with_policy);
            checked_binary_policy!($ty, pcu_checked_mul_with_policy);
            checked_binary_policy!($ty, pcu_checked_div_with_policy);
            fn pcu_checked_neg(self) -> Result<Self, PcuExecutionFaultKind> {
                <$ty>::pcu_checked_neg(self)
            }
            fn pcu_checked_neg_with_policy(
                self,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuExecutionFaultKind> {
                <$ty>::pcu_checked_neg_with_policy(self, policy)
            }
            fn pcu_checked_relu(self) -> Result<Self, PcuExecutionFaultKind> {
                <$ty>::pcu_checked_relu(self)
            }
            fn pcu_checked_relu_with_policy(
                self,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuExecutionFaultKind> {
                <$ty>::pcu_checked_relu_with_policy(self, policy)
            }
            fn pcu_checked_relu_backward_with_policy(
                self,
                upstream: Self,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuExecutionFaultKind> {
                <$ty>::pcu_checked_relu_backward_with_policy(self, upstream, policy)
            }
        }
        impl PcuClampedFloat for $ty {
            clamped_binary!($ty, pcu_clamped_add);
            clamped_binary!($ty, pcu_clamped_sub);
            clamped_binary!($ty, pcu_clamped_mul);
            clamped_binary!($ty, pcu_clamped_div);
            clamped_binary_policy!($ty, pcu_clamped_add_with_policy);
            clamped_binary_policy!($ty, pcu_clamped_sub_with_policy);
            clamped_binary_policy!($ty, pcu_clamped_mul_with_policy);
            clamped_binary_policy!($ty, pcu_clamped_div_with_policy);
            fn pcu_clamped_neg(self) -> Result<Self, PcuClampedError<Self>> {
                <$ty>::pcu_clamped_neg_with_policy(self, PcuFloatUnderflowPolicy::default())
            }
            fn pcu_clamped_neg_with_policy(
                self,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuClampedError<Self>> {
                <$ty>::pcu_clamped_neg_with_policy(self, policy)
            }
            fn pcu_clamped_relu(self) -> Result<Self, PcuClampedError<Self>> {
                <$ty>::pcu_clamped_relu_with_policy(self, PcuFloatUnderflowPolicy::default())
            }
            fn pcu_clamped_relu_with_policy(
                self,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuClampedError<Self>> {
                <$ty>::pcu_clamped_relu_with_policy(self, policy)
            }
        }
    };
}
bridges!(PcuF16Bits);
bridges!(PcuBf16Bits);
bridges!(PcuF8E4M3FnBits);
bridges!(PcuF8E5M2Bits);
