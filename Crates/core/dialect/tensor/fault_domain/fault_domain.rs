//! Logical status domains for the specified checked, ordered tensor primitives.
//!
//! These positions describe arithmetic order, not a provider's packed status ABI,
//! output allocation size or number of launched threads. The contracts apply to
//! checked Strict execution only; native/library Boundary reductions need their
//! own domain and exception contract. All construction is allocation-free.

#[rustfmt::skip]
use crate::{
    PcuCheckedScalarFaultLaw,
    PcuDispatchFloatBinaryOp,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuScalarType,
};
use super::TensorArithmeticStep;

/// Semantic coordinates within an ordered checked tensor primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorStrictFaultLocation {
    pub element_index: u64,
    pub reduction_index: u64,
    pub step: TensorArithmeticStep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Primitive {
    MatMul { cells: u64, inner: u64 },
    Sgd { elements: u64 },
    Mse { elements: u64 },
}

/// Cold-retained coordinates and per-step fault laws for checked Strict F32/F64.
///
/// Each scalar step rounds and checks before the next step. `MatMul` uses ordered
/// multiply/add pairs, SGD multiply/subtract pairs, and MSE subtract/multiply/add
/// triples followed by one mean division. This is PCU sequencing; IEEE 754-2019
/// specifies the constituent scalar rounding/exception rules, not this order.
/// Neither Clamp nor library-defined compound execution is certified here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorStrictFaultDomain {
    primitive: Primitive,
    event_extent: u64,
    laws: [PcuCheckedScalarFaultLaw; 4],
}

impl TensorStrictFaultDomain {
    /// Ordered output-cell/reduction/multiply-add domain. An empty output or
    /// zero-length reduction has no arithmetic events; a zero-length dot product
    /// yields +0 under the ordered reference contract. An overflowing event count
    /// rejects before device preparation. Backend shape support remains separate.
    #[must_use]
    pub const fn matmul(
        scalar: PcuScalarType,
        output_cells: u64,
        inner: u64,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Option<Self> {
        let Some(products) = output_cells.checked_mul(inner) else {
            return None;
        };
        let Some(events) = products.checked_mul(2) else {
            return None;
        };
        Self::new(
            scalar,
            Primitive::MatMul {
                cells: output_cells,
                inner,
            },
            events,
            underflow,
        )
    }

    /// Ordered per-element multiply/subtract domain for an SGD update.
    /// An empty update has zero arithmetic events and cannot report a fault.
    #[must_use]
    pub const fn sgd(
        scalar: PcuScalarType,
        elements: u64,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Option<Self> {
        let Some(events) = elements.checked_mul(2) else {
            return None;
        };
        Self::new(scalar, Primitive::Sgd { elements }, events, underflow)
    }

    /// Ordered reduction triples and final mean. The mean's reduction index is
    /// `elements`; it is not a second output element or an input access.
    #[must_use]
    pub const fn mse(
        scalar: PcuScalarType,
        elements: u64,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Option<Self> {
        if elements == 0 {
            return None;
        }
        let Some(triples) = elements.checked_mul(3) else {
            return None;
        };
        let Some(events) = triples.checked_add(1) else {
            return None;
        };
        Self::new(scalar, Primitive::Mse { elements }, events, underflow)
    }

    /// Number of ordered fault positions, independent of storage/launch size.
    /// Providers separately prove this domain fits their physical status encoding.
    #[must_use]
    pub const fn event_extent(self) -> u64 {
        self.event_extent
    }

    /// Interprets an ordinal in the specified arithmetic order. This does not
    /// decode physical provider bits or certify that a fault actually occurred.
    #[must_use]
    pub const fn location(self, ordinal: u64) -> Option<TensorStrictFaultLocation> {
        if ordinal >= self.event_extent {
            return None;
        }
        let (element_index, reduction_index, step) = match self.primitive {
            Primitive::MatMul { inner, .. } => {
                let pair = ordinal / 2;
                (
                    pair / inner,
                    pair % inner,
                    if ordinal.is_multiple_of(2) {
                        TensorArithmeticStep::Multiply
                    } else {
                        TensorArithmeticStep::Add
                    },
                )
            }
            Primitive::Sgd { .. } => (
                ordinal / 2,
                0,
                if ordinal.is_multiple_of(2) {
                    TensorArithmeticStep::Multiply
                } else {
                    TensorArithmeticStep::Subtract
                },
            ),
            Primitive::Mse { elements } => {
                if ordinal == self.event_extent - 1 {
                    (0, elements, TensorArithmeticStep::Divide)
                } else {
                    let step = match ordinal % 3 {
                        0 => TensorArithmeticStep::Subtract,
                        1 => TensorArithmeticStep::Multiply,
                        _ => TensorArithmeticStep::Add,
                    };
                    (0, ordinal / 3, step)
                }
            }
        };
        Some(TensorStrictFaultLocation {
            element_index,
            reduction_index,
            step,
        })
    }

    /// Converts valid semantic coordinates to their ordered fault position.
    /// All arithmetic is bounded by the construction-time event-count proof.
    #[must_use]
    pub const fn ordinal(self, location: TensorStrictFaultLocation) -> Option<u64> {
        let TensorStrictFaultLocation {
            element_index,
            reduction_index,
            step,
        } = location;
        match self.primitive {
            Primitive::MatMul { cells, inner } => {
                if element_index >= cells || reduction_index >= inner {
                    return None;
                }
                let offset = match step {
                    TensorArithmeticStep::Multiply => 0,
                    TensorArithmeticStep::Add => 1,
                    TensorArithmeticStep::Subtract | TensorArithmeticStep::Divide => return None,
                };
                Some((element_index * inner + reduction_index) * 2 + offset)
            }
            Primitive::Sgd { elements } => {
                if element_index >= elements || reduction_index != 0 {
                    return None;
                }
                let offset = match step {
                    TensorArithmeticStep::Multiply => 0,
                    TensorArithmeticStep::Subtract => 1,
                    TensorArithmeticStep::Add | TensorArithmeticStep::Divide => return None,
                };
                Some(element_index * 2 + offset)
            }
            Primitive::Mse { elements } => {
                if element_index != 0 {
                    return None;
                }
                if matches!(step, TensorArithmeticStep::Divide) {
                    return if reduction_index == elements {
                        Some(self.event_extent - 1)
                    } else {
                        None
                    };
                }
                if reduction_index >= elements {
                    return None;
                }
                let offset = match step {
                    TensorArithmeticStep::Subtract => 0,
                    TensorArithmeticStep::Multiply => 1,
                    TensorArithmeticStep::Add => 2,
                    TensorArithmeticStep::Divide => return None,
                };
                Some(reduction_index * 3 + offset)
            }
        }
    }

    /// Checks a structured candidate's coordinates, fault class and disposition.
    /// Actual operands, physical encoding and terminal publication stay independent.
    #[must_use]
    pub const fn allows(
        self,
        location: TensorStrictFaultLocation,
        kind: PcuExecutionFaultKind,
        recovered: bool,
    ) -> bool {
        if self.ordinal(location).is_none() {
            return false;
        }
        // Successful checked predecessors are finite. These dependent steps do
        // not reload external operands, so invalid-input status is impossible.
        // The first dot-product addition is +0 plus a successfully checked
        // product: exact, finite and already admitted by the tiny-result policy.
        match (self.primitive, location.step) {
            (Primitive::MatMul { .. }, TensorArithmeticStep::Add)
                if location.reduction_index == 0
                    || matches!(kind, PcuExecutionFaultKind::InvalidFloatingOperand) =>
            {
                return false;
            }
            (Primitive::Mse { .. }, TensorArithmeticStep::Multiply)
                if matches!(kind, PcuExecutionFaultKind::InvalidFloatingOperand) =>
            {
                return false;
            }
            // Squares and the +0 accumulator are nonnegative. IEEE Add has no
            // tiny-inexact result; tightened subnormal rejection already checked
            // every square. Only a later accumulation can overflow.
            (Primitive::Mse { .. }, TensorArithmeticStep::Add)
                if location.reduction_index == 0
                    || !matches!(kind, PcuExecutionFaultKind::ArithmeticOverflow) =>
            {
                return false;
            }
            _ => {}
        }
        // A positive count rounds to a finite denominator >= 1. Successful prior
        // checked steps provide a finite total; mean division cannot fault from
        // zero, nonfinite input or overflow. Only tiny-result policy may reject.
        if matches!(location.step, TensorArithmeticStep::Divide)
            && !matches!(kind, PcuExecutionFaultKind::ArithmeticUnderflow)
        {
            return false;
        }
        self.laws[step_slot(location.step)].allows(kind, recovered)
    }

    /// Checks an already-decoded ordinal record using this primitive's domain.
    /// It is not a scalar-map decoder and imposes no universal u32 ordinal limit.
    #[must_use]
    pub const fn accepts(self, fault: PcuExecutionFault) -> bool {
        let Some(location) = self.location(fault.invocation_id) else {
            return false;
        };
        self.allows(location, fault.kind, fault.recovered)
    }

    const fn new(
        scalar: PcuScalarType,
        primitive: Primitive,
        event_extent: u64,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Option<Self> {
        if !matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64) {
            return None;
        }
        let Some(mul) = PcuCheckedScalarFaultLaw::float_binary(
            scalar,
            PcuDispatchFloatBinaryOp::Mul,
            PcuRangePolicy::Reject,
            underflow,
        ) else {
            return None;
        };
        let Some(add) = PcuCheckedScalarFaultLaw::float_binary(
            scalar,
            PcuDispatchFloatBinaryOp::Add,
            PcuRangePolicy::Reject,
            underflow,
        ) else {
            return None;
        };
        let Some(sub) = PcuCheckedScalarFaultLaw::float_binary(
            scalar,
            PcuDispatchFloatBinaryOp::Sub,
            PcuRangePolicy::Reject,
            underflow,
        ) else {
            return None;
        };
        let Some(div) = PcuCheckedScalarFaultLaw::float_binary(
            scalar,
            PcuDispatchFloatBinaryOp::Div,
            PcuRangePolicy::Reject,
            underflow,
        ) else {
            return None;
        };
        Some(Self {
            primitive,
            event_extent,
            laws: [mul, add, sub, div],
        })
    }
}

const fn step_slot(step: TensorArithmeticStep) -> usize {
    match step {
        TensorArithmeticStep::Multiply => 0,
        TensorArithmeticStep::Add => 1,
        TensorArithmeticStep::Subtract => 2,
        TensorArithmeticStep::Divide => 3,
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
