//! Ordered `MatMul` synthesis with checked binary32/binary64 arithmetic.
//!
//! Each row-major output cell multiplies then adds in increasing K order. IEEE-derived
//! rounding and tininess rules come from the existing integer-only scalar helpers; this
//! reduction topology and exception granularity are PCU contracts, not IEEE requirements.
//! No host synchronization occurs inside the reduction. The existing owned Dispatch spine
//! retains arguments through terminal completion before publishing any output or fault.

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy,
    PcuImplementationRequirements,
    PcuRangePolicy,
    PcuInvocationShape,
    PcuNumericalMode,
    PcuOwnedBindingRequirement,
    PcuScalarType,
    PcuValueType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    TensorArithmeticStep,
    TensorError,
};
use core::num::NonZeroU32;
use std::fmt::Write as _;

/// Validated private profile; its fields cannot be forged outside this factory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::redundant_pub_crate)] // Preserve the private typed-factory safety boundary.
pub(crate) struct StrictMatMulSpec {
    rows: u32,
    inner: u32,
    columns: u32,
    transpose_left: bool,
    transpose_right: bool,
    scalar_type: PcuScalarType,
    underflow: PcuFloatUnderflowPolicy,
    numerical_requirements: PcuImplementationRequirements,
}

impl StrictMatMulSpec {
    // Private factory admission establishes positive dimensions and packed event bounds.
    pub(crate) fn fault_domain(self) -> fusion_pcu::dialect::tensor::TensorStrictFaultDomain {
        fusion_pcu::dialect::tensor::TensorStrictFaultDomain::matmul(
            self.scalar_type,
            u64::from(self.rows) * u64::from(self.columns),
            u64::from(self.inner),
            self.underflow,
        )
        .expect("verified checked compound dimensions and format")
    }
    pub(crate) fn fault_extent(self) -> u64 {
        self.fault_domain().event_extent()
    }

    pub(super) fn from_node(graph: &Graph, node: NodeDescriptor<'_>) -> Option<Self> {
        let OpDescriptor::MatMul {
            left,
            right,
            transpose_left,
            transpose_right,
        } = node.op
        else {
            return None;
        };
        if node.numerical_mode != Some(PcuNumericalMode::Strict)
            || node.numerical_options.reproducibility != fusion_pcu::PcuReproducibility::Unspecified
        {
            return None;
        }
        let actual = graph.node(node.value).ok()?;
        if actual.op != node.op
            || actual.scalar_type != node.scalar_type
            || actual.shape != node.shape
            || actual.numerical_mode != node.numerical_mode
            || actual.numerical_options != node.numerical_options
            || actual.float_underflow_policy != node.float_underflow_policy
            || graph.node(left).ok()?.scalar_type != node.scalar_type
            || graph.node(right).ok()?.scalar_type != node.scalar_type
        {
            return None;
        }
        let [left_rows, left_columns] = *graph.shape(left).ok()? else {
            return None;
        };
        let [right_rows, right_columns] = *graph.shape(right).ok()? else {
            return None;
        };
        let (rows, inner) = if transpose_left {
            (left_columns, left_rows)
        } else {
            (left_rows, left_columns)
        };
        let (right_inner, columns) = if transpose_right {
            (right_columns, right_rows)
        } else {
            (right_rows, right_columns)
        };
        if inner != right_inner || node.shape != [rows, columns] {
            return None;
        }
        let spec = Self {
            rows: u32::try_from(rows).ok()?,
            inner: u32::try_from(inner).ok()?,
            columns: u32::try_from(columns).ok()?,
            transpose_left,
            transpose_right,
            scalar_type: node.scalar_type,
            // Permissions may use this stronger ordered checker; retain the requested tuple.
            numerical_requirements: PcuImplementationRequirements {
                numerical_mode: PcuNumericalMode::Strict,
                numerical_options: node.numerical_options,
                float_underflow: node.float_underflow_policy.unwrap_or_default(),
                // Tensor nodes currently represent only Reject range.
                range_policy: PcuRangePolicy::Reject,
            },
            underflow: node.float_underflow_policy?,
        };
        if !matches!(spec.scalar_type, PcuScalarType::F32 | PcuScalarType::F64)
            || spec.rows == 0
            || spec.inner == 0
            || spec.columns == 0
        {
            return None;
        }
        let cells = spec.rows.checked_mul(spec.columns)?;
        // Reserve the normal Dispatch recovered flag even though this profile is reject-only.
        // Lexicographic [cell, K, multiply/add] plus three fault-kind bits must fit below it.
        let steps = u64::from(cells)
            .checked_mul(u64::from(spec.inner))?
            .checked_mul(2)?;
        if steps > (u64::MAX >> 4) {
            return None;
        }
        for count in [
            u64::from(spec.rows) * u64::from(spec.inner),
            u64::from(spec.inner) * u64::from(spec.columns),
            u64::from(cells),
        ] {
            let bytes = count.checked_mul(spec.scalar_bytes())?;
            usize::try_from(bytes).ok()?;
        }
        Some(spec)
    }

    const fn scalar_bytes(self) -> u64 {
        match self.scalar_type {
            PcuScalarType::F32 => 4,
            PcuScalarType::F64 => 8,
            _ => panic!("private validated scalar profile"),
        }
    }

    pub(crate) const fn shape(self) -> PcuInvocationShape {
        // Construction rejects zero and overflowing output-cell counts.
        PcuInvocationShape::invocations(
            NonZeroU32::new(self.rows * self.columns).expect("validated cells"),
        )
    }

    pub(crate) fn requirements(self) -> [PcuOwnedBindingRequirement; 3] {
        let counts = [
            u64::from(self.rows) * u64::from(self.inner),
            u64::from(self.inner) * u64::from(self.columns),
            u64::from(self.rows) * u64::from(self.columns),
        ];
        [0_u32, 1, 2].map(|binding| PcuOwnedBindingRequirement {
            target: PcuBindingRef::new(0, binding),
            access: if binding == 2 {
                PcuBindingAccess::WriteOnly
            } else {
                PcuBindingAccess::ReadOnly
            },
            binding_type: PcuBindingType::Value(PcuValueType::Scalar(self.scalar_type)),
            min_required_bytes: counts[binding as usize] * self.scalar_bytes(),
        })
    }

    pub(crate) fn source(self) -> String {
        let (width, bits_type) = match self.scalar_type {
            PcuScalarType::F32 => ("F32", "unsigned int"),
            PcuScalarType::F64 => ("F64", "unsigned long long"),
            _ => unreachable!("private validated scalar profile"),
        };
        let prefix = width.to_ascii_lowercase();
        let policy = match self.underflow {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        let left_index = if self.transpose_left {
            format!("k * {}ull + row", self.rows)
        } else {
            format!("row * {}ull + k", self.inner)
        };
        let right_index = if self.transpose_right {
            format!("column * {}ull + k", self.inner)
        } else {
            format!("k * {}ull + column", self.columns)
        };
        let mut source = String::new();
        crate::codegen::lower::emit_compound_float_helpers(&mut source, self.scalar_type);
        write!(source, r#"
extern "C" __global__ void fusion_kernel(
    const {bits_type}* left, const {bits_type}* right,
    {bits_type}* output, unsigned long long* fault) {{
    const unsigned long long cell = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= {cells}ull) return;
    const unsigned long long row = cell / {columns}ull;
    const unsigned long long column = cell % {columns}ull;
    {bits_type} accumulator = 0;
    for (unsigned long long k = 0; k < {inner}ull; ++k) {{
        const unsigned long long position = (cell * {inner}ull + k) * 2ull;
        const Fusion{width}CheckedResult product = fusion_checked_{prefix}_binary(left[{left_index}], right[{right_index}], 2u, {policy}u);
        if (product.fault != 0u) {{
            atomicMin(fault, (position << 3) | product.fault);
            return;
        }}
        const Fusion{width}CheckedResult sum = fusion_checked_{prefix}_binary(accumulator, product.bits, 0u, {policy}u);
        if (sum.fault != 0u) {{
            atomicMin(fault, ((position + 1ull) << 3) | sum.fault);
            return;
        }}
        accumulator = sum.bits;
    }}
    output[cell] = accumulator;
}}
"#, cells = self.rows * self.columns, columns = self.columns, inner = self.inner)
            .expect("writing to a String is infallible");
        source
    }

    pub(super) fn fault_error(
        self,
        value: fusion_pcu::dialect::tensor::ValueId,
        fault: PcuExecutionFault,
    ) -> Result<TensorError, super::RocmTensorExecutionError> {
        let operations_per_cell = u64::from(self.inner) * 2;
        if fault.recovered
            || !matches!(
                fault.kind,
                fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow
                    | fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow
                    | fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand
            )
            || fault.invocation_id
                >= u64::from(self.rows) * u64::from(self.columns) * operations_per_cell
        {
            return Err(super::RocmTensorExecutionError::InvalidPlan(value));
        }
        Ok(TensorError::CompoundArithmeticFault {
            value,
            element_index: usize::try_from(fault.invocation_id / operations_per_cell)
                .expect("validated cells"),
            reduction_index: usize::try_from((fault.invocation_id % operations_per_cell) / 2)
                .expect("validated reduction extent"),
            step: if fault.invocation_id.is_multiple_of(2) {
                TensorArithmeticStep::Multiply
            } else {
                TensorArithmeticStep::Add
            },
            kind: fault.kind,
        })
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "hardware.rs"]
mod hardware;
