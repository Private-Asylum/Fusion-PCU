//! Safe, session-affine ownership over the bounded Metal implementation.

#[rustfmt::skip]
use std::{
    fmt,
    rc::Rc,
};

#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
};

use crate::ffi;

/// Failure of discovery, admission, transport, compilation or terminal completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalError {
    /// This platform has no admitted Metal runtime.
    Unsupported,
    /// The requested allocation or map length is outside the bounded profile.
    InvalidExtent,
    /// A resource belongs to another independently opened session.
    ForeignSession,
    /// Metal or its compiler reported an operational failure.
    Runtime(String),
    /// Checked arithmetic failed; no useful output is published.
    Arithmetic(MetalFault),
}

impl fmt::Display for MetalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MetalError {}

/// Terminal numerical fault selected by increasing logical invocation index.
pub type MetalFault = PcuExecutionFault;

/// Admitted checked unsigned integer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalIntegerOp {
    /// Exact identity transport with no arithmetic.
    Identity,
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl MetalIntegerOp {
    const fn code(self) -> u32 {
        match self {
            Self::Identity => 8,
            Self::Add => 0,
            Self::Subtract => 1,
            Self::Multiply => 2,
            Self::Divide => 3,
        }
    }
}

/// Physical facts returned by Metal, independent of checked-program capability claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceFacts {
    pub name: String,
    pub registry_id: u64,
    pub unified_memory: bool,
    pub max_buffer_bytes: u64,
}

struct Session {
    native: ffi::Session,
    facts: MetalDeviceFacts,
}

/// One device and command queue. Clones preserve the same resource affinity.
///
/// Sessions deliberately remain thread-local; no unchecked Send/Sync promise is made.
#[derive(Clone)]
pub struct MetalSession(Rc<Session>);

impl MetalSession {
    pub(crate) fn same_session(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    pub(crate) fn ensure_quiescent(&self) -> Result<(), MetalError> {
        self.0.native.ensure_quiescent()
    }
    /// Enumerates physical devices without compiling or executing a kernel.
    ///
    /// # Errors
    /// Returns Unsupported off macOS or the native discovery error.
    pub fn discover() -> Result<Vec<MetalDeviceFacts>, MetalError> {
        ffi::discover()
    }

    /// Opens the device at the current physical enumeration index.
    ///
    /// # Errors
    /// Returns Unsupported off macOS or a missing-device/queue error.
    pub fn open(index: usize) -> Result<Self, MetalError> {
        let (native, facts) = ffi::Session::open(index)?;
        Ok(Self(Rc::new(Session { native, facts })))
    }

    #[must_use]
    pub fn facts(&self) -> &MetalDeviceFacts {
        &self.0.facts
    }

    /// Allocates shared storage and transfers exact u32 words to it.
    ///
    /// # Errors
    /// Returns `InvalidExtent` or a native allocation/transport error.
    pub fn upload_u32(&self, words: &[u32]) -> Result<MetalBuffer, MetalError> {
        self.ensure_quiescent()?;
        let bytes = validate_extent(words.len(), self.0.facts.max_buffer_bytes)?;
        let native = self.0.native.allocate(bytes)?;
        native.write(words)?;
        Ok(MetalBuffer {
            session: self.clone(),
            native,
            words: words.len(),
        })
    }

    /// Allocates owned Shared storage and copies the exact initialized byte slice.
    /// No caller pointer is retained; this is ordinary staging, not a no-copy import.
    ///
    /// # Errors
    /// Returns `InvalidExtent` for empty, non-word, overflowing or oversized extents,
    /// or a native allocation/transport error. Quarantined sessions reject before copying.
    pub fn upload_bytes(&self, bytes: &[u8]) -> Result<MetalBuffer, MetalError> {
        if !bytes.len().is_multiple_of(4) {
            return Err(MetalError::InvalidExtent);
        }
        let buffer = self.allocate_zeroed(bytes.len() / 4)?;
        buffer.native.write_bytes(0, bytes)?;
        Ok(buffer)
    }

    pub(crate) fn allocate_zeroed(&self, words: usize) -> Result<MetalBuffer, MetalError> {
        self.ensure_quiescent()?;
        let bytes = validate_extent(words, self.0.facts.max_buffer_bytes)?;
        // Metal newBufferWithLength:options: clears the allocation to zero. This concrete
        // allocator therefore establishes initialization without a temporary host zero vector.
        let native = self.0.native.allocate(bytes)?;
        Ok(MetalBuffer {
            session: self.clone(),
            native,
            words,
        })
    }

    /// Compiles the fixed integer checker for one admitted operation.
    ///
    /// # Errors
    /// Returns a native compiler/pipeline error.
    pub fn prepare_integer_map(
        &self,
        operation: MetalIntegerOp,
    ) -> Result<MetalPreparedIntegerMap, MetalError> {
        let pipeline = self.0.native.compile(INTEGER_SOURCE, "pcu_checked_u32")?;
        Ok(MetalPreparedIntegerMap {
            session: self.clone(),
            pipeline,
            operation,
        })
    }
}

/// Shared allocation retaining its originating session.
pub struct MetalBuffer {
    session: MetalSession,
    native: ffi::Buffer,
    words: usize,
}

impl MetalBuffer {
    pub(crate) const fn session(&self) -> &MetalSession {
        &self.session
    }
    #[allow(
        clippy::needless_pass_by_ref_mut,
        reason = "Exclusive host ownership is required to modify native shared storage."
    )]
    pub(crate) fn write_bytes(&mut self, offset: usize, bytes: &[u8]) -> Result<(), MetalError> {
        self.session.ensure_quiescent()?;
        self.native.write_bytes(offset, bytes)
    }
    pub(crate) fn read_bytes(&self, offset: usize, bytes: &mut [u8]) -> Result<(), MetalError> {
        self.session.ensure_quiescent()?;
        self.native.read_bytes(offset, bytes)
    }
    #[must_use]
    pub const fn len(&self) -> usize {
        self.words
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.words == 0
    }

    /// Copies the complete owned Shared buffer into an exact initialized host byte slice.
    /// Quiescence and the entire nonempty word extent are checked before any destination write.
    /// This API retains no host pointer and exposes no Shared mapping or borrowed device view.
    ///
    /// # Errors
    /// Returns `InvalidExtent` on a size mismatch or a native/quarantine error. On rejection
    /// no host bytes are modified. Checked callers must complete their numerical gate first.
    pub fn read_into_bytes(&self, bytes: &mut [u8]) -> Result<(), MetalError> {
        self.session.ensure_quiescent()?;
        if self.words.checked_mul(4) != Some(bytes.len()) || bytes.is_empty() {
            return Err(MetalError::InvalidExtent);
        }
        self.native.read_bytes(0, bytes)
    }

    /// Transfers exact words after all synchronous submissions have completed.
    ///
    /// # Errors
    /// Returns a native transport error.
    pub fn download_u32(&self) -> Result<Vec<u32>, MetalError> {
        self.session.ensure_quiescent()?;
        self.native.read(self.words)
    }
}

/// Fixed checked map executable retaining its queue, device and pipeline.
pub struct MetalPreparedIntegerMap {
    session: MetalSession,
    pipeline: ffi::Pipeline,
    operation: MetalIntegerOp,
}

impl MetalPreparedIntegerMap {
    /// Executes with fresh output and private per-element status records.
    ///
    /// The command buffer reaches terminal completion before any record or output is read.
    /// Each lane owns one status record; increasing-index host selection requires no 64-bit
    /// atomic. Numerical failure discards output, and retry uses fresh status storage.
    ///
    /// # Errors
    /// Returns a shape/affinity error before work, a terminal runtime failure, or the first
    /// checked arithmetic fault. A failed call never returns partially useful output.
    pub fn execute(&self, lhs: &MetalBuffer, rhs: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        if lhs.words != rhs.words {
            return Err(MetalError::InvalidExtent);
        }
        self.execute_prefix(lhs, rhs, lhs.words)
    }
    pub(crate) fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
        words: usize,
    ) -> Result<(), MetalError> {
        execute_profile_into(
            &self.session,
            &self.pipeline,
            inputs,
            output,
            self.operation.code(),
            words,
        )
    }
    pub(crate) fn execute_prefix(
        &self,
        lhs: &MetalBuffer,
        rhs: &MetalBuffer,
        words: usize,
    ) -> Result<MetalBuffer, MetalError> {
        execute_profile(
            &self.session,
            &self.pipeline,
            lhs,
            rhs,
            self.operation.code(),
            words,
        )
    }
}

/// Encoding-only checked F32 unary executable; no native floating arithmetic is executed.
pub struct MetalPreparedF32Unary {
    session: MetalSession,
    pipeline: ffi::Pipeline,
    operation: u32,
}
impl MetalPreparedF32Unary {
    /// Runs exact bit selection/sign inversion with nonfinite and underflow diagnostics.
    ///
    /// # Errors
    /// Returns affinity, extent, operational or checked numerical failure.
    pub fn execute(&self, input: &MetalBuffer) -> Result<MetalBuffer, MetalError> {
        self.execute_prefix(input, input.words)
    }
    pub(crate) fn execute_into(
        &self,
        input: &MetalBuffer,
        output: &MetalBuffer,
        words: usize,
    ) -> Result<(), MetalError> {
        execute_profile_into(
            &self.session,
            &self.pipeline,
            [input; 2],
            output,
            self.operation,
            words,
        )
    }
    pub(crate) fn execute_prefix(
        &self,
        input: &MetalBuffer,
        words: usize,
    ) -> Result<MetalBuffer, MetalError> {
        execute_profile(
            &self.session,
            &self.pipeline,
            input,
            input,
            self.operation,
            words,
        )
    }
}
impl MetalSession {
    /// Prepares exact F32 sign inversion using integer encodings and neutral underflow policy.
    ///
    /// # Errors
    /// Returns a native compiler/pipeline failure. Nonfinite encodings fail during execution.
    pub fn prepare_f32_neg(
        &self,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedF32Unary, MetalError> {
        Ok(MetalPreparedF32Unary {
            session: self.clone(),
            pipeline: self.0.native.compile(INTEGER_SOURCE, "pcu_checked_u32")?,
            operation: if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                7
            } else {
                6
            },
        })
    }
    /// Prepares checked F32 `ReLU` with the neutral underflow policy.
    ///
    /// # Errors
    /// Returns a native compiler/pipeline failure. Input/output use exact u32 float encodings.
    pub fn prepare_f32_relu(
        &self,
        underflow: PcuFloatUnderflowPolicy,
    ) -> Result<MetalPreparedF32Unary, MetalError> {
        Ok(MetalPreparedF32Unary {
            session: self.clone(),
            pipeline: self.0.native.compile(INTEGER_SOURCE, "pcu_checked_u32")?,
            operation: if underflow == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                5
            } else {
                4
            },
        })
    }
}
fn execute_profile(
    session: &MetalSession,
    pipeline: &ffi::Pipeline,
    lhs: &MetalBuffer,
    rhs: &MetalBuffer,
    operation: u32,
    words: usize,
) -> Result<MetalBuffer, MetalError> {
    session.ensure_quiescent()?;
    if !Rc::ptr_eq(&session.0, &lhs.session.0) || !Rc::ptr_eq(&session.0, &rhs.session.0) {
        return Err(MetalError::ForeignSession);
    }
    if lhs.words < words || rhs.words < words {
        return Err(MetalError::InvalidExtent);
    }
    let bytes = validate_extent(words, session.0.facts.max_buffer_bytes)?;
    let output = session.0.native.allocate(bytes)?;
    let output = MetalBuffer {
        session: session.clone(),
        native: output,
        words,
    };
    execute_profile_into(session, pipeline, [lhs, rhs], &output, operation, words)?;
    Ok(output)
}

// The safe callers hold exclusive output ownership or allocate private scratch storage.
fn execute_profile_into(
    session: &MetalSession,
    pipeline: &ffi::Pipeline,
    inputs: [&MetalBuffer; 2],
    output: &MetalBuffer,
    operation: u32,
    words: usize,
) -> Result<(), MetalError> {
    session.ensure_quiescent()?;
    for buffer in [inputs[0], inputs[1], output] {
        if !session.same_session(buffer.session()) {
            return Err(MetalError::ForeignSession);
        }
        if buffer.words < words {
            return Err(MetalError::InvalidExtent);
        }
    }
    let bytes = validate_extent(words, session.0.facts.max_buffer_bytes)?;
    let records = session.0.native.allocate(bytes)?;
    records.fill_ones();
    let count = u32::try_from(words).map_err(|_| MetalError::InvalidExtent)?;
    session.0.native.execute(
        pipeline,
        [
            &inputs[0].native,
            &inputs[1].native,
            &output.native,
            &records,
        ],
        [count, operation],
        words,
    )?;
    records.inspect_words(words, select_fault)
}

fn validate_extent(words: usize, maximum: u64) -> Result<usize, MetalError> {
    let bytes = words.checked_mul(4).ok_or(MetalError::InvalidExtent)?;
    if words == 0
        || u32::try_from(words).is_err()
        || u64::try_from(bytes).map_or(true, |bytes| bytes > maximum)
    {
        return Err(MetalError::InvalidExtent);
    }
    Ok(bytes)
}

fn select_fault(records: &[u32]) -> Result<(), MetalError> {
    if records.iter().any(|&record| record > 4) {
        return Err(MetalError::Runtime("invalid checked status record".into()));
    }
    for (element, &record) in records.iter().enumerate() {
        match record {
            0 => {}
            1..=4 => {
                return Err(MetalError::Arithmetic(MetalFault {
                    invocation_id: u64::try_from(element).map_err(|_| MetalError::InvalidExtent)?,
                    kind: match record {
                        1 => PcuExecutionFaultKind::ArithmeticOverflow,
                        2 => PcuExecutionFaultKind::DivideByZero,
                        3 => PcuExecutionFaultKind::ArithmeticUnderflow,
                        _ => PcuExecutionFaultKind::InvalidFloatingOperand,
                    },
                    recovered: false,
                }));
            }
            _ => return Err(MetalError::Runtime("invalid checked status record".into())),
        }
    }
    Ok(())
}

// Every path assigns its private record. Guard before division avoids undefined arithmetic;
// unsigned subtraction/wrap is used only after the range proof succeeds. No floating ALU.
const INTEGER_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void pcu_checked_u32(device const uint* a [[buffer(0)]],
                            device const uint* b [[buffer(1)]],
                            device uint* out [[buffer(2)]],
                            device uint* fault [[buffer(3)]],
                            constant uint2& config [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= config.x) return;
    uint x = a[i], y = b[i], value = 0, code = 0;
    switch (config.y) {
        case 0: if (x > 0xffffffffu - y) code = 1; else value = x + y; break;
        case 1: if (x < y) code = 3; else value = x - y; break;
        case 2: if (y != 0 && x > 0xffffffffu / y) code = 1; else value = x * y; break;
        case 3: if (y == 0) code = 2; else value = x / y; break;
        case 4: case 5:
            if ((x & 0x7f800000u) == 0x7f800000u) code = 4;
            else {
                value = (x & 0x80000000u) != 0 ? 0 : x;
                if (config.y == 5 && value != 0 && (value & 0x7f800000u) == 0) code = 3;
            }
            break;
        case 6: case 7:
            if ((x & 0x7f800000u) == 0x7f800000u) code = 4;
            else {
                value = x ^ 0x80000000u;
                if (config.y == 7 && (value & 0x7fffffffu) != 0 && (value & 0x7f800000u) == 0) code = 3;
            }
            break;
        case 8: value = x; break;
        default: code = 99; break;
    }
    out[i] = value;
    fault[i] = code;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extent_checks_precede_allocation() {
        assert_eq!(validate_extent(0, u64::MAX), Err(MetalError::InvalidExtent));
        assert_eq!(validate_extent(2, 7), Err(MetalError::InvalidExtent));
        assert_eq!(validate_extent(2, 8), Ok(8));
        assert_eq!(
            validate_extent(usize::MAX, u64::MAX),
            Err(MetalError::InvalidExtent)
        );
    }

    #[test]
    fn records_select_first_logical_fault_and_preserve_kind() {
        assert_eq!(
            select_fault(&[0, 2, 1]),
            Err(MetalError::Arithmetic(MetalFault {
                invocation_id: 1,
                kind: PcuExecutionFaultKind::DivideByZero,
                recovered: false,
            }))
        );
        assert_eq!(select_fault(&[0, 0]), Ok(()));
        assert!(matches!(
            select_fault(&[u32::MAX]),
            Err(MetalError::Runtime(_))
        ));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn unsupported_platform_never_opens_a_fallback() {
        assert_eq!(MetalSession::discover(), Err(MetalError::Unsupported));
        assert!(matches!(
            MetalSession::open(0),
            Err(MetalError::Unsupported)
        ));
    }
}

#[cfg(all(test, target_os = "macos"))]
mod hardware_tests {
    use super::*;

    #[test]
    #[ignore = "Requires an admitted macOS Metal device; runs actual compiler and GPU work."]
    fn checked_u32_maps_terminal_fault_retry_and_escaped_ownership() {
        let session = MetalSession::open(0).unwrap();
        assert!(!session.facts().name.is_empty());
        // Ordinary host byte staging may start at an unaligned address. Only the owned Shared
        // allocation requires word alignment; logical borrowed extents are never rounded up.
        let host = [91_u8, 1, 2, 3, 4, 92];
        let staged = session.upload_bytes(&host[1..5]).unwrap();
        let mut copied = [93_u8; 6];
        staged.read_into_bytes(&mut copied[1..5]).unwrap();
        assert_eq!(copied, [93, 1, 2, 3, 4, 93]);
        assert_eq!(
            staged.read_into_bytes(&mut copied),
            Err(MetalError::InvalidExtent)
        );
        assert_eq!(copied, [93, 1, 2, 3, 4, 93]);
        assert!(matches!(
            session.upload_bytes(&host[..5]),
            Err(MetalError::InvalidExtent)
        ));
        assert!(matches!(
            session.upload_bytes(&[]),
            Err(MetalError::InvalidExtent)
        ));
        let zeroed = session.allocate_zeroed(3).unwrap();
        let mut initialized = [94_u8; 12];
        zeroed.read_into_bytes(&mut initialized).unwrap();
        assert_eq!(initialized, [0; 12]);
        for (operation, lhs, rhs, expected) in [
            (
                MetalIntegerOp::Add,
                vec![0, 17, u32::MAX - 1],
                vec![0, 8, 1],
                vec![0, 25, u32::MAX],
            ),
            (
                MetalIntegerOp::Subtract,
                vec![0, 17, u32::MAX],
                vec![0, 8, 1],
                vec![0, 9, u32::MAX - 1],
            ),
            (
                MetalIntegerOp::Multiply,
                vec![0, 17, u32::MAX],
                vec![u32::MAX, 8, 1],
                vec![0, 136, u32::MAX],
            ),
            (
                MetalIntegerOp::Divide,
                vec![0, 17, u32::MAX],
                vec![1, 8, 1],
                vec![0, 2, u32::MAX],
            ),
        ] {
            let prepared = session.prepare_integer_map(operation).unwrap();
            let output = prepared
                .execute(
                    &session.upload_u32(&lhs).unwrap(),
                    &session.upload_u32(&rhs).unwrap(),
                )
                .unwrap();
            assert_eq!(output.download_u32().unwrap(), expected);
        }
        // Partial threadgroups and full-status transfers retain logical ordering.
        let prepared = session.prepare_integer_map(MetalIntegerOp::Add).unwrap();
        let mut left = vec![1; 257];
        left[3] = u32::MAX;
        left[256] = u32::MAX;
        let right = session.upload_u32(&vec![1; 257]).unwrap();
        assert!(matches!(
            prepared.execute(&session.upload_u32(&left).unwrap(), &right),
            Err(MetalError::Arithmetic(MetalFault {
                invocation_id: 3,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                recovered: false
            }))
        ));
        let good = session.upload_u32(&vec![2; 257]).unwrap();
        let output = prepared.execute(&good, &right).unwrap();
        drop(prepared);
        drop(right);
        drop(good);
        drop(session);
        assert_eq!(output.download_u32().unwrap(), vec![3; 257]);
    }

    #[test]
    #[ignore = "Requires an admitted macOS Metal device; runs actual compiler and GPU work."]
    fn checked_u32_fault_kinds_and_session_affinity() {
        let session = MetalSession::open(0).unwrap();
        for (operation, lhs, rhs, zero) in [
            (
                MetalIntegerOp::Add,
                u32::MAX,
                1,
                PcuExecutionFaultKind::ArithmeticOverflow,
            ),
            (
                MetalIntegerOp::Subtract,
                0,
                1,
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ),
            (
                MetalIntegerOp::Multiply,
                u32::MAX,
                2,
                PcuExecutionFaultKind::ArithmeticOverflow,
            ),
            (
                MetalIntegerOp::Divide,
                1,
                0,
                PcuExecutionFaultKind::DivideByZero,
            ),
        ] {
            let prepared = session.prepare_integer_map(operation).unwrap();
            assert!(
                matches!(prepared.execute(&session.upload_u32(&[lhs]).unwrap(), &session.upload_u32(&[rhs]).unwrap()),
                Err(MetalError::Arithmetic(MetalFault { invocation_id: 0, kind, recovered: false })) if kind == zero)
            );
        }
        let other = MetalSession::open(0).unwrap();
        let prepared = session.prepare_integer_map(MetalIntegerOp::Add).unwrap();
        assert!(matches!(
            prepared.execute(
                &session.upload_u32(&[1]).unwrap(),
                &other.upload_u32(&[1]).unwrap()
            ),
            Err(MetalError::ForeignSession)
        ));
    }
    #[test]
    #[ignore = "Requires actual macOS Metal device."]
    fn bit_only_f32_relu_nonfinite_underflow_signed_zero_and_retry() {
        let session = MetalSession::open(0).unwrap();
        let encodings = [
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
        ];
        let expected = [
            0,
            0,
            1,
            0,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0,
            0x7f7f_ffff,
        ];
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let prepared = session.prepare_f32_relu(policy).unwrap();
            assert_eq!(
                prepared
                    .execute(&session.upload_u32(&encodings).unwrap())
                    .unwrap()
                    .download_u32()
                    .unwrap(),
                expected
            );
            for special in [0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0x7f80_0001] {
                assert!(matches!(
                    prepared.execute(&session.upload_u32(&[special]).unwrap()),
                    Err(MetalError::Arithmetic(MetalFault {
                        kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                        invocation_id: 0,
                        recovered: false
                    }))
                ));
            }
            assert_eq!(
                prepared
                    .execute(&session.upload_u32(&[0x3f80_0000]).unwrap())
                    .unwrap()
                    .download_u32()
                    .unwrap(),
                [0x3f80_0000]
            );
        }
        let tight = session
            .prepare_f32_relu(PcuFloatUnderflowPolicy::RejectSubnormalResult)
            .unwrap();
        assert!(matches!(
            tight.execute(&session.upload_u32(&[0, 1, 0x7fc0_0001]).unwrap()),
            Err(MetalError::Arithmetic(MetalFault {
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                invocation_id: 1,
                recovered: false
            }))
        ));
        assert_eq!(
            tight
                .execute(&session.upload_u32(&[0x8000_0001]).unwrap())
                .unwrap()
                .download_u32()
                .unwrap(),
            [0]
        );
    }
    #[test]
    #[ignore = "Requires actual macOS Metal device."]
    fn bit_only_f32_neg_preserves_signed_zero_and_exact_subnormal_policy() {
        let session = MetalSession::open(0).unwrap();
        let words = [
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
        ];
        let expected: Vec<u32> = words.iter().map(|word| word ^ 0x8000_0000).collect();
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let prepared = session.prepare_f32_neg(policy).unwrap();
            assert_eq!(
                prepared
                    .execute(&session.upload_u32(&words).unwrap())
                    .unwrap()
                    .download_u32()
                    .unwrap(),
                expected
            );
            for special in [0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0x7f80_0001] {
                assert!(matches!(
                    prepared.execute(&session.upload_u32(&[special]).unwrap()),
                    Err(MetalError::Arithmetic(MetalFault {
                        kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                        invocation_id: 0,
                        recovered: false
                    }))
                ));
            }
        }
        let tight = session
            .prepare_f32_neg(PcuFloatUnderflowPolicy::RejectSubnormalResult)
            .unwrap();
        assert!(matches!(
            tight.execute(&session.upload_u32(&[0x8000_0000, 1, 0x7fc0_0001]).unwrap()),
            Err(MetalError::Arithmetic(MetalFault {
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                invocation_id: 1,
                recovered: false
            }))
        ));
        assert_eq!(
            tight
                .execute(&session.upload_u32(&[0, 0x8000_0000, 0x0080_0000]).unwrap())
                .unwrap()
                .download_u32()
                .unwrap(),
            [0x8000_0000, 0, 0x8080_0000]
        );
    }
}
