//! Ergonomic stream-kernel builders layered over `fusion-pcu`.

use core::num::NonZeroU32;

use super::{
    Pcu,
    PcuBackendKind,
    PcuByteStreamBindings,
    PcuCompletedInvocation,
    PcuDispatchPolicy,
    PcuError,
    PcuHalfWordStreamBindings,
    PcuInvocationBindings,
    PcuInvocationDescriptor,
    PcuInvocationHandle,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuKernel,
    PcuKernelId,
    PcuParameter,
    PcuParameterSlot,
    PcuPort,
    PcuStreamCapabilities,
    PcuStreamKernelIr,
    PcuStreamPattern,
    PcuStreamValueType,
    PcuWordStreamBindings,
};

const DEFAULT_PATTERN_CAPACITY: usize = 16;

const BYTE_STREAM_PORTS: [PcuPort<'static>; 2] = [
    PcuPort::stream_input(Some("input"), PcuStreamValueType::U8.as_value_type()),
    PcuPort::stream_output(Some("output"), PcuStreamValueType::U8.as_value_type()),
];

const HALF_WORD_STREAM_PORTS: [PcuPort<'static>; 2] = [
    PcuPort::stream_input(Some("input"), PcuStreamValueType::U16.as_value_type()),
    PcuPort::stream_output(Some("output"), PcuStreamValueType::U16.as_value_type()),
];

const WORD_STREAM_PORTS: [PcuPort<'static>; 2] = [
    PcuPort::stream_input(Some("input"), PcuStreamValueType::U32.as_value_type()),
    PcuPort::stream_output(Some("output"), PcuStreamValueType::U32.as_value_type()),
];

/// Builder for one unary stream transform dispatched through the selected PCU backend.
#[derive(Debug, Clone, Copy)]
pub struct PcuStreamDispatchBuilder<'a, const MAX_PATTERNS: usize = DEFAULT_PATTERN_CAPACITY> {
    system: &'a Pcu,
    kernel_id: PcuKernelId,
    entry_point: &'a str,
    value_type: PcuStreamValueType,
    parameters: &'a [PcuParameter<'a>],
    patterns: [PcuStreamPattern; MAX_PATTERNS],
    pattern_len: usize,
    capabilities: PcuStreamCapabilities,
    threads: NonZeroU32,
    policy: PcuDispatchPolicy,
}

impl<'a, const MAX_PATTERNS: usize> PcuStreamDispatchBuilder<'a, MAX_PATTERNS> {
    pub(crate) fn new(
        system: &'a Pcu,
        kernel_id: PcuKernelId,
        entry_point: &'a str,
        value_type: PcuStreamValueType,
    ) -> Self {
        Self {
            system,
            kernel_id,
            entry_point,
            value_type,
            parameters: &[],
            patterns: [PcuStreamPattern::BitReverse; MAX_PATTERNS],
            pattern_len: 0,
            capabilities: PcuStreamCapabilities::FIFO_INPUT | PcuStreamCapabilities::FIFO_OUTPUT,
            threads: NonZeroU32::new(1).expect("one thread is a valid default PCU dispatch size"),
            policy: PcuDispatchPolicy::PreferHardwareAllowCpuFallback,
        }
    }

    /// Returns the stable kernel id for this dispatch builder.
    #[must_use]
    pub const fn kernel_id(&self) -> PcuKernelId {
        self.kernel_id
    }

    /// Returns the entry-point label used for dispatch.
    #[must_use]
    pub const fn entry_point(&self) -> &str {
        self.entry_point
    }

    /// Returns the bound stream element type.
    #[must_use]
    pub const fn value_type(&self) -> PcuStreamValueType {
        self.value_type
    }

    /// Returns the currently requested logical thread count.
    #[must_use]
    pub const fn thread_count(&self) -> NonZeroU32 {
        self.threads
    }

    /// Returns the currently selected dispatch policy.
    #[must_use]
    pub const fn policy(&self) -> PcuDispatchPolicy {
        self.policy
    }

    /// Returns the inferred capability set required by the configured patterns.
    #[must_use]
    pub const fn capabilities(&self) -> PcuStreamCapabilities {
        self.capabilities
    }

    /// Returns the currently configured pattern count.
    #[must_use]
    pub const fn pattern_count(&self) -> usize {
        self.pattern_len
    }

    /// Returns one borrowed view of the configured semantic patterns.
    #[must_use]
    pub fn patterns(&self) -> &[PcuStreamPattern] {
        &self.patterns[..self.pattern_len]
    }

    /// Returns the declared runtime parameters carried by this stream kernel.
    #[must_use]
    pub const fn parameters(&self) -> &'a [PcuParameter<'a>] {
        self.parameters
    }

    /// Synthesizes the corresponding `fusion-pcu` stream-kernel IR payload.
    #[must_use]
    pub fn ir(&self) -> PcuStreamKernelIr<'_> {
        self.kernel_ir()
    }

    /// Synthesizes the corresponding `fusion-pcu` generic kernel wrapper.
    #[must_use]
    pub fn kernel(&self) -> PcuKernel<'_> {
        PcuKernel::Stream(self.ir())
    }

    /// Builds the corresponding `fusion-pcu` invocation descriptor around one caller-owned kernel.
    #[must_use]
    pub fn descriptor<'kernel>(
        &self,
        kernel: &'kernel PcuKernel<'kernel>,
    ) -> PcuInvocationDescriptor<'kernel> {
        PcuInvocationDescriptor {
            kernel,
            shape: PcuInvocationShape::threads(self.threads),
            policy: self.policy,
        }
    }

    /// Replaces the requested logical thread count with one checked scalar value.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` when `threads == 0`.
    pub fn with_thread_count(mut self, threads: u32) -> Result<Self, PcuError> {
        self.threads = NonZeroU32::new(threads).ok_or_else(PcuError::invalid)?;
        Ok(self)
    }

    /// Replaces the requested logical thread count with one already-validated non-zero value.
    #[must_use]
    pub const fn threads(mut self, threads: NonZeroU32) -> Self {
        self.threads = threads;
        self
    }

    /// Replaces the dispatch policy.
    #[must_use]
    pub const fn with_policy(mut self, policy: PcuDispatchPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Forces CPU fallback execution only.
    #[must_use]
    pub const fn cpu_only(self) -> Self {
        self.with_policy(PcuDispatchPolicy::CpuOnly)
    }

    /// Requires one specific backend.
    #[must_use]
    pub const fn require_backend(self, backend: PcuBackendKind) -> Self {
        self.with_policy(PcuDispatchPolicy::Require(backend))
    }

    /// Prefers one specific backend and falls back to another supported executor when needed.
    #[must_use]
    pub const fn prefer_backend(self, backend: PcuBackendKind) -> Self {
        self.with_policy(PcuDispatchPolicy::Prefer(backend))
    }

    /// Prefers hardware execution and allows CPU fallback.
    #[must_use]
    pub const fn prefer_hardware(self) -> Self {
        self.with_policy(PcuDispatchPolicy::PreferHardwareAllowCpuFallback)
    }

    /// Requires Cortex-M PIO execution.
    #[must_use]
    pub const fn require_pio(self) -> Self {
        self.require_backend(PcuBackendKind::CortexMPio)
    }

    /// Appends one semantic stream pattern.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn with_pattern(mut self, pattern: PcuStreamPattern) -> Result<Self, PcuError> {
        self.push_pattern(pattern)?;
        Ok(self)
    }

    /// Replaces the declared runtime-parameter slice used by this stream kernel.
    #[must_use]
    pub const fn with_parameters(mut self, parameters: &'a [PcuParameter<'a>]) -> Self {
        self.parameters = parameters;
        self
    }

    /// Appends several semantic stream patterns in-order.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn with_patterns(mut self, patterns: &[PcuStreamPattern]) -> Result<Self, PcuError> {
        for pattern in patterns.iter().copied() {
            self.push_pattern(pattern)?;
        }
        Ok(self)
    }

    /// Appends one `BitReverse` transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn bit_reverse(self) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::BitReverse)
    }

    /// Appends one `BitInvert` transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn bit_invert(self) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::BitInvert)
    }

    /// Appends one wrapping increment transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn increment(self) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::Increment)
    }

    /// Appends one runtime-parameterized wrapping add transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn add_parameter(self, parameter: PcuParameterSlot) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::AddParameter { parameter })
    }

    /// Appends one runtime-parameterized xor transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn xor_parameter(self, parameter: PcuParameterSlot) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::XorParameter { parameter })
    }

    /// Appends one specialized left shift.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn shift_left(self, bits: u8) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::ShiftLeft { bits })
    }

    /// Appends one specialized right shift.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn shift_right(self, bits: u8) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::ShiftRight { bits })
    }

    /// Appends one specialized extract-bits transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn extract_bits(self, offset: u8, width: u8) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::ExtractBits { offset, width })
    }

    /// Appends one specialized mask-lower transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn mask_lower(self, bits: u8) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::MaskLower { bits })
    }

    /// Appends one `ByteSwap32` transform.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder pattern budget is exhausted.
    pub fn byte_swap32(self) -> Result<Self, PcuError> {
        self.with_pattern(PcuStreamPattern::ByteSwap32)
    }

    /// Dispatches one byte-stream transform.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, or dispatch failure.
    pub fn dispatch_bytes(
        self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<PcuCompletedInvocation, PcuError> {
        self.dispatch_bytes_with_parameters(input, output, PcuInvocationParameters::empty())
    }

    /// Dispatches one byte-stream transform with explicit runtime parameters.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, or dispatch failure.
    pub fn dispatch_bytes_with_parameters(
        self,
        input: &[u8],
        output: &mut [u8],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<PcuCompletedInvocation, PcuError> {
        if self.value_type != PcuStreamValueType::U8 {
            return Err(PcuError::invalid());
        }
        self.dispatch(
            PcuInvocationBindings::StreamBytes(PcuByteStreamBindings { input, output }),
            parameters,
        )
    }

    /// Dispatches one half-word stream transform.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, or dispatch failure.
    pub fn dispatch_half_words(
        self,
        input: &[u16],
        output: &mut [u16],
    ) -> Result<PcuCompletedInvocation, PcuError> {
        self.dispatch_half_words_with_parameters(input, output, PcuInvocationParameters::empty())
    }

    /// Dispatches one half-word stream transform with explicit runtime parameters.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, or dispatch failure.
    pub fn dispatch_half_words_with_parameters(
        self,
        input: &[u16],
        output: &mut [u16],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<PcuCompletedInvocation, PcuError> {
        if self.value_type != PcuStreamValueType::U16 {
            return Err(PcuError::invalid());
        }
        self.dispatch(
            PcuInvocationBindings::StreamHalfWords(PcuHalfWordStreamBindings { input, output }),
            parameters,
        )
    }

    /// Dispatches one word-stream transform.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, or dispatch failure.
    pub fn dispatch_words(
        self,
        input: &[u32],
        output: &mut [u32],
    ) -> Result<PcuCompletedInvocation, PcuError> {
        self.dispatch_words_with_parameters(input, output, PcuInvocationParameters::empty())
    }

    /// Dispatches one word-stream transform with explicit runtime parameters.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, or dispatch failure.
    pub fn dispatch_words_with_parameters(
        self,
        input: &[u32],
        output: &mut [u32],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<PcuCompletedInvocation, PcuError> {
        if self.value_type != PcuStreamValueType::U32 {
            return Err(PcuError::invalid());
        }
        self.dispatch(
            PcuInvocationBindings::StreamWords(PcuWordStreamBindings { input, output }),
            parameters,
        )
    }

    /// Dispatches one byte-stream transform and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, dispatch, or completion failure.
    pub fn run_bytes(self, input: &[u8], output: &mut [u8]) -> Result<(), PcuError> {
        self.dispatch_bytes(input, output)?.wait()
    }

    /// Dispatches one half-word stream transform and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, dispatch, or completion failure.
    pub fn run_half_words(self, input: &[u16], output: &mut [u16]) -> Result<(), PcuError> {
        self.dispatch_half_words(input, output)?.wait()
    }

    /// Dispatches one word-stream transform and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, dispatch, or completion failure.
    pub fn run_words(self, input: &[u32], output: &mut [u32]) -> Result<(), PcuError> {
        self.dispatch_words(input, output)?.wait()
    }

    /// Dispatches one byte-stream transform over a single scalar value and returns the result.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, dispatch, or completion failure.
    pub fn run_byte(self, value: u8) -> Result<u8, PcuError> {
        let mut output = [0u8; 1];
        self.run_bytes(&[value], &mut output)?;
        Ok(output[0])
    }

    /// Dispatches one half-word stream transform over a single scalar value and returns the
    /// result.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, dispatch, or completion failure.
    pub fn run_half_word(self, value: u16) -> Result<u16, PcuError> {
        let mut output = [0u16; 1];
        self.run_half_words(&[value], &mut output)?;
        Ok(output[0])
    }

    /// Dispatches one word-stream transform over a single scalar value and returns the result.
    ///
    /// # Errors
    ///
    /// Returns any honest planning, preparation, dispatch, or completion failure.
    pub fn run_word(self, value: u32) -> Result<u32, PcuError> {
        let mut output = [0u32; 1];
        self.run_words(&[value], &mut output)?;
        Ok(output[0])
    }

    fn dispatch(
        self,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<PcuCompletedInvocation, PcuError> {
        let ir = self.kernel_ir();
        if !ir.simple_transform_patterns_are_valid() {
            return Err(PcuError::invalid());
        }
        let kernel = PcuKernel::Stream(ir);
        let descriptor = self.descriptor(&kernel);
        let plan = self.system.plan(descriptor)?;
        let prepared = self.system.prepare(plan)?;
        prepared.dispatch_with_parameters(bindings, parameters)
    }

    fn kernel_ir(&self) -> PcuStreamKernelIr<'_> {
        PcuStreamKernelIr {
            id: self.kernel_id,
            entry_point: self.entry_point,
            bindings: &[],
            ports: match self.value_type {
                PcuStreamValueType::U8 => &BYTE_STREAM_PORTS,
                PcuStreamValueType::U16 => &HALF_WORD_STREAM_PORTS,
                PcuStreamValueType::U32 => &WORD_STREAM_PORTS,
            },
            parameters: self.parameters,
            patterns: self.patterns(),
            capabilities: self.capabilities,
        }
    }

    fn push_pattern(&mut self, pattern: PcuStreamPattern) -> Result<(), PcuError> {
        if self.pattern_len >= MAX_PATTERNS {
            return Err(PcuError::resource_exhausted());
        }
        self.patterns[self.pattern_len] = pattern;
        self.pattern_len += 1;
        self.capabilities |= pattern_capabilities(pattern);
        Ok(())
    }
}

const fn pattern_capabilities(pattern: PcuStreamPattern) -> PcuStreamCapabilities {
    match pattern {
        PcuStreamPattern::BitReverse => PcuStreamCapabilities::BIT_REVERSE,
        PcuStreamPattern::BitInvert => PcuStreamCapabilities::BIT_INVERT,
        PcuStreamPattern::Increment => PcuStreamCapabilities::INCREMENT,
        PcuStreamPattern::AddParameter { .. } => PcuStreamCapabilities::ADD_PARAMETER,
        PcuStreamPattern::XorParameter { .. } => PcuStreamCapabilities::XOR_PARAMETER,
        PcuStreamPattern::ShiftLeft { .. } => PcuStreamCapabilities::SHIFT_LEFT,
        PcuStreamPattern::ShiftRight { .. } => PcuStreamCapabilities::SHIFT_RIGHT,
        PcuStreamPattern::ExtractBits { .. } => PcuStreamCapabilities::EXTRACT_BITS,
        PcuStreamPattern::MaskLower { .. } => PcuStreamCapabilities::MASK_LOWER,
        PcuStreamPattern::ByteSwap32 => PcuStreamCapabilities::BYTE_SWAP32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PcuInvocationHandle;

    #[test]
    fn stream_builder_cpu_fallback_executes_bit_reverse_words() {
        let system = Pcu::new();
        let mut output = [0u32; 4];
        let handle = system
            .stream_words(0x101, "bit_reverse")
            .bit_reverse()
            .expect("single-pattern builder should have plenty of capacity")
            .cpu_only()
            .dispatch_words(
                &[0x0000_00f0, 0x1234_5678, 0x8000_0001, 0xffff_0000],
                &mut output,
            )
            .expect("CPU fallback dispatch should succeed");

        handle
            .wait()
            .expect("completed CPU dispatch should wait cleanly");
        assert_eq!(output, [0x0f00_0000, 0x1e6a_2c48, 0x8000_0001, 0x0000_ffff]);
    }

    #[test]
    fn stream_builder_infers_capabilities_from_patterns() {
        let system = Pcu::new();
        let builder = system
            .stream_words(0x102, "extract_bits")
            .extract_bits(4, 12)
            .expect("single-pattern builder should have plenty of capacity")
            .mask_lower(8)
            .expect("builder should accept a second pattern");

        assert_eq!(builder.patterns().len(), 2);
        assert!(
            builder
                .capabilities()
                .contains(PcuStreamCapabilities::FIFO_INPUT)
        );
        assert!(
            builder
                .capabilities()
                .contains(PcuStreamCapabilities::FIFO_OUTPUT)
        );
        assert!(
            builder
                .capabilities()
                .contains(PcuStreamCapabilities::EXTRACT_BITS)
        );
        assert!(
            builder
                .capabilities()
                .contains(PcuStreamCapabilities::MASK_LOWER)
        );
        assert!(
            !builder
                .capabilities()
                .contains(PcuStreamCapabilities::INCREMENT)
        );
    }

    #[test]
    fn stream_builder_cpu_fallback_executes_increment_words() {
        let system = Pcu::new();
        let mut output = [0u32; 3];
        let handle = system
            .stream_words(0x104, "increment")
            .increment()
            .expect("single-pattern builder should have plenty of capacity")
            .cpu_only()
            .dispatch_words(&[0, 41, u32::MAX], &mut output)
            .expect("CPU fallback increment dispatch should succeed");

        handle
            .wait()
            .expect("completed CPU dispatch should wait cleanly");
        assert_eq!(output, [1, 42, 0]);
    }

    #[test]
    fn stream_builder_run_word_executes_increment() {
        let system = Pcu::new();
        let value = system
            .stream_words(0x105, "increment")
            .increment()
            .expect("single-pattern builder should have plenty of capacity")
            .cpu_only()
            .run_word(41)
            .expect("single-value CPU increment should succeed");

        assert_eq!(value, 42);
    }

    #[test]
    fn stream_builder_rejects_zero_threads() {
        let error = Pcu::new()
            .stream_words(0x103, "bit_reverse")
            .with_thread_count(0)
            .expect_err("zero threads should be rejected");

        assert_eq!(error.kind(), crate::PcuErrorKind::Invalid);
    }
}
