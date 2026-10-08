//! Private effect-gated executable ABI; ordinary checked batching stays rejected.
#[rustfmt::skip]
use super::{
    CudaOwnedDispatchBackend,
    CudaOwnedDispatchError,
    CudaPreparedDispatch,
    DeviceBuffer,
    CudaError,
    CudaKernelArgument,
    CudaCompletionBatch,
    PcuOwnedBinding,
    find_binding,
};

/// Wrap only the independently validated generated entry, preserving its body verbatim.
fn guarded_source(mut source: String) -> Result<String, CudaOwnedDispatchError> {
    let start = source
        .find("void fusion_kernel(")
        .ok_or(CudaOwnedDispatchError::UnsupportedRequirements)?;
    let end = source[start..]
        .find(") {")
        .map(|offset| start + offset)
        .ok_or(CudaOwnedDispatchError::UnsupportedRequirements)?;
    let parameters = source[start..end].replace(
        "unsigned long long* fusion_fault_word",
        "unsigned long long* fusion_chain_records, unsigned int fusion_chain_stage",
    );
    let parameters = if parameters.contains("fusion_chain_records") {
        parameters
    } else {
        format!(
            "{parameters}, unsigned long long* fusion_chain_records, unsigned int fusion_chain_stage"
        )
    };
    source.replace_range(start..end, &parameters);
    let body = start + parameters.len() + 3;
    source.insert_str(body, "\n    unsigned long long* fusion_fault_word = fusion_chain_records + 2ull * fusion_chain_stage;\n    const bool fusion_chain_admitted = fusion_chain_stage == 0u || (fusion_chain_records[2ull * (fusion_chain_stage - 1u) + 1ull] == 1ull && fusion_chain_records[2ull * (fusion_chain_stage - 1u)] == 0xffffffffffffffffull);\n    if (blockIdx.x == 0u && threadIdx.x == 0u) fusion_fault_word[1] = fusion_chain_admitted ? 1ull : 2ull;\n    if (!fusion_chain_admitted) return;\n");
    Ok(source)
}

impl CudaOwnedDispatchBackend {
    pub(crate) fn retain_guarded_dispatch(
        &self,
        prepared: &CudaPreparedDispatch,
        kernel: &fusion_pcu::PcuDispatchKernelIr<'_>,
    ) -> Result<(), CudaOwnedDispatchError> {
        if prepared.guarded_function.borrow().is_some() {
            return Ok(());
        }
        let source = guarded_source(crate::lower_dispatch_to_cuda_rtc_source(kernel)?)?;
        let image = self.compile_tensor_source(&source)?;
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        *prepared.guarded_function.borrow_mut() = Some(function);
        Ok(())
    }
}

impl CudaPreparedDispatch {
    pub(crate) fn has_guarded_dispatch(&self) -> bool {
        self.guarded_function.borrow().is_some()
    }

    pub(crate) fn validate_guarded_bindings(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<(), CudaOwnedDispatchError> {
        self.validate_bindings(bindings)?;
        for binding in bindings {
            let _lease = binding.resource.acquire_stream_access(&self.stream)?;
        }
        Ok(())
    }

    /// All user resources and the private slab are leased by the one exact-stream batch.
    pub(crate) fn submit_guarded_into_batch(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        batch: &mut CudaCompletionBatch,
        records: &DeviceBuffer,
        stage: u32,
    ) -> Result<(), CudaOwnedDispatchError> {
        if !self.uses_stream(batch.stream_handle()) {
            return Err(CudaError::DifferentStream.into());
        }
        self.validate_bindings(bindings)?;
        self.runtime
            .ensure_same_runtime(&records.allocation.runtime)?;
        let minimum = (u64::from(stage) + 1) * 16;
        if (records.len() as u64) < minimum {
            return Err(CudaError::InvalidExecutionFaultWord(minimum).into());
        }
        let function = self.guarded_function.borrow();
        let function = function
            .as_ref()
            .ok_or(CudaOwnedDispatchError::UnsupportedRequirements)?;
        let stage_bytes = stage.to_ne_bytes();
        let mut arguments = smallvec::SmallVec::<[_; 8]>::new();
        for &target in &self.binding_targets {
            let binding =
                find_binding(target, bindings).map_err(CudaOwnedDispatchError::Binding)?;
            arguments.push(CudaKernelArgument::Buffer(&binding.resource));
        }
        arguments.push(CudaKernelArgument::Buffer(records));
        arguments.push(CudaKernelArgument::Bytes(&stage_bytes));
        // SAFETY: guarded_source preserves the verified user binding ABI and adds exactly
        // one slab pointer and u32 ordinal. The slab extent and runtime are validated above.
        // The launch retains every allocation lease and executable until terminal completion.
        #[cfg(feature = "allocation-census")]
        crate::ffi::guarded_kernel();
        unsafe {
            function.launch_into_batch(
                batch,
                [self.grid_x, 1, 1],
                [self.block_size, 1, 1],
                0,
                &arguments,
            )
        }?;
        Ok(())
    }

    pub(crate) fn decode_guarded_record(
        &self,
        word: u64,
        disposition: u64,
    ) -> Result<fusion_pcu::PcuGuardedExecutionStageOutcome, CudaError> {
        decode_record(self.checked_fault_contract(), word, disposition)
    }
}

fn decode_record(
    contract: Option<super::CudaPreparedFaultContract>,
    word: u64,
    disposition: u64,
) -> Result<fusion_pcu::PcuGuardedExecutionStageOutcome, CudaError> {
    use fusion_pcu::PcuGuardedExecutionStageOutcome as Stage;
    let fault = if let Some(contract) = contract {
        contract.decode(word)?
    } else if word == u64::MAX {
        None
    } else {
        return Err(CudaError::InvalidExecutionFaultWord(word));
    };
    match (disposition, fault) {
        (1, None) => Ok(Stage::Succeeded),
        (1, Some(fault)) => Ok(Stage::Fault(fault)),
        (2, None) => Ok(Stage::Skipped),
        _ => Err(CudaError::InvalidExecutionFaultWord(disposition)),
    }
}

#[cfg(test)]
mod tests {
    use super::guarded_source;

    #[test]
    fn guard_precedes_original_body_and_preserves_it() {
        let body = "\n    const unsigned int fusion_gid = 0u;\n    user_load();\n}";
        let ordinary = format!(
            "extern \"C\" __global__ void fusion_kernel(float* input, unsigned long long* fusion_fault_word) {{{body}"
        );
        let guarded = guarded_source(ordinary).unwrap();
        assert!(guarded.ends_with(body));
        assert!(
            guarded.find("if (!fusion_chain_admitted) return;").unwrap()
                < guarded.find("user_load()").unwrap()
        );
        assert!(
            guarded
                .contains("fusion_chain_records[2ull * (fusion_chain_stage - 1u) + 1ull] == 1ull")
        );
        assert_eq!(guarded.matches("void fusion_kernel(").count(), 1);
    }
    #[test]
    fn unchecked_guard_preserves_entry_and_rejects_unknown_outcomes() {
        let source = guarded_source(
            "extern \"C\" __global__ void fusion_kernel(float* output) {\n    output[0] = 3.0f;\n}"
                .into(),
        )
        .unwrap();
        assert!(source.contains("float* output, unsigned long long* fusion_chain_records, unsigned int fusion_chain_stage"));
        assert!(
            source.find("if (!fusion_chain_admitted) return;").unwrap()
                < source.find("output[0]").unwrap()
        );
        for disposition in [0, 3, u64::MAX] {
            assert!(super::decode_record(None, u64::MAX, disposition).is_err());
        }
        for disposition in [1, 2] {
            assert!(super::decode_record(None, 3, disposition).is_err());
        }
    }

    #[test]
    fn complete_record_decode_preserves_exact_law_and_catches_later_corruption() {
        use fusion_pcu::PcuGuardedExecutionStageOutcome as Stage;
        let law = fusion_pcu::PcuCheckedScalarFaultLaw::integer_binary(
            fusion_pcu::PcuScalarType::I32,
            fusion_pcu::PcuDispatchIntegerBinaryOp::Add,
            fusion_pcu::PcuRangePolicy::Reject,
        )
        .unwrap();
        let contract = Some(super::super::CudaPreparedFaultContract::scalar(65, law));
        assert!(matches!(
            super::decode_record(contract, 3, 1),
            Ok(Stage::Fault(_))
        ));
        let recovered_law = fusion_pcu::PcuCheckedScalarFaultLaw::integer_binary(
            fusion_pcu::PcuScalarType::I32,
            fusion_pcu::PcuDispatchIntegerBinaryOp::Add,
            fusion_pcu::PcuRangePolicy::Clamp,
        )
        .unwrap();
        let recovered_contract = Some(super::super::CudaPreparedFaultContract::scalar(
            65,
            recovered_law,
        ));
        assert!(matches!(
            super::decode_record(recovered_contract, (1 << 63) | 3, 1),
            Ok(Stage::Fault(fusion_pcu::PcuExecutionFault {
                recovered: true,
                ..
            }))
        ));
        assert!(super::decode_record(contract, 3, 2).is_err());
        assert!(super::decode_record(contract, (65 << 3) | 3, 1).is_err());
        assert!(super::decode_record(contract, 1, 1).is_err());
        let result = fusion_pcu::validate_guarded_execution(3, |stage| {
            super::decode_record(
                contract,
                if stage == 0 {
                    3
                } else if stage == 2 {
                    1
                } else {
                    u64::MAX
                },
                if stage == 0 { 1 } else { 2 },
            )
        });
        assert!(matches!(
            result,
            Err(fusion_pcu::PcuGuardedExecutionValidationError::StageDecode { stage: 2, .. })
        ));
    }
}
