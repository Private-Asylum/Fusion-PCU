//! Private effect-gated executable ABI; ordinary checked batching stays rejected.
#[rustfmt::skip]
use super::{
    RocmOwnedDispatchBackend,
    RocmOwnedDispatchError,
    RocmPreparedDispatch,
    DeviceBuffer,
    HipError,
    HipKernelArgument,
    HipCompletionBatch,
    PcuOwnedBinding,
    find_binding,
};

/// Wrap only the independently validated generated entry, preserving its body verbatim.
fn guarded_source(mut source: String) -> Result<String, RocmOwnedDispatchError> {
    let start = source
        .find("void fusion_kernel(")
        .ok_or(RocmOwnedDispatchError::UnsupportedRequirements)?;
    let end = source[start..]
        .find(") {")
        .map(|offset| start + offset)
        .ok_or(RocmOwnedDispatchError::UnsupportedRequirements)?;
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

impl RocmOwnedDispatchBackend {
    pub(crate) fn retain_guarded_dispatch(
        &self,
        prepared: &RocmPreparedDispatch,
        kernel: &fusion_pcu::PcuDispatchKernelIr<'_>,
    ) -> Result<(), RocmOwnedDispatchError> {
        if prepared.guarded_function.borrow().is_some() {
            return Ok(());
        }
        let source = guarded_source(crate::lower_dispatch_to_hip_rtc_source(kernel)?)?;
        let image = self.compile_tensor_source(&source)?;
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        *prepared.guarded_function.borrow_mut() = Some(function);
        Ok(())
    }
}

impl RocmPreparedDispatch {
    pub(crate) fn has_guarded_dispatch(&self) -> bool {
        self.guarded_function.borrow().is_some()
    }

    pub(crate) fn validate_guarded_bindings(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<(), RocmOwnedDispatchError> {
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
        batch: &mut HipCompletionBatch,
        records: &DeviceBuffer,
        stage: u32,
    ) -> Result<(), RocmOwnedDispatchError> {
        if !self.uses_stream(batch.stream_handle()) {
            return Err(HipError::DifferentStream.into());
        }
        self.validate_bindings(bindings)?;
        self.runtime
            .ensure_same_runtime(&records.allocation.runtime)?;
        let minimum = (u64::from(stage) + 1) * 16;
        if (records.len() as u64) < minimum {
            return Err(HipError::InvalidExecutionFaultWord(minimum).into());
        }
        let function = self.guarded_function.borrow();
        let function = function
            .as_ref()
            .ok_or(RocmOwnedDispatchError::UnsupportedRequirements)?;
        let stage_bytes = stage.to_ne_bytes();
        let mut arguments = smallvec::SmallVec::<[_; 8]>::new();
        for &target in &self.binding_targets {
            let binding =
                find_binding(target, bindings).map_err(RocmOwnedDispatchError::Binding)?;
            arguments.push(HipKernelArgument::Buffer(&binding.resource));
        }
        arguments.push(HipKernelArgument::Buffer(records));
        arguments.push(HipKernelArgument::Bytes(&stage_bytes));
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
    ) -> Result<fusion_pcu::PcuGuardedExecutionStageOutcome, HipError> {
        decode_record(
            word,
            disposition,
            self.checked_arithmetic,
            self.fault_extent,
            self.scalar_fault_word,
            self.fault_law,
        )
    }
}

fn decode_record(
    word: u64,
    disposition: u64,
    checked: bool,
    extent: u64,
    scalar_abi: bool,
    law: Option<super::fault_law::Retained>,
) -> Result<fusion_pcu::PcuGuardedExecutionStageOutcome, HipError> {
    use fusion_pcu::{
        PcuGuardedExecutionDisposition as Disposition, PcuGuardedExecutionStageOutcome as Stage,
    };
    let fault = if checked {
        super::decode_fault_word_under_law(word, extent, scalar_abi, law)?
    } else if word == u64::MAX {
        None
    } else {
        return Err(HipError::InvalidExecutionFaultWord(word));
    };
    match (Disposition::from_raw(disposition), fault) {
        (Some(Disposition::Executed), None) => Ok(Stage::Succeeded),
        (Some(Disposition::Executed), Some(fault)) => Ok(Stage::Fault(fault)),
        (Some(Disposition::Skipped), None) => Ok(Stage::Skipped),
        _ => Err(HipError::InvalidExecutionFaultWord(disposition)),
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_record, guarded_source};

    #[test]
    fn guarded_records_reject_malformed_and_validate_every_stage() {
        use fusion_pcu::*;
        let law = PcuCheckedScalarFaultLaw::integer_binary(
            PcuScalarType::I32,
            PcuDispatchIntegerBinaryOp::Add,
            PcuRangePolicy::Reject,
        )
        .unwrap();
        let decode = |word, disposition| {
            decode_record(
                word,
                disposition,
                true,
                65,
                true,
                Some(super::super::fault_law::Retained::Scalar(law)),
            )
        };
        for disposition in [0, 3, u64::MAX] {
            assert!(decode(u64::MAX, disposition).is_err());
        }
        assert!(decode((65 << 3) | 3, 1).is_err());
        assert!(decode((7 << 3) | 1, 1).is_err());
        assert!(decode((7 << 3) | 3, 2).is_err());
        assert!(decode_record(3, 1, false, 65, true, None).is_err());
        let result = validate_guarded_execution(3, |stage| match stage {
            0 => decode((7 << 3) | 3, 1),
            1 => decode(u64::MAX, 2),
            _ => decode(u64::MAX, 0),
        });
        assert!(matches!(
            result,
            Err(PcuGuardedExecutionValidationError::StageDecode { stage: 2, .. })
        ));
        let clamp = PcuCheckedScalarFaultLaw::integer_binary(
            PcuScalarType::I32,
            PcuDispatchIntegerBinaryOp::Add,
            PcuRangePolicy::Clamp,
        )
        .unwrap();
        let recovered = decode_record(
            (1 << 63) | 3,
            1,
            true,
            65,
            true,
            Some(super::super::fault_law::Retained::Scalar(clamp)),
        )
        .unwrap();
        assert!(matches!(
            validate_guarded_execution(2, |stage| Ok::<_, super::HipError>(if stage == 0 {
                recovered
            } else {
                PcuGuardedExecutionStageOutcome::Succeeded
            })),
            Err(PcuGuardedExecutionValidationError::ExecutedAfterFault { stage: 1, .. })
        ));
    }

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
}
