//! Minimal SPIR-V module emission helpers.

use super::{
    PcuSpirvCapabilityCaps,
    PcuSpirvError,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
    PcuSpirvSink,
};

pub const SPIRV_MAGIC: u32 = 0x0723_0203;

pub(crate) const OP_CAPABILITY: u16 = 17;
pub(crate) const OP_MEMORY_MODEL: u16 = 14;
pub(crate) const OP_ENTRY_POINT: u16 = 15;
pub(crate) const OP_EXECUTION_MODE: u16 = 16;
pub(crate) const OP_TYPE_VOID: u16 = 19;
pub(crate) const OP_TYPE_INT: u16 = 21;
pub(crate) const OP_TYPE_FLOAT: u16 = 22;
pub(crate) const OP_TYPE_VECTOR: u16 = 23;
pub(crate) const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
pub(crate) const OP_TYPE_STRUCT: u16 = 30;
pub(crate) const OP_TYPE_POINTER: u16 = 32;
pub(crate) const OP_TYPE_FUNCTION: u16 = 33;
pub(crate) const OP_CONSTANT: u16 = 43;
pub(crate) const OP_FUNCTION: u16 = 54;
pub(crate) const OP_FUNCTION_END: u16 = 56;
pub(crate) const OP_VARIABLE: u16 = 59;
pub(crate) const OP_LOAD: u16 = 61;
pub(crate) const OP_STORE: u16 = 62;
pub(crate) const OP_ACCESS_CHAIN: u16 = 65;
pub(crate) const OP_DECORATE: u16 = 71;
pub(crate) const OP_MEMBER_DECORATE: u16 = 72;
pub(crate) const OP_F_ADD: u16 = 129;
pub(crate) const OP_F_MUL: u16 = 133;
pub(crate) const OP_LABEL: u16 = 248;
pub(crate) const OP_RETURN: u16 = 253;

pub(crate) const CAPABILITY_SHADER: u32 = 1;
pub(crate) const ADDRESSING_MODEL_LOGICAL: u32 = 0;
pub(crate) const MEMORY_MODEL_GLSL450: u32 = 1;
pub(crate) const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
pub(crate) const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;
pub(crate) const FUNCTION_CONTROL_NONE: u32 = 0;
pub(crate) const STORAGE_CLASS_INPUT: u32 = 1;
pub(crate) const STORAGE_CLASS_UNIFORM: u32 = 2;
pub(crate) const DECORATION_BUFFER_BLOCK: u32 = 3;
pub(crate) const DECORATION_ARRAY_STRIDE: u32 = 6;
pub(crate) const DECORATION_BUILT_IN: u32 = 11;
pub(crate) const DECORATION_BINDING: u32 = 33;
pub(crate) const DECORATION_DESCRIPTOR_SET: u32 = 34;
pub(crate) const DECORATION_OFFSET: u32 = 35;
pub(crate) const BUILT_IN_GLOBAL_INVOCATION_ID: u32 = 28;

pub(crate) const VOID_TYPE_ID: u32 = 1;
pub(crate) const VOID_FUNCTION_TYPE_ID: u32 = 2;
pub(crate) const ENTRY_POINT_ID: u32 = 3;
pub(crate) const ENTRY_LABEL_ID: u32 = 4;
pub(crate) const DEFAULT_BOUND: u32 = 5;

pub(crate) const PARALLEL_FLOAT_UINT_TYPE_ID: u32 = 5;
pub(crate) const PARALLEL_FLOAT_UINT_ZERO_ID: u32 = 6;
pub(crate) const PARALLEL_FLOAT_TYPE_ID: u32 = 7;
pub(crate) const PARALLEL_FLOAT_ONE_ID: u32 = 8;
pub(crate) const PARALLEL_FLOAT_TWO_ID: u32 = 9;
pub(crate) const PARALLEL_FLOAT_VEC3_UINT_TYPE_ID: u32 = 10;
pub(crate) const PARALLEL_FLOAT_PTR_INPUT_VEC3_UINT_TYPE_ID: u32 = 11;
pub(crate) const PARALLEL_FLOAT_PTR_INPUT_UINT_TYPE_ID: u32 = 12;
pub(crate) const PARALLEL_FLOAT_RUNTIME_ARRAY_TYPE_ID: u32 = 13;
pub(crate) const PARALLEL_FLOAT_BUFFER_TYPE_ID: u32 = 14;
pub(crate) const PARALLEL_FLOAT_PTR_UNIFORM_BUFFER_TYPE_ID: u32 = 15;
pub(crate) const PARALLEL_FLOAT_PTR_UNIFORM_FLOAT_TYPE_ID: u32 = 16;
pub(crate) const PARALLEL_FLOAT_INPUT_A_VAR_ID: u32 = 17;
pub(crate) const PARALLEL_FLOAT_INPUT_B_VAR_ID: u32 = 18;
pub(crate) const PARALLEL_FLOAT_OUTPUT_VAR_ID: u32 = 19;
pub(crate) const PARALLEL_FLOAT_GLOBAL_INVOCATION_ID_VAR_ID: u32 = 20;
pub(crate) const PARALLEL_FLOAT_INDEX_PTR_ID: u32 = 21;
pub(crate) const PARALLEL_FLOAT_INDEX_ID: u32 = 22;
pub(crate) const PARALLEL_FLOAT_A_PTR_ID: u32 = 23;
pub(crate) const PARALLEL_FLOAT_A_VALUE_ID: u32 = 24;
pub(crate) const PARALLEL_FLOAT_B_PTR_ID: u32 = 25;
pub(crate) const PARALLEL_FLOAT_B_VALUE_ID: u32 = 26;
pub(crate) const PARALLEL_FLOAT_OUTPUT_PTR_ID: u32 = 27;
pub(crate) const PARALLEL_FLOAT_SCALED_ID: u32 = 28;
pub(crate) const PARALLEL_FLOAT_SUM_ID: u32 = 29;
pub(crate) const PARALLEL_FLOAT_RESULT_ID: u32 = 30;
pub(crate) const PARALLEL_FLOAT_BOUND: u32 = 31;

/// Stateful SPIR-V writer over a caller-owned sink.
pub(crate) struct PcuSpirvWriter<'a, S: PcuSpirvSink> {
    sink: &'a mut S,
    word_count: usize,
}

impl<'a, S: PcuSpirvSink> PcuSpirvWriter<'a, S> {
    pub(crate) fn new(sink: &'a mut S) -> Self {
        Self {
            sink,
            word_count: 0,
        }
    }

    pub(crate) fn emit_minimal_compute_module(
        &mut self,
        entry_point: &str,
        local_size: [u32; 3],
        options: PcuSpirvLoweringOptions,
    ) -> Result<PcuSpirvModuleInfo, PcuSpirvError> {
        self.push_header(options, DEFAULT_BOUND)?;
        self.push_instruction(OP_CAPABILITY, &[CAPABILITY_SHADER])?;
        self.push_instruction(
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        )?;
        self.push_entry_point(entry_point)?;
        self.push_instruction(
            OP_EXECUTION_MODE,
            &[
                ENTRY_POINT_ID,
                EXECUTION_MODE_LOCAL_SIZE,
                local_size[0],
                local_size[1],
                local_size[2],
            ],
        )?;
        self.push_instruction(OP_TYPE_VOID, &[VOID_TYPE_ID])?;
        self.push_instruction(OP_TYPE_FUNCTION, &[VOID_FUNCTION_TYPE_ID, VOID_TYPE_ID])?;
        self.push_instruction(
            OP_FUNCTION,
            &[
                VOID_TYPE_ID,
                ENTRY_POINT_ID,
                FUNCTION_CONTROL_NONE,
                VOID_FUNCTION_TYPE_ID,
            ],
        )?;
        self.push_instruction(OP_LABEL, &[ENTRY_LABEL_ID])?;
        self.push_instruction(OP_RETURN, &[])?;
        self.push_instruction(OP_FUNCTION_END, &[])?;

        Ok(PcuSpirvModuleInfo {
            version: options.version,
            bound: DEFAULT_BOUND,
            word_count: self.word_count,
            capabilities: PcuSpirvCapabilityCaps::SHADER,
        })
    }

    pub(crate) fn emit_parallel_float_map_module(
        &mut self,
        entry_point: &str,
        local_size: [u32; 3],
        options: PcuSpirvLoweringOptions,
    ) -> Result<PcuSpirvModuleInfo, PcuSpirvError> {
        self.push_header(options, PARALLEL_FLOAT_BOUND)?;
        self.push_instruction(OP_CAPABILITY, &[CAPABILITY_SHADER])?;
        self.push_instruction(
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        )?;
        self.push_entry_point_with_interface(
            entry_point,
            &[PARALLEL_FLOAT_GLOBAL_INVOCATION_ID_VAR_ID],
        )?;
        self.push_instruction(
            OP_EXECUTION_MODE,
            &[
                ENTRY_POINT_ID,
                EXECUTION_MODE_LOCAL_SIZE,
                local_size[0],
                local_size[1],
                local_size[2],
            ],
        )?;
        self.push_parallel_float_decorations()?;
        self.push_parallel_float_types_and_variables()?;
        self.push_parallel_float_function()?;

        Ok(PcuSpirvModuleInfo {
            version: options.version,
            bound: PARALLEL_FLOAT_BOUND,
            word_count: self.word_count,
            capabilities: PcuSpirvCapabilityCaps::SHADER,
        })
    }

    fn push_header(
        &mut self,
        options: PcuSpirvLoweringOptions,
        bound: u32,
    ) -> Result<(), PcuSpirvError> {
        self.push_word(SPIRV_MAGIC)?;
        self.push_word(options.version.0)?;
        self.push_word(options.generator)?;
        self.push_word(bound)?;
        self.push_word(0)
    }

    fn push_entry_point(&mut self, entry_point: &str) -> Result<(), PcuSpirvError> {
        self.push_entry_point_with_interface(entry_point, &[])
    }

    fn push_entry_point_with_interface(
        &mut self,
        entry_point: &str,
        interface_ids: &[u32],
    ) -> Result<(), PcuSpirvError> {
        let string_words = literal_string_word_count(entry_point);
        self.push_opcode_word(3 + string_words + interface_ids.len(), OP_ENTRY_POINT)?;
        self.push_word(EXECUTION_MODEL_GL_COMPUTE)?;
        self.push_word(ENTRY_POINT_ID)?;
        self.push_literal_string(entry_point)?;
        for id in interface_ids.iter().copied() {
            self.push_word(id)?;
        }
        Ok(())
    }

    fn push_parallel_float_decorations(&mut self) -> Result<(), PcuSpirvError> {
        self.push_instruction(
            OP_DECORATE,
            &[
                PARALLEL_FLOAT_GLOBAL_INVOCATION_ID_VAR_ID,
                DECORATION_BUILT_IN,
                BUILT_IN_GLOBAL_INVOCATION_ID,
            ],
        )?;
        self.push_instruction(
            OP_DECORATE,
            &[
                PARALLEL_FLOAT_RUNTIME_ARRAY_TYPE_ID,
                DECORATION_ARRAY_STRIDE,
                4,
            ],
        )?;
        self.push_instruction(
            OP_MEMBER_DECORATE,
            &[PARALLEL_FLOAT_BUFFER_TYPE_ID, 0, DECORATION_OFFSET, 0],
        )?;
        self.push_instruction(
            OP_DECORATE,
            &[PARALLEL_FLOAT_BUFFER_TYPE_ID, DECORATION_BUFFER_BLOCK],
        )?;
        self.push_descriptor_decorations(PARALLEL_FLOAT_INPUT_A_VAR_ID, 0, 0)?;
        self.push_descriptor_decorations(PARALLEL_FLOAT_INPUT_B_VAR_ID, 0, 1)?;
        self.push_descriptor_decorations(PARALLEL_FLOAT_OUTPUT_VAR_ID, 0, 2)
    }

    fn push_descriptor_decorations(
        &mut self,
        variable_id: u32,
        descriptor_set: u32,
        binding: u32,
    ) -> Result<(), PcuSpirvError> {
        self.push_instruction(
            OP_DECORATE,
            &[variable_id, DECORATION_DESCRIPTOR_SET, descriptor_set],
        )?;
        self.push_instruction(OP_DECORATE, &[variable_id, DECORATION_BINDING, binding])
    }

    fn push_parallel_float_types_and_variables(&mut self) -> Result<(), PcuSpirvError> {
        self.push_instruction(OP_TYPE_VOID, &[VOID_TYPE_ID])?;
        self.push_instruction(OP_TYPE_FUNCTION, &[VOID_FUNCTION_TYPE_ID, VOID_TYPE_ID])?;
        self.push_instruction(OP_TYPE_INT, &[PARALLEL_FLOAT_UINT_TYPE_ID, 32, 0])?;
        self.push_instruction(
            OP_CONSTANT,
            &[PARALLEL_FLOAT_UINT_TYPE_ID, PARALLEL_FLOAT_UINT_ZERO_ID, 0],
        )?;
        self.push_instruction(OP_TYPE_FLOAT, &[PARALLEL_FLOAT_TYPE_ID, 32])?;
        self.push_instruction(
            OP_CONSTANT,
            &[
                PARALLEL_FLOAT_TYPE_ID,
                PARALLEL_FLOAT_ONE_ID,
                1.0_f32.to_bits(),
            ],
        )?;
        self.push_instruction(
            OP_CONSTANT,
            &[
                PARALLEL_FLOAT_TYPE_ID,
                PARALLEL_FLOAT_TWO_ID,
                2.0_f32.to_bits(),
            ],
        )?;
        self.push_instruction(
            OP_TYPE_VECTOR,
            &[
                PARALLEL_FLOAT_VEC3_UINT_TYPE_ID,
                PARALLEL_FLOAT_UINT_TYPE_ID,
                3,
            ],
        )?;
        self.push_instruction(
            OP_TYPE_POINTER,
            &[
                PARALLEL_FLOAT_PTR_INPUT_VEC3_UINT_TYPE_ID,
                STORAGE_CLASS_INPUT,
                PARALLEL_FLOAT_VEC3_UINT_TYPE_ID,
            ],
        )?;
        self.push_instruction(
            OP_TYPE_POINTER,
            &[
                PARALLEL_FLOAT_PTR_INPUT_UINT_TYPE_ID,
                STORAGE_CLASS_INPUT,
                PARALLEL_FLOAT_UINT_TYPE_ID,
            ],
        )?;
        self.push_instruction(
            OP_TYPE_RUNTIME_ARRAY,
            &[PARALLEL_FLOAT_RUNTIME_ARRAY_TYPE_ID, PARALLEL_FLOAT_TYPE_ID],
        )?;
        self.push_instruction(
            OP_TYPE_STRUCT,
            &[
                PARALLEL_FLOAT_BUFFER_TYPE_ID,
                PARALLEL_FLOAT_RUNTIME_ARRAY_TYPE_ID,
            ],
        )?;
        self.push_instruction(
            OP_TYPE_POINTER,
            &[
                PARALLEL_FLOAT_PTR_UNIFORM_BUFFER_TYPE_ID,
                STORAGE_CLASS_UNIFORM,
                PARALLEL_FLOAT_BUFFER_TYPE_ID,
            ],
        )?;
        self.push_instruction(
            OP_TYPE_POINTER,
            &[
                PARALLEL_FLOAT_PTR_UNIFORM_FLOAT_TYPE_ID,
                STORAGE_CLASS_UNIFORM,
                PARALLEL_FLOAT_TYPE_ID,
            ],
        )?;
        self.push_instruction(
            OP_VARIABLE,
            &[
                PARALLEL_FLOAT_PTR_UNIFORM_BUFFER_TYPE_ID,
                PARALLEL_FLOAT_INPUT_A_VAR_ID,
                STORAGE_CLASS_UNIFORM,
            ],
        )?;
        self.push_instruction(
            OP_VARIABLE,
            &[
                PARALLEL_FLOAT_PTR_UNIFORM_BUFFER_TYPE_ID,
                PARALLEL_FLOAT_INPUT_B_VAR_ID,
                STORAGE_CLASS_UNIFORM,
            ],
        )?;
        self.push_instruction(
            OP_VARIABLE,
            &[
                PARALLEL_FLOAT_PTR_UNIFORM_BUFFER_TYPE_ID,
                PARALLEL_FLOAT_OUTPUT_VAR_ID,
                STORAGE_CLASS_UNIFORM,
            ],
        )?;
        self.push_instruction(
            OP_VARIABLE,
            &[
                PARALLEL_FLOAT_PTR_INPUT_VEC3_UINT_TYPE_ID,
                PARALLEL_FLOAT_GLOBAL_INVOCATION_ID_VAR_ID,
                STORAGE_CLASS_INPUT,
            ],
        )
    }

    fn push_parallel_float_function(&mut self) -> Result<(), PcuSpirvError> {
        self.push_instruction(
            OP_FUNCTION,
            &[
                VOID_TYPE_ID,
                ENTRY_POINT_ID,
                FUNCTION_CONTROL_NONE,
                VOID_FUNCTION_TYPE_ID,
            ],
        )?;
        self.push_instruction(OP_LABEL, &[ENTRY_LABEL_ID])?;
        self.push_instruction(
            OP_ACCESS_CHAIN,
            &[
                PARALLEL_FLOAT_PTR_INPUT_UINT_TYPE_ID,
                PARALLEL_FLOAT_INDEX_PTR_ID,
                PARALLEL_FLOAT_GLOBAL_INVOCATION_ID_VAR_ID,
                PARALLEL_FLOAT_UINT_ZERO_ID,
            ],
        )?;
        self.push_instruction(
            OP_LOAD,
            &[
                PARALLEL_FLOAT_UINT_TYPE_ID,
                PARALLEL_FLOAT_INDEX_ID,
                PARALLEL_FLOAT_INDEX_PTR_ID,
            ],
        )?;
        self.push_storage_float_access(
            PARALLEL_FLOAT_A_PTR_ID,
            PARALLEL_FLOAT_INPUT_A_VAR_ID,
            PARALLEL_FLOAT_INDEX_ID,
        )?;
        self.push_instruction(
            OP_LOAD,
            &[
                PARALLEL_FLOAT_TYPE_ID,
                PARALLEL_FLOAT_A_VALUE_ID,
                PARALLEL_FLOAT_A_PTR_ID,
            ],
        )?;
        self.push_storage_float_access(
            PARALLEL_FLOAT_B_PTR_ID,
            PARALLEL_FLOAT_INPUT_B_VAR_ID,
            PARALLEL_FLOAT_INDEX_ID,
        )?;
        self.push_instruction(
            OP_LOAD,
            &[
                PARALLEL_FLOAT_TYPE_ID,
                PARALLEL_FLOAT_B_VALUE_ID,
                PARALLEL_FLOAT_B_PTR_ID,
            ],
        )?;
        self.push_storage_float_access(
            PARALLEL_FLOAT_OUTPUT_PTR_ID,
            PARALLEL_FLOAT_OUTPUT_VAR_ID,
            PARALLEL_FLOAT_INDEX_ID,
        )?;
        self.push_instruction(
            OP_F_MUL,
            &[
                PARALLEL_FLOAT_TYPE_ID,
                PARALLEL_FLOAT_SCALED_ID,
                PARALLEL_FLOAT_A_VALUE_ID,
                PARALLEL_FLOAT_TWO_ID,
            ],
        )?;
        self.push_instruction(
            OP_F_ADD,
            &[
                PARALLEL_FLOAT_TYPE_ID,
                PARALLEL_FLOAT_SUM_ID,
                PARALLEL_FLOAT_SCALED_ID,
                PARALLEL_FLOAT_B_VALUE_ID,
            ],
        )?;
        self.push_instruction(
            OP_F_ADD,
            &[
                PARALLEL_FLOAT_TYPE_ID,
                PARALLEL_FLOAT_RESULT_ID,
                PARALLEL_FLOAT_SUM_ID,
                PARALLEL_FLOAT_ONE_ID,
            ],
        )?;
        self.push_instruction(
            OP_STORE,
            &[PARALLEL_FLOAT_OUTPUT_PTR_ID, PARALLEL_FLOAT_RESULT_ID],
        )?;
        self.push_instruction(OP_RETURN, &[])?;
        self.push_instruction(OP_FUNCTION_END, &[])
    }

    fn push_storage_float_access(
        &mut self,
        result_id: u32,
        variable_id: u32,
        index_id: u32,
    ) -> Result<(), PcuSpirvError> {
        self.push_instruction(
            OP_ACCESS_CHAIN,
            &[
                PARALLEL_FLOAT_PTR_UNIFORM_FLOAT_TYPE_ID,
                result_id,
                variable_id,
                PARALLEL_FLOAT_UINT_ZERO_ID,
                index_id,
            ],
        )
    }

    fn push_instruction(&mut self, opcode: u16, operands: &[u32]) -> Result<(), PcuSpirvError> {
        self.push_opcode_word(operands.len() + 1, opcode)?;
        for operand in operands.iter().copied() {
            self.push_word(operand)?;
        }
        Ok(())
    }

    fn push_opcode_word(&mut self, word_count: usize, opcode: u16) -> Result<(), PcuSpirvError> {
        let Ok(word_count) = u32::try_from(word_count) else {
            return Err(PcuSpirvError::IdSpaceExhausted);
        };
        self.push_word((word_count << 16) | u32::from(opcode))
    }

    fn push_literal_string(&mut self, value: &str) -> Result<(), PcuSpirvError> {
        let mut word = 0_u32;
        let mut shift = 0_u32;
        for byte in value.bytes().chain(core::iter::once(0)) {
            word |= u32::from(byte) << shift;
            shift += 8;
            if shift == 32 {
                self.push_word(word)?;
                word = 0;
                shift = 0;
            }
        }
        if shift != 0 {
            self.push_word(word)?;
        }
        Ok(())
    }

    fn push_word(&mut self, word: u32) -> Result<(), PcuSpirvError> {
        self.sink.push_word(word)?;
        self.word_count += 1;
        Ok(())
    }
}

pub(crate) fn literal_string_word_count(value: &str) -> usize {
    (value.len() + 1).div_ceil(4)
}
