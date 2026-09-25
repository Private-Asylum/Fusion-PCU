//! Minimal SPIR-V module emission helpers.

use super::{
    PcuSpirvCapabilityCaps,
    PcuSpirvError,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
    PcuSpirvSink,
};
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuParameterValue,
};

pub const SPIRV_MAGIC: u32 = 0x0723_0203;

pub(crate) const OP_CAPABILITY: u16 = 17;
pub(crate) const OP_MEMORY_MODEL: u16 = 14;
pub(crate) const OP_ENTRY_POINT: u16 = 15;
pub(crate) const OP_EXECUTION_MODE: u16 = 16;
pub(crate) const OP_TYPE_VOID: u16 = 19;
pub(crate) const OP_TYPE_BOOL: u16 = 20;
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
pub(crate) const OP_F_SUB: u16 = 131;
pub(crate) const OP_F_MUL: u16 = 133;
pub(crate) const OP_F_DIV: u16 = 136;
pub(crate) const OP_I_ADD: u16 = 128;
pub(crate) const OP_I_SUB: u16 = 130;
pub(crate) const OP_U_LESS_THAN: u16 = 176;
pub(crate) const OP_U_LESS_THAN_EQUAL: u16 = 178;
pub(crate) const OP_LOGICAL_AND: u16 = 167;
pub(crate) const OP_SELECT: u16 = 169;
pub(crate) const OP_LOOP_MERGE: u16 = 246;
pub(crate) const OP_BRANCH: u16 = 249;
pub(crate) const OP_BRANCH_CONDITIONAL: u16 = 250;
pub(crate) const OP_PHI: u16 = 245;
pub(crate) const OP_LABEL: u16 = 248;
pub(crate) const OP_RETURN: u16 = 253;

pub(crate) const CAPABILITY_SHADER: u32 = 1;
pub(crate) const ADDRESSING_MODEL_LOGICAL: u32 = 0;
pub(crate) const MEMORY_MODEL_GLSL450: u32 = 1;
pub(crate) const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
pub(crate) const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;
/// The first bounded lowering profile uses one invocation per workgroup. The PCU logical
/// shape describes the dispatch extent and is supplied by the host through group counts.
pub(crate) const DEFAULT_LOCAL_SIZE: [u32; 3] = [1, 1, 1];
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

const DATAFLOW_UINT_TYPE_ID: u32 = 5;
const DATAFLOW_UINT_ZERO_ID: u32 = 6;
const DATAFLOW_FLOAT_TYPE_ID: u32 = 7;
const DATAFLOW_VEC3_UINT_TYPE_ID: u32 = 8;
const DATAFLOW_PTR_INPUT_VEC3_UINT_TYPE_ID: u32 = 9;
const DATAFLOW_PTR_INPUT_UINT_TYPE_ID: u32 = 10;
const DATAFLOW_RUNTIME_ARRAY_TYPE_ID: u32 = 11;
const DATAFLOW_BUFFER_TYPE_ID: u32 = 12;
const DATAFLOW_PTR_UNIFORM_BUFFER_TYPE_ID: u32 = 13;
const DATAFLOW_PTR_UNIFORM_FLOAT_TYPE_ID: u32 = 14;
const DATAFLOW_GLOBAL_INVOCATION_ID_VAR_ID: u32 = 15;
const DATAFLOW_INDEX_PTR_ID: u32 = 16;
const DATAFLOW_INDEX_ID: u32 = 17;
const DATAFLOW_BOOL_TYPE_ID: u32 = 18;
const DATAFLOW_BINDING_VAR_BASE_ID: u32 = 19;
const DATAFLOW_LOOP_AUX_ID_COUNT: u32 = 14;

/// Stateful SPIR-V writer over a caller-owned sink.
pub(crate) struct PcuSpirvWriter<'a, S: PcuSpirvSink> {
    sink: &'a mut S,
    word_count: usize,
}

impl<'a, S: PcuSpirvSink> PcuSpirvWriter<'a, S> {
    pub(crate) const fn new(sink: &'a mut S) -> Self {
        Self {
            sink,
            word_count: 0,
        }
    }

    pub(crate) fn emit_minimal_compute_module(
        &mut self,
        entry_point: &str,
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
                DEFAULT_LOCAL_SIZE[0],
                DEFAULT_LOCAL_SIZE[1],
                DEFAULT_LOCAL_SIZE[2],
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
                DEFAULT_LOCAL_SIZE[0],
                DEFAULT_LOCAL_SIZE[1],
                DEFAULT_LOCAL_SIZE[2],
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

    pub(crate) fn emit_f32_dataflow_map_module(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
        options: PcuSpirvLoweringOptions,
    ) -> Result<PcuSpirvModuleInfo, PcuSpirvError> {
        let bound = dataflow_bound(kernel)?;
        self.push_header(options, bound)?;
        self.push_instruction(OP_CAPABILITY, &[CAPABILITY_SHADER])?;
        self.push_instruction(
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        )?;
        self.push_entry_point_with_interface(
            kernel.entry.name,
            &[DATAFLOW_GLOBAL_INVOCATION_ID_VAR_ID],
        )?;
        self.push_instruction(
            OP_EXECUTION_MODE,
            &[
                ENTRY_POINT_ID,
                EXECUTION_MODE_LOCAL_SIZE,
                DEFAULT_LOCAL_SIZE[0],
                DEFAULT_LOCAL_SIZE[1],
                DEFAULT_LOCAL_SIZE[2],
            ],
        )?;
        self.push_f32_dataflow_decorations(kernel)?;
        self.push_f32_dataflow_types_constants_and_variables(kernel)?;
        self.push_f32_dataflow_function(kernel)?;

        Ok(PcuSpirvModuleInfo {
            version: options.version,
            bound,
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

    fn push_f32_dataflow_decorations(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<(), PcuSpirvError> {
        self.push_instruction(
            OP_DECORATE,
            &[
                DATAFLOW_GLOBAL_INVOCATION_ID_VAR_ID,
                DECORATION_BUILT_IN,
                BUILT_IN_GLOBAL_INVOCATION_ID,
            ],
        )?;
        self.push_instruction(
            OP_DECORATE,
            &[DATAFLOW_RUNTIME_ARRAY_TYPE_ID, DECORATION_ARRAY_STRIDE, 4],
        )?;
        self.push_instruction(
            OP_MEMBER_DECORATE,
            &[DATAFLOW_BUFFER_TYPE_ID, 0, DECORATION_OFFSET, 0],
        )?;
        self.push_instruction(
            OP_DECORATE,
            &[DATAFLOW_BUFFER_TYPE_ID, DECORATION_BUFFER_BLOCK],
        )?;
        for (index, binding) in kernel.bindings.iter().copied().enumerate() {
            self.push_descriptor_decorations(
                dataflow_binding_var_id(index)?,
                binding.set,
                binding.binding,
            )?;
        }
        Ok(())
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
        self.push_parallel_float_variables()
    }

    fn push_parallel_float_variables(&mut self) -> Result<(), PcuSpirvError> {
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

    #[allow(clippy::too_many_lines)] // Types and globals must precede the function body in SPIR-V.
    fn push_f32_dataflow_types_constants_and_variables(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<(), PcuSpirvError> {
        self.push_instruction(OP_TYPE_VOID, &[VOID_TYPE_ID])?;
        self.push_instruction(OP_TYPE_FUNCTION, &[VOID_FUNCTION_TYPE_ID, VOID_TYPE_ID])?;
        self.push_instruction(OP_TYPE_INT, &[DATAFLOW_UINT_TYPE_ID, 32, 0])?;
        self.push_instruction(OP_TYPE_BOOL, &[DATAFLOW_BOOL_TYPE_ID])?;
        self.push_instruction(
            OP_CONSTANT,
            &[DATAFLOW_UINT_TYPE_ID, DATAFLOW_UINT_ZERO_ID, 0],
        )?;
        self.push_instruction(OP_TYPE_FLOAT, &[DATAFLOW_FLOAT_TYPE_ID, 32])?;
        for op in kernel.ops.iter().copied() {
            if let PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result,
                value: PcuParameterValue::F32(bits),
            }) = op
            {
                self.push_instruction(
                    OP_CONSTANT,
                    &[
                        DATAFLOW_FLOAT_TYPE_ID,
                        dataflow_value_id(kernel, result)?,
                        bits,
                    ],
                )?;
            }
            if let PcuDispatchOp::GridStrideLoop { body, .. } = op {
                for body_op in body.iter().copied() {
                    if let PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                        result,
                        value: PcuParameterValue::F32(bits),
                    }) = body_op
                    {
                        self.push_instruction(
                            OP_CONSTANT,
                            &[
                                DATAFLOW_FLOAT_TYPE_ID,
                                dataflow_value_id(kernel, result)?,
                                bits,
                            ],
                        )?;
                    }
                }
            }
            if let PcuDispatchOp::GridStrideLoop { extent, .. } = op {
                let ids = dataflow_loop_ids(kernel)?;
                self.push_instruction(OP_CONSTANT, &[DATAFLOW_UINT_TYPE_ID, ids.extent, extent])?;
                self.push_instruction(
                    OP_CONSTANT,
                    &[
                        DATAFLOW_UINT_TYPE_ID,
                        ids.stride,
                        kernel.entry.logical_shape[0],
                    ],
                )?;
            }
        }
        self.push_instruction(
            OP_TYPE_VECTOR,
            &[DATAFLOW_VEC3_UINT_TYPE_ID, DATAFLOW_UINT_TYPE_ID, 3],
        )?;
        self.push_instruction(
            OP_TYPE_POINTER,
            &[
                DATAFLOW_PTR_INPUT_VEC3_UINT_TYPE_ID,
                STORAGE_CLASS_INPUT,
                DATAFLOW_VEC3_UINT_TYPE_ID,
            ],
        )?;
        self.push_instruction(
            OP_TYPE_POINTER,
            &[
                DATAFLOW_PTR_INPUT_UINT_TYPE_ID,
                STORAGE_CLASS_INPUT,
                DATAFLOW_UINT_TYPE_ID,
            ],
        )?;
        self.push_instruction(
            OP_TYPE_RUNTIME_ARRAY,
            &[DATAFLOW_RUNTIME_ARRAY_TYPE_ID, DATAFLOW_FLOAT_TYPE_ID],
        )?;
        self.push_instruction(
            OP_TYPE_STRUCT,
            &[DATAFLOW_BUFFER_TYPE_ID, DATAFLOW_RUNTIME_ARRAY_TYPE_ID],
        )?;
        self.push_instruction(
            OP_TYPE_POINTER,
            &[
                DATAFLOW_PTR_UNIFORM_BUFFER_TYPE_ID,
                STORAGE_CLASS_UNIFORM,
                DATAFLOW_BUFFER_TYPE_ID,
            ],
        )?;
        self.push_instruction(
            OP_TYPE_POINTER,
            &[
                DATAFLOW_PTR_UNIFORM_FLOAT_TYPE_ID,
                STORAGE_CLASS_UNIFORM,
                DATAFLOW_FLOAT_TYPE_ID,
            ],
        )?;
        for (index, _) in kernel.bindings.iter().enumerate() {
            self.push_instruction(
                OP_VARIABLE,
                &[
                    DATAFLOW_PTR_UNIFORM_BUFFER_TYPE_ID,
                    dataflow_binding_var_id(index)?,
                    STORAGE_CLASS_UNIFORM,
                ],
            )?;
        }
        self.push_instruction(
            OP_VARIABLE,
            &[
                DATAFLOW_PTR_INPUT_VEC3_UINT_TYPE_ID,
                DATAFLOW_GLOBAL_INVOCATION_ID_VAR_ID,
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

    fn push_f32_dataflow_function(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<(), PcuSpirvError> {
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
                DATAFLOW_PTR_INPUT_UINT_TYPE_ID,
                DATAFLOW_INDEX_PTR_ID,
                DATAFLOW_GLOBAL_INVOCATION_ID_VAR_ID,
                DATAFLOW_UINT_ZERO_ID,
            ],
        )?;
        self.push_instruction(
            OP_LOAD,
            &[
                DATAFLOW_UINT_TYPE_ID,
                DATAFLOW_INDEX_ID,
                DATAFLOW_INDEX_PTR_ID,
            ],
        )?;
        if let Some(body) = kernel.ops.iter().find_map(|op| match op {
            PcuDispatchOp::GridStrideLoop { body, .. } => Some(*body),
            _ => None,
        }) {
            return self.push_f32_grid_stride_function_body(kernel, body);
        }
        for (op_index, op) in kernel.ops.iter().copied().enumerate() {
            match op {
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result,
                    binding,
                    index,
                }) => self.push_f32_dataflow_load(kernel, op_index, result, binding, index)?,
                PcuDispatchOp::Data(PcuDispatchDataOp::Constant { .. }) => {}
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    result,
                    op,
                    lhs,
                    rhs,
                }) => self.push_instruction(
                    dataflow_alu_opcode(op)?,
                    &[
                        DATAFLOW_FLOAT_TYPE_ID,
                        dataflow_value_id(kernel, result)?,
                        dataflow_value_id(kernel, lhs)?,
                        dataflow_value_id(kernel, rhs)?,
                    ],
                )?,
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding,
                    index,
                    value,
                }) => self.push_f32_dataflow_store(kernel, op_index, binding, index, value)?,
                PcuDispatchOp::Control(PcuDispatchControlOp::Return) => {
                    self.push_instruction(OP_RETURN, &[])?;
                }
                _ => return Err(PcuSpirvError::UnsupportedInstruction(op.support_flag())),
            }
        }
        self.push_instruction(OP_FUNCTION_END, &[])
    }

    #[allow(clippy::too_many_lines)] // Keep the structured loop's block and phi wiring together.
    fn push_f32_grid_stride_function_body(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
        body: &[PcuDispatchOp<'_>],
    ) -> Result<(), PcuSpirvError> {
        let ids = dataflow_loop_ids(kernel)?;
        self.push_instruction(OP_BRANCH, &[ids.header])?;
        self.push_instruction(OP_LABEL, &[ids.header])?;
        self.push_instruction(
            OP_PHI,
            &[
                DATAFLOW_UINT_TYPE_ID,
                ids.index,
                DATAFLOW_INDEX_ID,
                ENTRY_LABEL_ID,
                ids.next_index,
                ids.continue_label,
            ],
        )?;
        self.push_instruction(
            OP_U_LESS_THAN,
            &[
                DATAFLOW_BOOL_TYPE_ID,
                ids.index_in_range,
                ids.index,
                ids.extent,
            ],
        )?;
        self.push_instruction(
            OP_U_LESS_THAN,
            &[
                DATAFLOW_BOOL_TYPE_ID,
                ids.base_in_range,
                DATAFLOW_INDEX_ID,
                ids.stride,
            ],
        )?;
        self.push_instruction(
            OP_LOGICAL_AND,
            &[
                DATAFLOW_BOOL_TYPE_ID,
                ids.condition,
                ids.index_in_range,
                ids.base_in_range,
            ],
        )?;
        self.push_instruction(OP_LOOP_MERGE, &[ids.merge_label, ids.continue_label, 0])?;
        self.push_instruction(
            OP_BRANCH_CONDITIONAL,
            &[ids.condition, ids.body_label, ids.merge_label],
        )?;
        self.push_instruction(OP_LABEL, &[ids.body_label])?;
        let top_index = kernel
            .ops
            .iter()
            .position(|op| matches!(op, PcuDispatchOp::GridStrideLoop { .. }))
            .ok_or(PcuSpirvError::InvalidKernelSignature)?;
        for (body_index, op) in body.iter().copied().enumerate() {
            let ptr_index = top_index
                .checked_add(body_index + 1)
                .ok_or(PcuSpirvError::IdSpaceExhausted)?;
            match op {
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                    result,
                    binding,
                    index,
                }) => self.push_f32_dataflow_load_at(kernel, ptr_index, result, binding, index)?,
                PcuDispatchOp::Data(PcuDispatchDataOp::Constant { .. }) => {}
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                    result,
                    op,
                    lhs,
                    rhs,
                }) => self.push_instruction(
                    dataflow_alu_opcode(op)?,
                    &[
                        DATAFLOW_FLOAT_TYPE_ID,
                        dataflow_value_id(kernel, result)?,
                        dataflow_value_id(kernel, lhs)?,
                        dataflow_value_id(kernel, rhs)?,
                    ],
                )?,
                PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                    binding,
                    index,
                    value,
                }) => self.push_f32_dataflow_store_at(kernel, ptr_index, binding, index, value)?,
                _ => return Err(PcuSpirvError::UnsupportedInstruction(op.support_flag())),
            }
        }
        self.push_instruction(OP_BRANCH, &[ids.continue_label])?;
        self.push_instruction(OP_LABEL, &[ids.continue_label])?;
        self.push_instruction(
            OP_I_ADD,
            &[
                DATAFLOW_UINT_TYPE_ID,
                ids.raw_next_index,
                ids.index,
                ids.stride,
            ],
        )?;
        self.push_instruction(
            OP_I_SUB,
            &[DATAFLOW_UINT_TYPE_ID, ids.remaining, ids.extent, ids.index],
        )?;
        self.push_instruction(
            OP_U_LESS_THAN_EQUAL,
            &[
                DATAFLOW_BOOL_TYPE_ID,
                ids.stop_after_this_iteration,
                ids.remaining,
                ids.stride,
            ],
        )?;
        self.push_instruction(
            OP_SELECT,
            &[
                DATAFLOW_UINT_TYPE_ID,
                ids.next_index,
                ids.stop_after_this_iteration,
                ids.extent,
                ids.raw_next_index,
            ],
        )?;
        self.push_instruction(OP_BRANCH, &[ids.header])?;
        self.push_instruction(OP_LABEL, &[ids.merge_label])?;
        self.push_instruction(OP_RETURN, &[])?;
        self.push_instruction(OP_FUNCTION_END, &[])
    }

    fn push_f32_dataflow_load(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
        op_index: usize,
        result: PcuDispatchValueId,
        binding: PcuBindingRef,
        index: PcuDispatchIndex,
    ) -> Result<(), PcuSpirvError> {
        self.push_f32_dataflow_load_at(kernel, op_index, result, binding, index)
    }

    fn push_f32_dataflow_load_at(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
        op_index: usize,
        result: PcuDispatchValueId,
        binding: PcuBindingRef,
        index: PcuDispatchIndex,
    ) -> Result<(), PcuSpirvError> {
        let ptr_id = dataflow_op_ptr_id(kernel, op_index)?;
        let variable_id = dataflow_binding_var_id(dataflow_binding_ordinal(kernel, binding)?)?;
        self.push_f32_dataflow_access(ptr_id, variable_id, dataflow_index_id(kernel, index)?)?;
        self.push_instruction(
            OP_LOAD,
            &[
                DATAFLOW_FLOAT_TYPE_ID,
                dataflow_value_id(kernel, result)?,
                ptr_id,
            ],
        )
    }

    fn push_f32_dataflow_store(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
        op_index: usize,
        binding: PcuBindingRef,
        index: PcuDispatchIndex,
        value: PcuDispatchValueId,
    ) -> Result<(), PcuSpirvError> {
        self.push_f32_dataflow_store_at(kernel, op_index, binding, index, value)
    }

    fn push_f32_dataflow_store_at(
        &mut self,
        kernel: &PcuDispatchKernelIr<'_>,
        op_index: usize,
        binding: PcuBindingRef,
        index: PcuDispatchIndex,
        value: PcuDispatchValueId,
    ) -> Result<(), PcuSpirvError> {
        let ptr_id = dataflow_op_ptr_id(kernel, op_index)?;
        let variable_id = dataflow_binding_var_id(dataflow_binding_ordinal(kernel, binding)?)?;
        self.push_f32_dataflow_access(ptr_id, variable_id, dataflow_index_id(kernel, index)?)?;
        self.push_instruction(OP_STORE, &[ptr_id, dataflow_value_id(kernel, value)?])
    }

    fn push_f32_dataflow_access(
        &mut self,
        result_id: u32,
        variable_id: u32,
        index_id: u32,
    ) -> Result<(), PcuSpirvError> {
        self.push_instruction(
            OP_ACCESS_CHAIN,
            &[
                DATAFLOW_PTR_UNIFORM_FLOAT_TYPE_ID,
                result_id,
                variable_id,
                DATAFLOW_UINT_ZERO_ID,
                index_id,
            ],
        )
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

const fn dataflow_alu_opcode(op: PcuDispatchAluOp) -> Result<u16, PcuSpirvError> {
    match op {
        PcuDispatchAluOp::Add => Ok(OP_F_ADD),
        PcuDispatchAluOp::Sub => Ok(OP_F_SUB),
        PcuDispatchAluOp::Mul => Ok(OP_F_MUL),
        PcuDispatchAluOp::Div => Ok(OP_F_DIV),
        _ => Err(PcuSpirvError::UnsupportedInstruction(op.support_flag())),
    }
}

fn dataflow_bound(kernel: &PcuDispatchKernelIr<'_>) -> Result<u32, PcuSpirvError> {
    let binding_count =
        u32::try_from(kernel.bindings.len()).map_err(|_| PcuSpirvError::IdSpaceExhausted)?;
    let op_count =
        u32::try_from(dataflow_op_count(kernel)).map_err(|_| PcuSpirvError::IdSpaceExhausted)?;
    let max_value = u32::from(max_dataflow_value_id(kernel));
    let value_base = DATAFLOW_BINDING_VAR_BASE_ID
        .checked_add(binding_count)
        .ok_or(PcuSpirvError::IdSpaceExhausted)?;
    let ptr_base = value_base
        .checked_add(max_value)
        .ok_or(PcuSpirvError::IdSpaceExhausted)?;
    let loop_aux_count = if kernel
        .ops
        .iter()
        .any(|op| matches!(op, PcuDispatchOp::GridStrideLoop { .. }))
    {
        DATAFLOW_LOOP_AUX_ID_COUNT
    } else {
        0
    };
    ptr_base
        .checked_add(op_count)
        .and_then(|base| base.checked_add(loop_aux_count))
        .ok_or(PcuSpirvError::IdSpaceExhausted)
}

fn dataflow_op_count(kernel: &PcuDispatchKernelIr<'_>) -> usize {
    kernel
        .ops
        .iter()
        .map(|op| match op {
            PcuDispatchOp::GridStrideLoop { body, .. } => body.len().saturating_add(1),
            _ => 1,
        })
        .sum()
}

#[derive(Clone, Copy)]
struct DataflowLoopIds {
    header: u32,
    body_label: u32,
    continue_label: u32,
    merge_label: u32,
    index: u32,
    extent: u32,
    stride: u32,
    index_in_range: u32,
    base_in_range: u32,
    condition: u32,
    next_index: u32,
    raw_next_index: u32,
    remaining: u32,
    stop_after_this_iteration: u32,
}

fn dataflow_loop_ids(kernel: &PcuDispatchKernelIr<'_>) -> Result<DataflowLoopIds, PcuSpirvError> {
    let op_count =
        u32::try_from(dataflow_op_count(kernel)).map_err(|_| PcuSpirvError::IdSpaceExhausted)?;
    let start = dataflow_op_ptr_base(kernel)?
        .checked_add(op_count)
        .ok_or(PcuSpirvError::IdSpaceExhausted)?;
    let id = |offset| {
        start
            .checked_add(offset)
            .ok_or(PcuSpirvError::IdSpaceExhausted)
    };
    Ok(DataflowLoopIds {
        header: id(0)?,
        body_label: id(1)?,
        continue_label: id(2)?,
        merge_label: id(3)?,
        index: id(4)?,
        extent: id(5)?,
        stride: id(6)?,
        index_in_range: id(7)?,
        base_in_range: id(8)?,
        condition: id(9)?,
        next_index: id(10)?,
        raw_next_index: id(11)?,
        remaining: id(12)?,
        stop_after_this_iteration: id(13)?,
    })
}

fn dataflow_binding_var_id(index: usize) -> Result<u32, PcuSpirvError> {
    let index = u32::try_from(index).map_err(|_| PcuSpirvError::IdSpaceExhausted)?;
    DATAFLOW_BINDING_VAR_BASE_ID
        .checked_add(index)
        .ok_or(PcuSpirvError::IdSpaceExhausted)
}

fn dataflow_value_base(kernel: &PcuDispatchKernelIr<'_>) -> Result<u32, PcuSpirvError> {
    let binding_count =
        u32::try_from(kernel.bindings.len()).map_err(|_| PcuSpirvError::IdSpaceExhausted)?;
    DATAFLOW_BINDING_VAR_BASE_ID
        .checked_add(binding_count)
        .ok_or(PcuSpirvError::IdSpaceExhausted)
}

fn dataflow_value_id(
    kernel: &PcuDispatchKernelIr<'_>,
    value: PcuDispatchValueId,
) -> Result<u32, PcuSpirvError> {
    if value.0 == 0 {
        return Err(PcuSpirvError::InvalidKernelSignature);
    }
    dataflow_value_base(kernel)?
        .checked_add(u32::from(value.0 - 1))
        .ok_or(PcuSpirvError::IdSpaceExhausted)
}

fn dataflow_op_ptr_base(kernel: &PcuDispatchKernelIr<'_>) -> Result<u32, PcuSpirvError> {
    dataflow_value_base(kernel)?
        .checked_add(u32::from(max_dataflow_value_id(kernel)))
        .ok_or(PcuSpirvError::IdSpaceExhausted)
}

fn dataflow_op_ptr_id(
    kernel: &PcuDispatchKernelIr<'_>,
    op_index: usize,
) -> Result<u32, PcuSpirvError> {
    let op_index = u32::try_from(op_index).map_err(|_| PcuSpirvError::IdSpaceExhausted)?;
    dataflow_op_ptr_base(kernel)?
        .checked_add(op_index)
        .ok_or(PcuSpirvError::IdSpaceExhausted)
}

fn dataflow_binding_ordinal(
    kernel: &PcuDispatchKernelIr<'_>,
    binding: PcuBindingRef,
) -> Result<usize, PcuSpirvError> {
    kernel
        .bindings
        .iter()
        .position(|candidate| {
            candidate.set == binding.set
                && candidate.binding == binding.binding
                && matches!(
                    candidate.access,
                    PcuBindingAccess::ReadOnly
                        | PcuBindingAccess::WriteOnly
                        | PcuBindingAccess::ReadWrite
                )
        })
        .ok_or(PcuSpirvError::InvalidBinding)
}

fn dataflow_index_id(
    kernel: &PcuDispatchKernelIr<'_>,
    index: PcuDispatchIndex,
) -> Result<u32, PcuSpirvError> {
    match index {
        PcuDispatchIndex::InvocationId => Ok(DATAFLOW_INDEX_ID),
        PcuDispatchIndex::GridStrideId => Ok(dataflow_loop_ids(kernel)?.index),
        PcuDispatchIndex::Value(_) => Err(PcuSpirvError::UnsupportedInstruction(
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index,
            })
            .support_flag(),
        )),
    }
}

fn max_dataflow_value_id(kernel: &PcuDispatchKernelIr<'_>) -> u16 {
    let mut max_value = 0;
    for op in kernel.ops.iter().copied() {
        let mut record = |op| match op {
            PcuDispatchOp::Data(
                PcuDispatchDataOp::BindingLoad { result, .. }
                | PcuDispatchDataOp::Constant { result, .. }
                | PcuDispatchDataOp::Alu { result, .. },
            ) => {
                max_value = max_value.max(result.0);
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { value, .. }) => {
                max_value = max_value.max(value.0);
            }
            _ => {}
        };
        record(op);
        if let PcuDispatchOp::GridStrideLoop { body, .. } = op {
            for body_op in body.iter().copied() {
                record(body_op);
            }
        }
    }
    max_value
}

pub(crate) const fn literal_string_word_count(value: &str) -> usize {
    (value.len() + 1).div_ceil(4)
}
