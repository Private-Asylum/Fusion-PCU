//! Cold native integer/bit arithmetic emission; all checked effects get status columns.
use std::fmt::Write;
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchDataOp as Data,
    PcuDispatchFloatBinaryOp as FloatOp,
    PcuDispatchIntegerBinaryOp as IntegerOp,
    PcuDispatchIndex,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuParameterValue,
    PcuRangePolicy,
    PcuScalarType,
    PcuValueType,
};
use super::MetalCheckedMapPlan;
use crate::MetalError;

pub(super) fn lower(plan: &MetalCheckedMapPlan) -> Result<String, MetalError> {
    let PcuValueType::Scalar(scalar) = plan.value_type() else {
        return Err(MetalError::Unsupported);
    };
    let mut source = header(scalar)?;
    source.push_str("\nkernel void pcu_checked_composed(device uchar* r0 [[buffer(0)]],device uchar* r1 [[buffer(1)]],device uchar* r2 [[buffer(2)]],device uchar* r3 [[buffer(3)]],device uint* records [[buffer(4)]],constant uint2& profile [[buffer(5)]],uint id [[thread_position_in_grid]]) { if(id>=profile.x)return; bool lane_failed=false;\n");
    let mut stored: [Option<PcuDispatchValueId>; 4] = [None; 4];
    let mut effect = 0;
    for instruction in plan.instructions() {
        match *instruction {
            Data::BindingLoad {
                result,
                binding,
                index,
            } => {
                let slot = plan
                    .resources()
                    .iter()
                    .position(|r| r.binding == binding)
                    .ok_or(MetalError::Unsupported)?;
                if let Some(prior) = stored[slot] {
                    copy_value(&mut source, scalar, result, prior)?;
                } else {
                    load(&mut source, scalar, result, slot, index)?;
                }
            }
            Data::BindingStore { binding, value, .. } => {
                let slot = plan
                    .resources()
                    .iter()
                    .position(|r| r.binding == binding)
                    .ok_or(MetalError::Unsupported)?;
                stored[slot] = Some(value);
            }
            Data::Constant { result, value } => constant(&mut source, scalar, result, value)?,
            Data::CheckedIntegerBinary {
                result,
                lhs,
                rhs,
                op,
                range_policy,
                ..
            } => {
                integer(&mut source, result, lhs, rhs, op, range_policy)?;
                status(&mut source, effect)?;
                effect += 1;
            }
            Data::CheckedFloatBinary {
                result,
                lhs,
                rhs,
                op,
                range_policy,
                underflow_policy,
                ..
            } => {
                floating(
                    &mut source,
                    scalar,
                    result,
                    lhs,
                    rhs,
                    op,
                    range_policy,
                    underflow_policy,
                )?;
                status(&mut source, effect)?;
                effect += 1;
            }
            _ => return Err(MetalError::Unsupported),
        }
    }
    for (slot, value) in stored.into_iter().enumerate() {
        if let Some(value) = value {
            store(&mut source, scalar, value, slot)?;
        }
    }
    source.push_str("}\n");
    Ok(source)
}

pub fn header(scalar: PcuScalarType) -> Result<String, MetalError> {
    if matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64) {
        let (source, kernel) = if scalar == PcuScalarType::F32 {
            (
                include_str!("../../ffi/native/cpp/checked_f32/checked_f32.metal"),
                "kernel void pcu_checked_f32",
            )
        } else {
            (
                include_str!("../../ffi/native/cpp/checked_f64/checked_f64.metal"),
                "kernel void pcu_checked_f64",
            )
        };
        return source
            .split_once(kernel)
            .map(|(header, _)| header.to_owned())
            .ok_or(MetalError::Unsupported);
    }
    let bits = scalar.bit_width();
    let signed = matches!(
        scalar,
        PcuScalarType::I8
            | PcuScalarType::I16
            | PcuScalarType::I32
            | PcuScalarType::I64
            | PcuScalarType::I128
    );
    let mask = if bits < 32 {
        (1_u32 << bits) - 1
    } else {
        u32::MAX
    };
    let sign = 1_u32 << if bits < 32 { bits - 1 } else { 31 };
    let primitive = include_str!("../../runtime/integer/shader/checked_integer.metal");
    let (header, kernel) = primitive
        .split_once("kernel void pcu_checked_integer")
        .ok_or(MetalError::Unsupported)?;
    let begin = kernel.find("  bool an=").ok_or(MetalError::Unsupported)?;
    let end = kernel
        .find("  store_value(output,id,result);")
        .ok_or(MetalError::Unsupported)?;
    let arithmetic = kernel[begin..end]
        .replace("uint operation=profile.y&0xffu,fault=0u;", "uint fault=0u;")
        .replace("  bool clamp=(profile.y&(1u<<8u))!=0u;", "");
    Ok(format!(
        "#define PCU_WIDTH {bits}u\n#define PCU_SIGNED {}u\n#define PCU_MASK {mask}u\n#define PCU_SIGN {sign}u\n{header}\nuint pcu_integer(thread const uint* a,thread const uint* b,thread uint* result,uint operation,bool clamp){{\n{arithmetic}\nreturn fault==0u?0u:(fault|(clamp?0x100u:0u)); }}\n",
        u32::from(signed)
    ))
}
fn scalar_name(scalar: PcuScalarType) -> &'static str {
    if scalar == PcuScalarType::F64 {
        "ulong"
    } else {
        "uint"
    }
}
const fn is_float(scalar: PcuScalarType) -> bool {
    matches!(scalar, PcuScalarType::F32 | PcuScalarType::F64)
}
fn load(
    source: &mut String,
    scalar: PcuScalarType,
    result: PcuDispatchValueId,
    slot: usize,
    index: PcuDispatchIndex,
) -> Result<(), MetalError> {
    let index = if index == PcuDispatchIndex::BindingElementZero {
        "0u"
    } else {
        "id"
    };
    if is_float(scalar) {
        writeln!(
            source,
            "{} v{}=reinterpret_cast<device const {}*>(r{slot})[{index}];",
            scalar_name(scalar),
            result.0,
            scalar_name(scalar)
        )
    } else {
        writeln!(
            source,
            "uint v{}[LIMBS];load_value(r{slot},{index},v{});",
            result.0, result.0
        )
    }
    .map_err(|_| MetalError::Unsupported)
}
fn store(
    source: &mut String,
    scalar: PcuScalarType,
    value: PcuDispatchValueId,
    slot: usize,
) -> Result<(), MetalError> {
    if is_float(scalar) {
        writeln!(
            source,
            "reinterpret_cast<device {}*>(r{slot})[id]=v{};",
            scalar_name(scalar),
            value.0
        )
    } else {
        writeln!(source, "store_value(r{slot},id,v{});", value.0)
    }
    .map_err(|_| MetalError::Unsupported)
}
fn copy_value(
    source: &mut String,
    scalar: PcuScalarType,
    result: PcuDispatchValueId,
    prior: PcuDispatchValueId,
) -> Result<(), MetalError> {
    if is_float(scalar) {
        writeln!(
            source,
            "{} v{}=v{};",
            scalar_name(scalar),
            result.0,
            prior.0
        )
    } else {
        writeln!(
            source,
            "uint v{}[LIMBS];for(uint j=0;j<LIMBS;++j)v{}[j]=v{}[j];",
            result.0, result.0, prior.0
        )
    }
    .map_err(|_| MetalError::Unsupported)
}
fn constant(
    source: &mut String,
    scalar: PcuScalarType,
    result: PcuDispatchValueId,
    value: PcuParameterValue,
) -> Result<(), MetalError> {
    match (scalar, value) {
        (PcuScalarType::F32, PcuParameterValue::F32(bits)) => {
            writeln!(source, "uint v{}=0x{bits:x}u;", result.0).map_err(|_| MetalError::Unsupported)
        }
        (PcuScalarType::F64, PcuParameterValue::F64(bits)) => {
            writeln!(source, "ulong v{}=0x{bits:x}ul;", result.0)
                .map_err(|_| MetalError::Unsupported)
        }
        _ => Err(MetalError::Unsupported),
    }
}
fn integer(
    source: &mut String,
    result: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
    op: IntegerOp,
    range: PcuRangePolicy,
) -> Result<(), MetalError> {
    let code = match op {
        IntegerOp::Add => 0,
        IntegerOp::Sub => 1,
        IntegerOp::Mul => 2,
    };
    writeln!(source,"uint v{}[LIMBS];for(uint j=0;j<LIMBS;++j)v{}[j]=0u;{{uint status=0u;if(!lane_failed)status=pcu_integer(v{},v{},v{},{code}u,{});",result.0,result.0,lhs.0,rhs.0,result.0,range==PcuRangePolicy::Clamp).map_err(|_|MetalError::Unsupported)
}
fn status(source: &mut String, effect: usize) -> Result<(), MetalError> {
    writeln!(source,"records[{effect}u*profile.x+id]=status;lane_failed=lane_failed||(status!=0u&&(status&0x100u)==0u);}}") .map_err(|_|MetalError::Unsupported)
}
#[allow(clippy::too_many_arguments)] // Every frozen local opcode/operand/range/UF is independent.
fn floating(
    source: &mut String,
    scalar: PcuScalarType,
    result: PcuDispatchValueId,
    lhs: PcuDispatchValueId,
    rhs: PcuDispatchValueId,
    op: FloatOp,
    range: PcuRangePolicy,
    uf: PcuFloatUnderflowPolicy,
) -> Result<(), MetalError> {
    let policy = match uf {
        PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
        PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
    };
    let name = scalar_name(scalar);
    let (mask, exponent) = if scalar == PcuScalarType::F32 {
        ("255u", 23)
    } else {
        ("2047ul", 52)
    };
    writeln!(source,"{name} v{}=0;{{uint status=0u;if(!lane_failed){{Result step;if(((v{}>>{exponent})&{mask})=={mask}||((v{}>>{exponent})&{mask})=={mask})step=fault({}u);else{{",result.0,lhs.0,rhs.0,if scalar==PcuScalarType::F32 {1}else{4}).map_err(|_|MetalError::Unsupported)?;
    let function = match op {
        FloatOp::Add | FloatOp::Sub => "add",
        FloatOp::Mul => "multiply",
        FloatOp::Div => "divide",
    };
    if scalar == PcuScalarType::F32 {
        writeln!(
            source,
            "Binary math={{{policy}u,{}}};step=math.{function}(v{},v{}{});",
            range == PcuRangePolicy::Clamp,
            lhs.0,
            rhs.0,
            if matches!(op, FloatOp::Add | FloatOp::Sub) {
                if op == FloatOp::Sub {
                    ",true"
                } else {
                    ",false"
                }
            } else {
                ""
            }
        )
        .map_err(|_| MetalError::Unsupported)?;
    } else {
        writeln!(
            source,
            "step={function}(v{},v{}{},{}u);",
            lhs.0,
            rhs.0,
            if matches!(op, FloatOp::Add | FloatOp::Sub) {
                if op == FloatOp::Sub {
                    ",true"
                } else {
                    ",false"
                }
            } else {
                ""
            },
            policy
                | if range == PcuRangePolicy::Clamp {
                    0x100
                } else {
                    0
                }
        )
        .map_err(|_| MetalError::Unsupported)?;
    }
    writeln!(source, "}}v{}=step.bits;", result.0).map_err(|_| MetalError::Unsupported)?;
    if scalar == PcuScalarType::F32 {
        source.push_str("uint kind=step.status&0xffu;status=(kind==1u?4u:kind==2u?3u:kind==3u?1u:kind==4u?2u:0u)|(step.status&0x100u);}");
    } else {
        source.push_str("status=step.status;}");
    }
    Ok(())
}
