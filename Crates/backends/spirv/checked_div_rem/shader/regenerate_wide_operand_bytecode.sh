#!/bin/sh
# Offline maintenance; integer operand indices are frozen specialization constants.
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT HUP INT TERM
glslangValidator -V --target-env vulkan1.0 -Os --vn PCU_DIV_REM_ROLES -o "$temporary/words.h" "$base/wide_operand_div_rem.comp"
glslangValidator -V --target-env vulkan1.0 -Os -o "$temporary/words.spv" "$base/wide_operand_div_rem.comp"
spirv-val --target-env vulkan1.0 "$temporary/words.spv"
{
 printf '%s\n' '//! Generated offline from the auditable U32 operand-role division shader.' '#[rustfmt::skip]' '#[allow(clippy::unreadable_literal)] // Exact compiler-produced SPIR-V words.' 'pub(super) const WORDS: &[u32] = &['
 sed -n '/^[[:space:]]*0x/p' "$temporary/words.h"
 printf '%s\n' '];'
} > "$base/../wide_operand_bytecode/wide_operand_bytecode.rs"
