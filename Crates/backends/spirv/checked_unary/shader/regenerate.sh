#!/bin/sh
# Offline maintenance only; no runtime compiler dependency.
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT HUP INT TERM
glslangValidator -V --target-env vulkan1.0 -Os --vn PCU_CHECKED_UNARY -o "$temporary/words.h" "$base/checked_unary.comp"
glslangValidator -V --target-env vulkan1.0 -Os -o "$temporary/words.spv" "$base/checked_unary.comp"
spirv-val --target-env vulkan1.0 "$temporary/words.spv"
{
    printf '%s\n' '//! Generated offline from the auditable GLSL unary shader.' '#[rustfmt::skip]' '#[allow(clippy::unreadable_literal)] // Compiler-produced SPIR-V words.' 'pub(super) const WORDS: &[u32] = &['
    sed -n '/^[[:space:]]*0x/p' "$temporary/words.h"
    printf '%s\n' '];'
} > "$base/../bytecode/bytecode.rs"
