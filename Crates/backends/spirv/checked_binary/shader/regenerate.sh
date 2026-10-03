#!/bin/sh
# Offline maintenance only: no shader compiler is a crate/runtime dependency.
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT HUP INT TERM
for scalar in f32 f64 low; do
    if [ "$scalar" = f32 ]; then
        shader=checked_binary.comp
        target=bytecode.rs
    elif [ "$scalar" = f64 ]; then
        shader=checked_binary_f64.comp
        target=f64.rs
    else
        shader=checked_binary_low.comp
        target=low.rs
    fi
    glslangValidator -V --target-env vulkan1.0 -Os --vn PCU_CHECKED_BINARY -o "$temporary/words.h" "$base/$shader"
    glslangValidator -V --target-env vulkan1.0 -Os -o "$temporary/words.spv" "$base/$shader"
    spirv-val --target-env vulkan1.0 "$temporary/words.spv"
    {
        printf '%s\n' '//! Generated offline from the auditable GLSL shader; see its regeneration script.' '#[rustfmt::skip]' '#[allow(clippy::unreadable_literal)] // Compiler-produced SPIR-V word encoding.' 'pub(super) const WORDS: &[u32] = &['
        sed -n '/^[[:space:]]*0x/p' "$temporary/words.h"
        printf '%s\n' '];'
    } > "$base/../bytecode/$target"
done
