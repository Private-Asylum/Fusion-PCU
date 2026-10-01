#!/usr/bin/env python3
"""Create a labeled safety-patched official-C comparison copy; never edit input."""
import argparse
import hashlib
import json
import pathlib
import shutil

p = argparse.ArgumentParser()
p.add_argument('--source', required=True, type=pathlib.Path)
p.add_argument('--output', required=True, type=pathlib.Path)
a = p.parse_args()
src, dst = a.source.resolve(), a.output.resolve()
if dst.exists() or src == dst or src in dst.parents or dst in src.parents:
    raise SystemExit('output must be a new directory separate from source')
shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.git', 'build'))
changes = []
for path in sorted((dst / 'mlx/c').rglob('*.cpp')):
    before = path.read_bytes()
    text = before.decode()
    count = text.count('mlx_error(e.what());')
    if count:
        text = text.replace('mlx_error(e.what());', 'mlx_error("%s", e.what());')
    if path.name == 'error.cpp':
        start = text.index('extern "C" void\n_mlx_error(')
        text = text[:start] + '''// EVALUATION ONLY: fixed stack storage, no allocation in error reporting.
extern "C" void
_mlx_error(const char* file, const int line, const char* fmt, ...) {
  char msg[2048] = {};
  va_list args;
  va_start(args, fmt);
  const int written = vsnprintf(msg, sizeof(msg), fmt ? fmt : "MLX error", args);
  va_end(args);
  if (written < 0) {
    snprintf(msg, sizeof(msg), "%s", "MLX error formatting failed");
  } else {
    const size_t used = static_cast<size_t>(written) < sizeof(msg)
        ? static_cast<size_t>(written) : sizeof(msg) - 1;
    if (used < sizeof(msg) - 1) {
      snprintf(msg + used, sizeof(msg) - used, " at %s:%d", file ? file : "?", line);
    }
  }
  msg[sizeof(msg) - 1] = '\\0';
  // Owned comparison image installs one permanent noexcept handler, no payload.
  mlx_error_handler_(msg, mlx_error_handler_data_.get());
}
'''
    after = text.encode()
    if before != after:
        path.write_bytes(after)
        changes.append({'path': str(path.relative_to(dst)), 'literal_forwarding_replacements': count,
                        'before_sha256': hashlib.sha256(before).hexdigest(),
                        'after_sha256': hashlib.sha256(after).hexdigest()})
# Compile only the unchanged official matmul endpoint, not unrelated operators
# whose signatures drifted between the Brew generator and pinned SDK 0.32.3.
ops = (dst / 'mlx/c/ops.cpp').read_text()
start = ops.index('extern "C" int mlx_matmul(')
end = ops.index('extern "C"', start + 1)
selected = ops[:ops.index('extern "C"')] + ops[start:end]
selected_path = dst / 'mlx/c/evaluation_matmul.cpp'
selected_path.write_text(selected)
changes.append({'path': 'mlx/c/evaluation_matmul.cpp',
                'kind': 'exact selected endpoint with literal-safe forwarding',
                'origin': 'mlx/c/ops.cpp::mlx_matmul',
                'after_sha256': hashlib.sha256(selected.encode()).hexdigest()})
(dst / 'safety-patch-provenance.json').write_text(json.dumps({
    'label': 'safety-patched official C; standalone private comparison; not unmodified mlx-c',
    'source': str(src), 'source_original_unchanged': True,
    'patches': changes,
    'limitations': ['sole permanent handler in private evaluation image',
                    'calling-thread containment only; worker failures remain SDK scope',
                    'no independent-consumer production acceptance claim'],
}, indent=2) + '\n')
print(dst)
