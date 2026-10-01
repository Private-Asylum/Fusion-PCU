// ABI 3 diagnostic only: safety-patched official mlx-c, pinned MLX 0.32.3.
// Private image owns one permanent handler. No production coexistence claim.
// All tensor operations, arrays and compiled closures use the official C API.
#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/compile.h"
#include "mlx/c/device.h"
#include "mlx/c/error.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"
#include "mlx/c/version.h"
#include <mlx/version.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

static_assert(MLX_VERSION_NUMERIC == 32003, "evaluation requires MLX 0.32.3 headers");
#define EVAL_EXPORT extern "C" __attribute__((visibility("default")))
namespace {
constexpr size_t capacity = 2048;
thread_local char reported[capacity] = {};
thread_local bool release_failed = false;
void text(char* dst, const char* src) noexcept {
    if (!dst) return;
    if (!src) src = "MLX evaluation error";
    const auto n = std::min(std::strlen(src), capacity - 1);
    std::memcpy(dst, src, n);
    dst[n] = '\0';
}
void handler(const char* message, void*) noexcept { text(reported, message); }

// Every call to the official foreign API passes this barrier, including calls
// made from Rust-facing callbacks and constructors. Never foreign-unwind to Rust.
template <class F> auto foreign(F&& call) {
    reported[0] = '\0';
    try {
        auto value = call();
        if (reported[0]) throw std::runtime_error(reported);
        return value;
    } catch (const std::exception& e) {
        text(reported, e.what());
        throw std::runtime_error(reported);
    } catch (...) {
        text(reported, "unknown C++ exception from official C API");
        throw std::runtime_error(reported);
    }
}
template <class F> void status(F&& call) {
    if (foreign(std::forward<F>(call)) != 0)
        throw std::runtime_error("official C API returned an error without text");
}
// Destructors never throw or retry uncertain destruction. ABI status 2 tells
// Rust to retain the loaded library; it does not assert destruction succeeded.
template <class F> void cleanup(F&& call) noexcept {
    try {
        if (foreign(std::forward<F>(call)) != 0) release_failed = true;
    } catch (...) { release_failed = true; }
}
template <class H, int (*Free)(H)> struct Handle {
    H value{};
    Handle() = default;
    explicit Handle(H h) : value(h) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() noexcept {
        if (value.ctx) cleanup([&] { return Free(value); });
    }
};
using AHandle = Handle<mlx_array, mlx_array_free>;
using VHandle = Handle<mlx_vector_array, mlx_vector_array_free>;
using CHandle = Handle<mlx_closure, mlx_closure_free>;
using DHandle = Handle<mlx_device, mlx_device_free>;
using SHandle = Handle<mlx_stream, mlx_stream_free>;
using THandle = Handle<mlx_string, mlx_string_free>;

size_t count(int32_t rows, int32_t columns) {
    if (rows <= 0 || columns <= 0) throw std::invalid_argument("positive matrix dimensions required");
    const auto r = static_cast<size_t>(rows), c = static_cast<size_t>(columns);
    if (r > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / c / sizeof(float))
        throw std::invalid_argument("matrix extent overflow");
    return r * c;
}
void initialized() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        (void)foreign([] { mlx_set_error_handler(handler, nullptr, nullptr); return 0; });
    });
}
void require_version() {
    initialized();
    THandle str;
    status([&] { return mlx_version(&str.value); });
    const char* version = foreign([&] { return mlx_string_data(str.value); });
    if (!version || std::strcmp(version, "0.32.3"))
        throw std::runtime_error("evaluation requires MLX 0.32.3 runtime");
}
struct Session {
    DHandle device;
    SHandle stream;
    std::thread::id thread = std::this_thread::get_id();
    bool poisoned = false;
    explicit Session(int32_t index) {
        int devices = 0;
        status([&] { return mlx_device_count(&devices, MLX_GPU); });
        if (index < 0 || index >= devices) throw std::invalid_argument("GPU index unavailable");
        device.value = foreign([&] { return mlx_device_new_type(MLX_GPU, index); });
        if (!device.value.ctx) throw std::runtime_error("device constructor returned empty handle");
        bool available = false;
        status([&] { return mlx_device_is_available(&available, device.value); });
        if (!available) throw std::invalid_argument("GPU unavailable");
        stream.value = foreign([&] { return mlx_stream_new_device(device.value); });
        if (!stream.value.ctx) throw std::runtime_error("stream constructor returned empty handle");
    }
    void validate() const {
        if (thread != std::this_thread::get_id()) throw std::runtime_error("session used from another thread");
        if (poisoned) throw std::runtime_error("session quarantined after unknown completion");
    }
};
using SessionOwner = std::shared_ptr<Session>;
struct Array {
    SessionOwner owner;
    AHandle array;
    int32_t rows, columns;
    Array(SessionOwner s, int32_t r, int32_t c) : owner(std::move(s)), rows(r), columns(c) {}
};
struct TraceState {
    SessionOwner owner;
    size_t traces = 0;
    explicit TraceState(SessionOwner s) : owner(std::move(s)) {}
};
struct Prepared {
    SessionOwner owner;
    std::shared_ptr<TraceState> trace;
    CHandle compiled;
    int32_t rows, inner, columns;
    Prepared(SessionOwner s, int32_t r, int32_t i, int32_t c)
        : owner(std::move(s)), trace(std::make_shared<TraceState>(owner)), rows(r), inner(i), columns(c) {}
};
using PreparedOwner = std::shared_ptr<Prepared>;
struct Pending {
    SessionOwner owner;
    PreparedOwner prepared;
    AHandle left, right;
    std::unique_ptr<Array> output;
    explicit Pending(SessionOwner s) : owner(std::move(s)) {}
};
void shape(mlx_array array, int32_t rows, int32_t columns) {
    if (!array.ctx || foreign([&] { return mlx_array_dtype(array); }) != MLX_FLOAT32
        || foreign([&] { return mlx_array_ndim(array); }) != 2
        || foreign([&] { return mlx_array_dim(array, 0); }) != rows
        || foreign([&] { return mlx_array_dim(array, 1); }) != columns)
        throw std::invalid_argument("exact F32 matrix shape mismatch");
}
void copy(AHandle& dst, mlx_array src) { status([&] { return mlx_array_set(&dst.value, src); }); }
void host(AHandle& dst, const float* data, int32_t rows, int32_t columns) {
    const int dimensions[2] = {rows, columns};
    status([&] { return mlx_array_set_data(&dst.value, data, dimensions, 2, MLX_FLOAT32); });
}
void terminal(Pending& pending) {
    status([&] { return mlx_array_eval(pending.output->array.value); });
    status([&] { return mlx_synchronize(pending.owner->stream.value); });
    status([&] { return _mlx_array_wait(pending.output->array.value); });
}
// Handler capture is nested/reentrant on one thread: callbacks may invoke further
// official calls; every enclosing call also checks the actual status return.
int trace_callback(mlx_vector_array* result, const mlx_vector_array inputs, void* payload) noexcept {
    try {
        auto& trace = **static_cast<std::shared_ptr<TraceState>*>(payload);
        trace.owner->validate();
        ++trace.traces;
        if (foreign([&] { return mlx_vector_array_size(inputs); }) != 2)
            throw std::invalid_argument("compiled callback requires two inputs");
        AHandle a, b, output;
        status([&] { return mlx_vector_array_get(&a.value, inputs, 0); });
        status([&] { return mlx_vector_array_get(&b.value, inputs, 1); });
        status([&] { return mlx_matmul(&output.value, a.value, b.value, trace.owner->stream.value); });
        status([&] { return mlx_vector_array_set_value(result, output.value); });
        return release_failed ? 1 : 0;
    } catch (const std::exception& e) { text(reported, e.what()); }
      catch (...) { text(reported, "unknown exception in compiled callback"); }
    return 1;
}
void trace_payload_free(void* payload) noexcept {
    try { delete static_cast<std::shared_ptr<TraceState>*>(payload); }
    catch (...) { release_failed = true; }
}
void apply(Prepared& compiled, mlx_array a, mlx_array b, AHandle& output) {
    VHandle inputs, outputs;
    const mlx_array pair[2] = {a, b};
    status([&] { return mlx_vector_array_set_data(&inputs.value, pair, 2); });
    status([&] { return mlx_closure_apply(&outputs.value, compiled.compiled.value, inputs.value); });
    if (foreign([&] { return mlx_vector_array_size(outputs.value); }) != 1)
        throw std::runtime_error("compiled closure output count mismatch");
    status([&] { return mlx_vector_array_get(&output.value, outputs.value, 0); });
    shape(output.value, compiled.rows, compiled.columns);
}
template <class F> int checked(char* error, F&& call) noexcept {
    release_failed = false;
    if (error) error[0] = '\0';
    int result = 0;
    try { call(); }
    catch (const std::exception& e) { text(error, e.what()); result = 1; }
    catch (...) { text(error, "unknown C++ exception in evaluation bridge"); result = 1; }
    if (release_failed) {
        text(error, "official C destruction failed; raw handle and library quarantined");
        return 2;
    }
    return result;
}
SessionOwner session(void* raw) {
    if (!raw) throw std::invalid_argument("null session");
    auto owner = *static_cast<SessionOwner*>(raw);
    owner->validate();
    return owner;
}
Array& matrix(void* raw, const SessionOwner& owner) {
    if (!raw) throw std::invalid_argument("null matrix");
    auto& result = *static_cast<Array*>(raw);
    if (result.owner != owner) throw std::invalid_argument("matrix belongs to another session");
    shape(result.array.value, result.rows, result.columns);
    return result;
}
template <class F> int execute(char* error, void** result, F&& graph) noexcept {
    std::unique_ptr<Pending> pending;
    int code = checked(error, [&] {
        if (!result) throw std::invalid_argument("null result slot");
        *result = nullptr;
        graph(pending);
        terminal(*pending);
        auto output = std::move(pending->output);
        pending.reset();
        if (release_failed) throw std::runtime_error("input cleanup failed");
        *result = output.release();
    });
    if (code && pending) {
        const int synchronized = checked(nullptr, [&] {
            status([&] { return mlx_synchronize(pending->owner->stream.value); });
        });
        if (synchronized) {
            pending->owner->poisoned = true;
            (void)pending.release();
            text(error, "completion unknown; input/output/compiled/session owners quarantined");
            code = 2;
        } else {
            pending.reset();
            if (release_failed) { text(error, "cleanup failed; library quarantined"); code = 2; }
        }
    }
    return code;
}
} // namespace

EVAL_EXPORT uint32_t eval_abi() noexcept { return 3; }
EVAL_EXPORT int eval_version(char* version, char* error) noexcept {
    return checked(error, [&] {
        require_version();
        if (!version) throw std::invalid_argument("null version slot");
        std::memcpy(version, "0.32.3", 7);
    });
}
EVAL_EXPORT int eval_session_new(int32_t index, void** result, char* error) noexcept {
    return checked(error, [&] {
        if (!result) throw std::invalid_argument("null result slot");
        *result = nullptr;
        require_version();
        *result = new SessionOwner(std::make_shared<Session>(index));
    });
}
EVAL_EXPORT int eval_session_free(void* value, char* error) noexcept {
    return checked(error, [&] { delete static_cast<SessionOwner*>(value); });
}
EVAL_EXPORT int eval_array_new(void* raw, const float* data, size_t length,
    int32_t rows, int32_t columns, void** result, char* error) noexcept {
    return checked(error, [&] {
        if (!result) throw std::invalid_argument("null result slot");
        *result = nullptr;
        auto owner = session(raw);
        if (!data || count(rows, columns) != length) throw std::invalid_argument("host matrix extent mismatch");
        auto array = std::make_unique<Array>(owner, rows, columns);
        host(array->array, data, rows, columns);
        *result = array.release();
    });
}
EVAL_EXPORT int eval_array_free(void* value, char* error) noexcept {
    return checked(error, [&] { delete static_cast<Array*>(value); });
}
EVAL_EXPORT int eval_read(void* raw, float* dst, size_t length, char* error) noexcept {
    std::unique_ptr<Array> retained;
    bool materializing = false;
    const int code = checked(error, [&] {
        if (!raw || !dst) throw std::invalid_argument("null readback argument");
        auto& array = *static_cast<Array*>(raw);
        array.owner->validate();
        shape(array.array.value, array.rows, array.columns);
        if (count(array.rows, array.columns) != length) throw std::invalid_argument("readback extent mismatch");
        bool available = false, contiguous = false;
        status([&] { return _mlx_array_is_available(&available, array.array.value); });
        status([&] { return _mlx_array_is_row_contiguous(&contiguous, array.array.value); });
        if (!available || !contiguous) throw std::runtime_error("terminal row-contiguous array required");
        retained = std::make_unique<Array>(array.owner, array.rows, array.columns);
        copy(retained->array, array.array.value);
        materializing = true;
        const float* source = foreign([&] { return mlx_array_data_float32(retained->array.value); });
        if (!source) throw std::runtime_error("CPU-visible data unavailable");
        // mlx_array_set copies the C++ array descriptor (shared backing), so the
        // original array keeps this pointer alive after releasing the extra copy.
        retained.reset();
        if (release_failed) throw std::runtime_error("readback owner cleanup failed before publication");
        std::memcpy(dst, source, length * sizeof(float));
    });
    if (code && materializing && retained) {
        // Conservatively retain actual backing if SDK host materialization fails.
        retained->owner->poisoned = true;
        (void)retained.release();
        text(error, "host materialization failed; backing/session/library quarantined");
        return 2;
    }
    return code;
}
EVAL_EXPORT int eval_matmul(void* raw, void* a, void* b, void** result, char* error) noexcept {
    return execute(error, result, [&](auto& pending) {
        auto owner = session(raw);
        auto& left = matrix(a, owner);
        auto& right = matrix(b, owner);
        if (left.columns != right.rows) throw std::invalid_argument("matmul inner dimensions mismatch");
        (void)count(left.rows, right.columns);
        pending = std::make_unique<Pending>(owner);
        copy(pending->left, left.array.value);
        copy(pending->right, right.array.value);
        pending->output = std::make_unique<Array>(owner, left.rows, right.columns);
        status([&] { return mlx_matmul(&pending->output->array.value,
            pending->left.value, pending->right.value, owner->stream.value); });
    });
}
EVAL_EXPORT int eval_prepare(void* raw, int32_t rows, int32_t inner, int32_t columns,
    void** result, char* error) noexcept {
    return checked(error, [&] {
        if (!result) throw std::invalid_argument("null result slot");
        *result = nullptr;
        auto owner = session(raw);
        const auto left_count = count(rows, inner), right_count = count(inner, columns);
        (void)count(rows, columns);
        auto compiled = std::make_shared<Prepared>(owner, rows, inner, columns);
        CHandle function;
        // Constructor takes payload ownership on entry, including its allocation-failure path.
        auto* payload = new std::shared_ptr<TraceState>(compiled->trace);
        function.value = foreign([&] {
            return mlx_closure_new_func_payload(trace_callback, payload, trace_payload_free);
        });
        if (!function.value.ctx) throw std::runtime_error("callback constructor returned empty handle");
        status([&] { return mlx_compile(&compiled->compiled.value, function.value, false); });
        // Official C lacks a logical empty descriptor constructor. These initialized
        // MLX-owned host allocations are outside measurement; no evaluation/GPU launch.
        std::vector<float> left_data(left_count, 0.0f), right_data(right_count, 0.0f);
        AHandle left, right, output;
        host(left, left_data.data(), rows, inner);
        host(right, right_data.data(), inner, columns);
        apply(*compiled, left.value, right.value, output);
        if (compiled->trace->traces != 1) throw std::runtime_error("expected exactly one cold trace");
        *result = new PreparedOwner(std::move(compiled));
    });
}
EVAL_EXPORT int eval_compiled(void* raw, void* a, void* b, void** result, char* error) noexcept {
    return execute(error, result, [&](auto& pending) {
        if (!raw) throw std::invalid_argument("null compiled owner");
        auto compiled = *static_cast<PreparedOwner*>(raw);
        auto owner = compiled->owner;
        owner->validate();
        auto& left = matrix(a, owner);
        auto& right = matrix(b, owner);
        if (left.rows != compiled->rows || left.columns != compiled->inner
            || right.rows != compiled->inner || right.columns != compiled->columns)
            throw std::invalid_argument("compiled exact shapes mismatch");
        pending = std::make_unique<Pending>(owner);
        pending->prepared = compiled;
        copy(pending->left, left.array.value);
        copy(pending->right, right.array.value);
        pending->output = std::make_unique<Array>(owner, compiled->rows, compiled->columns);
        apply(*compiled, pending->left.value, pending->right.value, pending->output->array);
        // This path intentionally uses official C's warm SDK compiler-cache lookup,
        // unlike production retained-primitive replay. Ambient cache keys may retrace.
    });
}
EVAL_EXPORT int eval_prepared_free(void* value, char* error) noexcept {
    return checked(error, [&] { delete static_cast<PreparedOwner*>(value); });
}
EVAL_EXPORT size_t eval_traces(void* value) noexcept {
    try {
        if (!value) return 0;
        return (*static_cast<PreparedOwner*>(value))->trace->traces;
    } catch (...) { return 0; }
}
