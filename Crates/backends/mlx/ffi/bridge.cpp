// Private C ABI containing calling-thread C++ exceptions over pinned public MLX APIs.
// Upstream worker-thread exceptions require separate SDK containment; see README.
// No Objective-C ABI, mlx-c error-handler mutation, global default-stream mutation,
// borrowed/no-copy backing import, or CPU tensor arithmetic is used here.
#include <mlx/array.h>
#include <mlx/compile.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/version.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static_assert(MLX_VERSION_NUMERIC == 32003, "PCU bridge requires MLX0.32.3 headers");

#define PCU_EXPORT extern "C" __attribute__((visibility("default")))

namespace mx = mlx::core;
namespace {
constexpr size_t error_capacity = 2048;

void error_text(char* destination, const char* text) noexcept {
    if (!destination) return;
    const size_t length = std::min(std::strlen(text), error_capacity - 1);
    std::memcpy(destination, text, length);
    destination[length] = '\0';
}

template <typename F>
int checked(char* error, F&& operation) noexcept {
    try {
        operation();
        return 0;
    } catch (const std::exception& failure) {
        error_text(error, failure.what());
    } catch (...) {
        error_text(error, "MLX raised an unknown C++ exception");
    }
    return 1;
}

void require_version() {
    if (std::strcmp(mx::version(), "0.32.3") != 0)
        throw std::runtime_error("PCU bridge requires MLX0.32.3 runtime");
}

uint32_t gpu_backend() {
    if (!mx::metal::is_available())
        throw std::runtime_error("MLX requires an available Apple Metal GPU");
    return 1;
}

struct Session {
    mx::Device device;
    mx::Stream stream;
    std::thread::id thread;
    bool poisoned = false;

    explicit Session(int index)
        : device(mx::Device::gpu, index), stream(mx::new_stream(device)),
          thread(std::this_thread::get_id()) {}

    void validate() const {
        if (thread != std::this_thread::get_id())
            throw std::runtime_error("MLX session used from another thread");
        if (poisoned)
            throw std::runtime_error("MLX session quarantined after unknown completion");
    }
};

using SessionOwner = std::shared_ptr<Session>;
struct Array {
    SessionOwner owner;
    mx::array value;
    Array(SessionOwner session, mx::array array)
        : owner(std::move(session)), value(std::move(array)) {}
};

using ArrayFunction = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;
struct CompiledMatmul {
    SessionOwner owner;
    mx::Shape left_shape;
    mx::Shape right_shape;
    mx::Shape output_shape;
    std::shared_ptr<size_t> traces = std::make_shared<size_t>(0);
    // Retain the public compiler closure and its own cache lifetime. Warm calls never invoke
    // it: MLX0.32.3's wrapper keys cache lookup on mutable thread-local default-stream
    // state selected through the ambient default device, in addition to shape/dtype/constants.
    ArrayFunction compiled;
    ArrayFunction replay;
    CompiledMatmul(SessionOwner session, int32_t rows, int32_t inner, int32_t columns)
        : owner(std::move(session)), left_shape{rows, inner}, right_shape{inner, columns},
          output_shape{rows, columns} {}
};
using CompiledOwner = std::shared_ptr<CompiledMatmul>;

// Allocated before evaluation. Leaking this owner after failed synchronization retains
// actual array backings and the explicit stream/device; Rust also retains the loaded bridge.
struct Pending {
    SessionOwner session;
    CompiledOwner compiled;
    mx::array left;
    mx::array right;
    std::unique_ptr<Array> output;
    Pending(SessionOwner owner, const mx::array& a, const mx::array& b)
        : session(std::move(owner)), left(a), right(b) {}
};

size_t count(int32_t rows, int32_t columns) {
    if (rows <= 0 || columns <= 0)
        throw std::invalid_argument("MLX requires positive matrix dimensions");
    const auto a = static_cast<size_t>(rows);
    const auto b = static_cast<size_t>(columns);
    if (a > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(float) / b)
        throw std::invalid_argument("MLX matrix byte extent overflow");
    return a * b;
}

template <size_t N>
void copy_text(char (&destination)[N], const std::string& source) {
    if (source.size() >= N) throw std::runtime_error("MLX device fact exceeds ABI bound");
    std::memcpy(destination, source.c_str(), source.size() + 1);
}
} // namespace

struct PcuMlxFacts {
    uint32_t backend;
    char name[256];
    char architecture[128];
    // Preserve ABI2 layout without exposing additional device facts.
    char reserved[192];
};

PCU_EXPORT uint32_t pcu_mlx_bridge_abi() noexcept { return 2; }

PCU_EXPORT int pcu_mlx_version(char* version, char* error) noexcept {
    return checked(error, [&] {
        require_version();
        std::memcpy(version, "0.32.3", 7);
    });
}

PCU_EXPORT int pcu_mlx_gpu_count(int32_t* result, char* error) noexcept {
    return checked(error, [&] {
        require_version();
        (void)gpu_backend();
        *result = mx::device_count(mx::Device::gpu);
    });
}

PCU_EXPORT int pcu_mlx_gpu_facts(int32_t index, PcuMlxFacts* result, char* error) noexcept {
    return checked(error, [&] {
        require_version();
        *result = {};
        result->backend = gpu_backend();
        const mx::Device device(mx::Device::gpu, index);
        if (index < 0 || index >= mx::device_count(mx::Device::gpu) || !mx::is_available(device))
            throw std::invalid_argument("MLX GPU index unavailable");
        const auto& info = mx::device_info(device);
        const auto read = [&](const char* key, auto& destination) {
            const auto found = info.find(key);
            if (found != info.end()) {
                const auto* value = std::get_if<std::string>(&found->second);
                if (!value) throw std::runtime_error("MLX device fact type mismatch");
                copy_text(destination, *value);
            }
        };
        read("device_name", result->name);
        read("architecture", result->architecture);
    });
}

PCU_EXPORT int pcu_mlx_session_new(int32_t index, void** result, char* error) noexcept {
    return checked(error, [&] {
        require_version();
        (void)gpu_backend();
        const mx::Device device(mx::Device::gpu, index);
        if (index < 0 || index >= mx::device_count(mx::Device::gpu) || !mx::is_available(device))
            throw std::invalid_argument("MLX GPU index unavailable");
        *result = new SessionOwner(std::make_shared<Session>(index));
    });
}

PCU_EXPORT int pcu_mlx_session_free(void* session, char* error) noexcept {
    return checked(error, [&] { delete static_cast<SessionOwner*>(session); });
}

PCU_EXPORT int pcu_mlx_array_new(
    void* session, const float* data, size_t length, int32_t rows, int32_t columns,
    void** result, char* error) noexcept {
    return checked(error, [&] {
        const auto& owner = *static_cast<SessionOwner*>(session);
        owner->validate();
        if (!data || length != count(rows, columns))
            throw std::invalid_argument("MLX host matrix extent mismatch");
        // Iterator constructor calls init/copies to MLX-owned allocation immediately.
        // No pointer/deleter constructor is used and caller backing is never retained.
        *result = new Array(owner, mx::array(data, mx::Shape{rows, columns}, mx::float32));
    });
}

PCU_EXPORT int pcu_mlx_array_free(void* array, char* error) noexcept {
    return checked(error, [&] { delete static_cast<Array*>(array); });
}

PCU_EXPORT int pcu_mlx_matmul_prepare(
    void* session, int32_t rows, int32_t inner, int32_t columns,
    void** result, size_t* traces, char* error) noexcept {
    return checked(error, [&] {
        const auto& owner = *static_cast<SessionOwner*>(session);
        owner->validate();
        (void)count(rows, inner);
        (void)count(inner, columns);
        (void)count(rows, columns);
        auto prepared = std::make_shared<CompiledMatmul>(owner, rows, inner, columns);
        const auto trace_count = prepared->traces;
        const auto stream = owner->stream;
        prepared->compiled = mx::compile(ArrayFunction([trace_count, stream](const auto& inputs) {
            ++*trace_count;
            if (*trace_count != 1 || inputs.size() != 2 || !inputs[0].is_tracer()
                || !inputs[1].is_tracer())
                throw std::runtime_error("MLX compiled MatMul requires exactly one cold trace");
            return std::vector<mx::array>{mx::matmul(inputs[0], inputs[1], stream)};
        }), false);
        // Public logical graph descriptors only. They have no data and are never evaluated,
        // read or published. The compiler traces fresh placeholders and rebinds these IDs.
        const std::vector<mx::array> inputs{
            mx::array(prepared->left_shape, mx::float32, nullptr, {}),
            mx::array(prepared->right_shape, mx::float32, nullptr, {})};
        const auto outputs = prepared->compiled(inputs);
        if (*trace_count != 1 || outputs.size() != 1)
            throw std::runtime_error("MLX compiler did not produce one cold MatMul trace");
        const auto& output = outputs[0];
        if (output.shape() != prepared->output_shape || output.dtype() != mx::float32
            || !output.has_primitive() || !output.siblings().empty()
            || std::strcmp(output.primitive().name(), "Matmul") != 0
            || output.primitive().stream() != stream || output.inputs().size() != 2
            || output.inputs()[0].id() != inputs[0].id()
            || output.inputs()[1].id() != inputs[1].id())
            throw std::runtime_error("MLX compiled graph exceeds retained single MatMul profile");
        // Freeze the compiled public primitive. Rebinding one result descriptor retains
        // MLX allocation/eval scheduling; it neither retraces nor looks up the compiler cache.
        prepared->replay = [primitive = output.primitive_ptr(), shape = prepared->output_shape]
            (const auto& bound) { return std::vector<mx::array>{
                mx::array(shape, mx::float32, primitive, bound)}; };
        *traces = *trace_count;
        *result = new CompiledOwner(std::move(prepared));
    });
}

PCU_EXPORT int pcu_mlx_matmul_prepared_free(void* prepared, char* error) noexcept {
    return checked(error, [&] { delete static_cast<CompiledOwner*>(prepared); });
}

PCU_EXPORT size_t pcu_mlx_matmul_trace_count(void* prepared) noexcept {
    // Immutable diagnostic metadata only; no device data access or scheduling.
    return *(*static_cast<CompiledOwner*>(prepared))->traces;
}

PCU_EXPORT int pcu_mlx_matmul_replay(
    void* prepared, void* a, void* b, void** result, char* error) noexcept {
    std::unique_ptr<Pending> pending;
    int status = checked(error, [&] {
        const auto& compiled = *static_cast<CompiledOwner*>(prepared);
        const auto& owner = compiled->owner;
        owner->validate();
        const auto& left = *static_cast<Array*>(a);
        const auto& right = *static_cast<Array*>(b);
        if (left.owner != owner || right.owner != owner)
            throw std::invalid_argument("MLX arrays belong to another session");
        if (left.value.dtype() != mx::float32 || right.value.dtype() != mx::float32
            || left.value.shape() != compiled->left_shape
            || right.value.shape() != compiled->right_shape)
            throw std::invalid_argument("MLX compiled MatMul exact shapes mismatch");
        pending = std::make_unique<Pending>(owner, left.value, right.value);
        pending->compiled = compiled;
        const auto output = compiled->replay({left.value, right.value});
        pending->output = std::make_unique<Array>(owner, output[0]);
        pending->output->value.eval();
        mx::synchronize(owner->stream);
        pending->output->value.wait();
        *result = pending->output.release();
    });
    if (status != 0 && pending) {
        try {
            mx::synchronize(pending->session->stream);
        } catch (...) {
            pending->session->poisoned = true;
            (void)pending.release();
            error_text(error, "MLX completion unknown; compiled/input/output/session owners quarantined");
            status = 2;
        }
    }
    return status;
}

PCU_EXPORT int pcu_mlx_matmul(void* session, void* a, void* b, void** result, char* error) noexcept {
    std::unique_ptr<Pending> pending;
    int status = checked(error, [&] {
        const auto& owner = *static_cast<SessionOwner*>(session);
        owner->validate();
        const auto& left = *static_cast<Array*>(a);
        const auto& right = *static_cast<Array*>(b);
        if (left.owner != owner || right.owner != owner)
            throw std::invalid_argument("MLX arrays belong to another session");
        if (left.value.dtype() != mx::float32 || right.value.dtype() != mx::float32
            || left.value.ndim() != 2 || right.value.ndim() != 2)
            throw std::invalid_argument("MLX dense F32 matrix shapes mismatch");
        (void)count(left.value.shape(0), right.value.shape(1));
        pending = std::make_unique<Pending>(owner, left.value, right.value);
        pending->output = std::make_unique<Array>(owner, mx::matmul(left.value, right.value, owner->stream));
        pending->output->value.eval();
        mx::synchronize(owner->stream);
        pending->output->value.wait();
        *result = pending->output.release();
    });
    if (status != 0 && pending) {
        try {
            mx::synchronize(pending->session->stream);
        } catch (...) {
            pending->session->poisoned = true;
            (void)pending.release();
            error_text(error, "MLX completion unknown; input/output/session owners quarantined");
            status = 2;
        }
    }
    return status;
}

PCU_EXPORT int pcu_mlx_array_read(void* array, float* output, size_t length, char* error) noexcept {
    return checked(error, [&] {
        const auto& value = *static_cast<Array*>(array);
        value.owner->validate();
        if (!output || value.value.dtype() != mx::float32 || value.value.size() != length)
            throw std::invalid_argument("MLX readback extent mismatch");
        if (!value.value.is_available())
            throw std::runtime_error("MLX readback requires a terminal array");
        // Resolve CPU-visible terminal Metal data before publication; memcpy thereafter
        // has no fallible SDK call. The original owner retains the actual MLX backing.
        const float* source = value.value.data<float>();
        if (!source) throw std::runtime_error("MLX CPU-visible data unavailable");
        std::memcpy(output, source, length * sizeof(float));
    });
}
