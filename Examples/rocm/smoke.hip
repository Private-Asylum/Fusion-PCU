#include <cmath>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define HIP_CHECK(call) do { \
    hipError_t status = (call); \
    if (status != hipSuccess) { \
        std::fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorString(status)); \
        return 1; \
    } \
} while (0)

#define ROCBLAS_CHECK(call) do { \
    rocblas_status status = (call); \
    if (status != rocblas_status_success) { \
        std::fprintf(stderr, "%s failed with rocBLAS status %d\n", #call, (int)status); \
        return 1; \
    } \
} while (0)

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

int main() {
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    std::printf("ROCm device %d: %s (%s)\n", device, properties.name, properties.gcnArchName);

    constexpr int n = 256;
    float host_a[n], host_b[n], host_sum[n];
    for (int i = 0; i < n; ++i) {
        host_a[i] = static_cast<float>(i);
        host_b[i] = static_cast<float>(2 * i + 1);
    }
    float *a = nullptr, *b = nullptr, *sum = nullptr;
    HIP_CHECK(hipMalloc(&a, sizeof(host_a)));
    HIP_CHECK(hipMalloc(&b, sizeof(host_b)));
    HIP_CHECK(hipMalloc(&sum, sizeof(host_sum)));
    HIP_CHECK(hipMemcpy(a, host_a, sizeof(host_a), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b, host_b, sizeof(host_b), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(add_kernel, dim3((n + 127) / 128), dim3(128), 0, 0, a, b, sum, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(host_sum, sum, sizeof(host_sum), hipMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (host_sum[i] != host_a[i] + host_b[i]) {
            std::fprintf(stderr, "HIP kernel mismatch at %d: expected %f, got %f\n", i,
                         host_a[i] + host_b[i], host_sum[i]);
            return 1;
        }
    }
    std::puts("HIP kernel launch and output verification passed");
    HIP_CHECK(hipFree(sum));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(a));

    // rocBLAS uses column-major matrices: C = [[19, 22], [43, 50]].
    const float matrix_a[] = {1, 3, 2, 4};
    const float matrix_b[] = {5, 7, 6, 8};
    const float expected[] = {19, 43, 22, 50};
    float matrix_c[] = {0, 0, 0, 0};
    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    HIP_CHECK(hipMalloc(&device_a, sizeof(matrix_a)));
    HIP_CHECK(hipMalloc(&device_b, sizeof(matrix_b)));
    HIP_CHECK(hipMalloc(&device_c, sizeof(matrix_c)));
    HIP_CHECK(hipMemcpy(device_a, matrix_a, sizeof(matrix_a), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, matrix_b, sizeof(matrix_b), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(device_c, 0, sizeof(matrix_c)));
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                2, 2, 2, &alpha, device_a, 2, device_b, 2,
                                &beta, device_c, 2));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(matrix_c, device_c, sizeof(matrix_c), hipMemcpyDeviceToHost));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(matrix_c[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "rocBLAS GEMM mismatch at %d: expected %f, got %f\n", i,
                         expected[i], matrix_c[i]);
            return 1;
        }
    }
    HIP_CHECK(hipFree(device_c));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_a));
    std::puts("rocBLAS SGEMM and output verification passed");
    return 0;
}
