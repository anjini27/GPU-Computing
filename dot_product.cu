#include <cuda_runtime.h>

__global__ void Dot_kernel(float* A, float* B, float* result, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = A[idx] * B[idx];
        atomicAdd(result, val);
    }
}

// A, B, result are host pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int size = N * sizeof(float);

    float *d_A, *d_B, *d_result;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // initialize device result to zero
    cudaMemset(d_result, 0, sizeof(float));

    Dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_result, N);

    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
}
