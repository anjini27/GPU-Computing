#include <cuda_runtime.h>

__global__ void reduction(const float* input, float* output, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        atomicAdd(output, input[idx]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int size = N * sizeof(float);

    float *d_in, *d_out;

    cudaMalloc((void**)&d_in, size);
    cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));  
    reduction<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
