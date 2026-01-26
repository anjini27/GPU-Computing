#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        if (input[idx] == K) {
            atomicAdd(output, 1);  
        }
    }
}

extern "C" void solve(const int* input, int* output, int N, int K) {

    int *d_input, *d_output;
    int size = N * sizeof(int);

 
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, sizeof(int));

    
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

   
    cudaMemset(d_output, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

   
    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, K);
    cudaDeviceSynchronize();

 
    cudaMemcpy(output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

 
    cudaFree(d_input);
    cudaFree(d_output);
}
