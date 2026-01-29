#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    
    if(idx<N){
        output[2*idx]     = A[idx];
        output[2*idx + 1] = B[idx];

    }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int size=N*sizeof(float);
    float *d_A,*d_B,*d_output;
    cudaMalloc((void**)&d_A,size);
    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B,size);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output,2*size);

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output,d_output,2*size,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);
}
