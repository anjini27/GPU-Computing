#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;

    if(idx<N){
  output[idx]=input[idx]* 1.0 / (1.0 + exp(-input[idx]));
    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int size=N*sizeof(float);
    float *d_output,*d_input;
    cudaMalloc((void**)&d_input,size);
    cudaMemcpy(d_input,input,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output,size);
    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input,d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(output,d_output,size,cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_input);
}
