#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        if(input[idx]<=0){
            output[idx]=0.0;
        }else{
          output[idx]=input[idx];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
      int size=N*sizeof(float);
      float* d_input,*d_output;
      cudaMalloc((void**)&d_input,size);
      cudaMemcpy(d_input,input,size,cudaMemcpyHostToDevice);

      cudaMalloc((void**)&d_output,size);
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input,d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(output,d_output,size,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}
