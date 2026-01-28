#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
     int idx=blockIdx.x*blockDim.x+threadIdx.x;
if(idx<halfN){
  float x1=input[idx];
  float x2=input[halfN+idx];
  float sigmoid=1.0f/(1.0f+expf(-x1));
  float silu=x1*sigmoid;
  output[idx]=silu*x2;
}
    
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    int size=N*sizeof(float);
    float *d_output,*d_input;
    cudaMalloc((void**)&d_input,size);
    cudaMemcpy(d_input,input,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output,halfN*sizeof(float));
    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input,d_output, halfN);
    cudaDeviceSynchronize();
    cudaMemcpy(output,d_output,halfN*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_input);
}
