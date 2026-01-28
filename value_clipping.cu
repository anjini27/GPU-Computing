#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        float x=input[idx];
        if(input[idx]<lo)
            output[idx]=lo;
        else if(input[idx]>hi)
         output[idx]=hi;
        else
            output[idx]=x;
        
        
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
     int size=N*sizeof(float);
    float *d_output,*d_input;
    cudaMalloc((void**)&d_input,size);
    cudaMemcpy(d_input,input,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output,size);
    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input,d_output, lo, hi, N);
    cudaDeviceSynchronize();
    cudaMemcpy(output,d_output,size,cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_input);

}
