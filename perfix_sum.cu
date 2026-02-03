#include <cuda_runtime.h>

__global__ void prefixsum_kernal(const float* input,float* output,int N){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N&&idx==0)
    output[0]=input[0];
    if(idx<N&&idx>=1){
        int val=input[idx];
       output[idx] = output[idx-1];     // normal load
       atomicAdd(&output[idx], val);   // atomic increment


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
   prefixsum_kernal<<<blocksPerGrid, threadsPerBlock>>>(d_input,d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(output,d_output,size,cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_input);
}
