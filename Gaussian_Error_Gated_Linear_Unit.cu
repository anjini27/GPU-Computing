#include <cuda_runtime.h>

#include <cuda_runtime.h>
#include <math.h>

__global__ void geglu_kernel(const float* input, float* output, int halfN) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < halfN) {

        float x1 = input[idx];
        float x2 = input[halfN + idx];

        float gaussian = 1.0f + erff(x2 / sqrtf(2.0f));
        float gelu = 0.5f * x2 * gaussian;

        output[idx] = x1 * gelu;
    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    int size_in=N*sizeof(float);
    int size_out=halfN*sizeof(float);
    float *d_input,*d_output;
    cudaMalloc((void**)&d_input,size_in);
    cudaMemcpy(d_input,input,size_in,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output,size_out);
    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input,d_output, halfN);
    cudaDeviceSynchronize();
    cudaMemcpy(output,d_output,size_out,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}
