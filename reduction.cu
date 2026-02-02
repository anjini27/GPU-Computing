#include <cuda_runtime.h>

__global__ void reduction(const float* input, float* output, int N) {

    __shared__ double sdata[256];   // double precision

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N)
        sdata[tid] = (double)input[idx];
    else
        sdata[tid] = 0.0;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(output, (float)sdata[0]);
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
