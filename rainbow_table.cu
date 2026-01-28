#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        unsigned int h = input[idx];
        for(int i=0;i<R;i++){
            h=fnv1a_hash(h);
        }
        output[idx]=h;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int *d_input;
    unsigned int *d_output;
    int size_in=N*sizeof(int);
    int size_out=N*sizeof(unsigned int);
    cudaMalloc((void**)&d_input,size_in);
    cudaMemcpy(d_input,input,size_in,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output,size_out);
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input,d_output, N, R);
    cudaDeviceSynchronize();
    cudaMemcpy(output,d_output,size_out,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}
