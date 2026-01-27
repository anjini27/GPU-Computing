#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
     int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<N && col<M){
        int idx=row*M+col;
       if (inaput[idx] == K) {
            atomicAdd(output, 1);  
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int *d_input, *d_output;
    int size = N * M *sizeof(int);

 
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, sizeof(int));

    
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

   
    cudaMemset(d_output, 0, sizeof(int));
    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input,d_output, N, M, K);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

 
    cudaFree(d_input);
    cudaFree(d_output);
}
