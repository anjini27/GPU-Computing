#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<N&&col<N){
        int idx=row*N+col;
        B[idx]=A[idx];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int size=total*sizeof(float);
    float *d_A,*d_B;
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    cudaDeviceSynchronize();
    cudaMemcpy(B,d_B,size,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
}
