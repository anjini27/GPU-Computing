#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A,const float*, float* C,int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < M && col < K) {
        float pval = 0.0f;

        for (int k = 0; k < N; ++k) {
            pval += A[row * N + k] * B[k * K + col];
        }

        C[row * K + col] = pval;
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int size_A=M*N*sizeof(float);
    int size_B=N*K*sizeof(float);
    int size_C=M*K*sizeof(float);
    
     float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size_A);
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_B, size_B);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_C, size_C);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
