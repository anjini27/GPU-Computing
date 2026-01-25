#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output,int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row in input
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col in input

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y); 
        int size_in=rows*cols*sizeof(float);
        int size_out=cols*rows*sizeof(float);
        float *d_in,*d_out;
        cudaMalloc((void**)&d_in, size_in); 
        cudaMemcpy(d_in,input,size_in,cudaMemcpyHostToDevice);
        
        cudaMalloc((void**)&d_out, size_out);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, size_out, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
