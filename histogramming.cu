#include <cuda_runtime.h>

// input, histogram are device pointers
__global__ void Histogram_kernal(const int* input, int* histogram, int N, int num_bins) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        int val=input[idx];
        if(val<num_bins){
         atomicAdd(&histogram[val], 1);
        }
    }
}


extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {

    int *d_input, *d_histogram;
    int size_in = N * sizeof(int);
    int size_histogram=num_bins*sizeof(int);
 
    cudaMalloc((void**)&d_input, size_in);
    cudaMalloc((void**)&d_histogram, size_histogram);

    
    cudaMemcpy(d_input, input, size_in, cudaMemcpyHostToDevice);

   
    cudaMemset(d_histogram, 0, size_histogram);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

   
       Histogram_kernal<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_histogram, N, num_bins);
    cudaDeviceSynchronize();

 
    cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);

 
    cudaFree(d_input);
    cudaFree(d_histogram);
}
