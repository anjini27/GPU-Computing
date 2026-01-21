#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadCount() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int totalThreads = gridDim.x * blockDim.x;
        printf("Total threads = %d\n", totalThreads);
    }
}

int main() {
    printThreadCount<<<3, 4>>>();  // you did NOT store 3 or 4 in variables
    cudaDeviceSynchronize();
    return 0;
}
