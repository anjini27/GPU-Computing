#include<iostream>
#include<cuda_runtime.h>

__global__ void printGlobalId(){
    int g=blockIdx.x*blockDim.x+threadIdx.x;
    printf("global Id: %d\n",g);
    printf("thread id: %d\n",threadIdx.x);
}

int main(){
    printGlobalId<<<2,5>>>();
    cudaDeviceSynchronize();
    return 0;
}