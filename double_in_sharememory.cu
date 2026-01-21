#include <stdio.h>
#include <cuda_runtime.h>

__global__ void SharedMemory(int *arr){
    __shared__ int s[8];
    int tid=threadIdx.x;
    s[tid]=arr[tid];
    __syncthreads();
    printf("threadId %d:%d\n",tid,s[tid]*2);
}

int main(){
   
    int h_arr[8]={1,2,3,4,5,6,7,8};
    int *d_arr;

    cudaMalloc((void**)&d_arr,8*sizeof(int));
    cudaMemcpy(d_arr,h_arr,8*sizeof(int),cudaMemcpyHostToDevice);
    
    SharedMemory<<<1,8>>>(d_arr);
    cudaDeviceSynchronize();
        

    cudaFree(d_arr);
    return 0;  
}
