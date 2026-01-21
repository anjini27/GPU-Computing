#include<stdio.h>
#include<cuda_runtime.h>

__global__ void Add5(int *arr,int n){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){//bound checking
        arr[gid]+=5;
    }
}

int main(){
    int N=7;
    int h_a[7]={1,2,3,4,5,6,7};
    int *d_a;
    cudaMalloc((void**)&d_a,N*sizeof(int));
    cudaMemcpy(d_a,h_a,N*sizeof(int),cudaMemcpyHostToDevice);
    int threads=4;
    int blocks=(N+threads-1)/threads;//no of blocks present
    Add5<<<blocks,threads>>>(d_a,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_a,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++){
        printf("%d\t",h_a[i]);
    }
    cudaFree(d_a);
    return 0;
}