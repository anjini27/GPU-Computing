#include<stdio.h>
#include<cuda_runtime.h>
 
 __global__ void printArray(int *arr){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<3){
        printf("array[%d]: %d\n",gid,arr[gid]);

    }
    printf("adding 10 to every element in the array\n");
    // if(gid<3){
    //     arr[gid]=arr[gid]+10;
    //     printf("%d\n",arr[gid]);
    // }
 }
 __global__ void adding10(int *arr){
     int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<3){
        arr[gid]=arr[gid]+10;
        printf("%d\n",arr[gid]);
    }
 }

 int main(){
    int h_a[3]={1,2,3};
    int *d_a;
    cudaMalloc((void**)&d_a,3*sizeof(int));
    cudaMemcpy(d_a,h_a,3*sizeof(int),cudaMemcpyHostToDevice);
    printArray<<<1,3>>>(d_a);
    adding10<<<1,3>>>(d_a);
    cudaFree(d_a);
    return 0;
 }
