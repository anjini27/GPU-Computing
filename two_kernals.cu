#include<iostream>
#include<cuda.h>
using namespace std;

__global__ void firstKernal(){
    printf("First kernal running\n");
}
__global__ void secondKernal(){
    printf("Second kernal running\n");
}
int main(){
    firstKernal<<<1,2>>>();
    cudaDeviceSynchronize();
    secondKernal<<<1,2>>>();
    cudaDeviceSynchronize();
    return 0;
}
