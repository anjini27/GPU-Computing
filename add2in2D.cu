#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add2(int *mat,int rows,int cols){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;

    if(row<rows&&col<cols){
        int idx=row*cols+col;
        mat[idx] +=2;
    }
}

int main(){
    int rows=3,cols=3;
    int h_mat[9]={0,1,2,3,4,5,6,7,8};
    int *d_mat;

    cudaMalloc((void**)&d_mat,rows*cols*sizeof(int));
    cudaMemcpy(d_mat,h_mat,rows*cols*sizeof(int),cudaMemcpyHostToDevice);
    dim3 threads(2, 2);
    dim3 blocks(
        (cols + threads.x - 1) / threads.x,
        (rows + threads.y - 1) / threads.y
    );

    add2<<<blocks, threads>>>(d_mat, rows, cols);
    cudaDeviceSynchronize();
        cudaMemcpy(h_mat, d_mat, rows * cols * sizeof(int),
               cudaMemcpyDeviceToHost);

    printf("Result:\n");
    for (int i = 0; i < rows * cols; i++) {
        printf("%d ", h_mat[i]);
    }

    cudaFree(d_mat);
    return 0;


    
   
}
