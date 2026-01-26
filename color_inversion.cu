#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * 4;

    if (idx < total) {

    
        if (idx % 4 != 3) {
            image[idx] = 255 - image[idx];
        }
      
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = ((width * height*4) + threadsPerBlock - 1) / threadsPerBlock;
    int size=width*height*4*sizeof(unsigned char);
    unsigned char* d_im;

    cudaMalloc((void**)&d_im, size);
    cudaMemcpy(d_im, image, size, cudaMemcpyHostToDevice);

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_im, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(image,d_im,size,cudaMemcpyDeviceToHost);

    cudaFree(d_im);
}
