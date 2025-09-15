// hello_cuda.cu
#include <stdio.h>

// A simple CUDA Kernel
__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // Launch 5 threads
    helloFromGPU<<<1, 5>>>();
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    printf("Hello from the CPU!\n");
    return 0;
}