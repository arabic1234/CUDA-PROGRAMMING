#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define I 4096
#define J 4096

#define errcheck(ans){gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t err, const char * file, int line){
    if(err != cudaSuccess){
        fprintf(stderr, "\n error is %s %s %d\n", cudaGetErrorString(err), file, line);
    }
}

#define asyncerr(){gpuAssert((__FILE__, __LINE__))}
inline void gpuAssert(const char * file, int line){
    cudaError_t ERR = cudaGetLastError();
    if(ERR != cudaSuccess){
        fprintf(stderr, "\n error is %s %s %d \n", cudaGetErrorString(ERR), file, line);
    }
}

__global__ void matrixadd(float *a, float *b, float *c, int n, int m){
    int row = threadIdx.x + ( blockIdx.x * blockDim.x);
    int col = threadIdx.y + ( blockIdx.y * blockDim.y);

    if(row < n && col < m){
        int idx = col + ( row * m );
        c[idx] = a[idx] + b[idx];
    }
}

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n Device Name: %s", prop.name);

    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    int size = (I*J) * sizeof(float);

    cudaMallocHost((void**)&a, size);
    cudaMallocHost((void**)&b, size);
    cudaMallocHost((void**)&c, size);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    for(int i =0; i < (I*J); i++){
        a[i] = 20*static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        b[i] = 30*static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpyAsync(d_a, a, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, b, size, cudaMemcpyHostToDevice, stream1);

    dim3 blockDim (16,8);
    dim3 gridDim (((I + blockDim.x -1) / blockDim.x) , ((J + blockDim.y - 1) / blockDim.y ));

    matrixadd <<< gridDim, blockDim, 0, stream1 >>>(d_a, d_b, d_c, I, J);

    cudaMemcpyAsync(c, d_c, size, cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream1);
    cudaEventRecord(stop);

    for(int i = 0; i < 5; i++){
        printf("\n %f + %f = %f \n", a[i], b[i], c[i]);
    }

    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\n The Execution Time is: %0.1f milliseconds\n", milliseconds);

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}