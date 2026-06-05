#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#define SIZE 1024*1024*1024
#define chunk_size 1024*1024*128
#define block_size 256

// SYNCHRONOUS ERROR CHECKING 
#define checkerr(ans){gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t err, const char * file, int line){
    if(err != cudaSuccess){
        fprintf(stderr,"\n error: %s %s %d", cudaGetErrorString(err), file, line);
    }
}

//ASYNCHRONOUS ERROR CHECKING
#define asyncerr(){gpuAssert( __FILE__, __LINE__ );}
inline void gpuAssert(const char * file, int line){
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "\n error: %s %s %d", cudaGetErrorString(err), file, line);
    }
}


//kernel to run on device 
__global__ void vecadd(int* c_A, int* c_B, int* c_C, int n){
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i<n){
    c_C[i] = c_A[i] + c_B[i];
    }

}

void rand_int(int *x, int n){
    for(int i =0; i<n; i++){
        x[i] = rand() % 100;
    }
}

//host code
int main(){

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n Async Enginecount: %d\n", prop.asyncEngineCount);
    printf("\n Concurrent kernels: %d\n", prop.concurrentKernels);

    int *c_A, *c_B, *c_C;
    int *d_A, *d_B, *d_C;
    int s = chunk_size * sizeof(int);


    // allocate memory for host variables 
    checkerr(cudaMallocHost((void**)&c_A, s));
    cudaMallocHost((void**)&c_B, s);
    cudaMallocHost((void**)&c_C, s);


    //allocate memory for device variables 
    cudaMalloc((void**)&d_A, s);
    cudaMalloc((void**)&d_B, s);
    cudaMalloc((void**)&d_C, s);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for(size_t offset = 0; offset < SIZE; offset += chunk_size){
        int current_chunk_size = (SIZE - offset) < chunk_size ? (SIZE - offset) : chunk_size;
        int grid_size = ((current_chunk_size + block_size - 1 ) / block_size);
        printf("\n The offset value is : %zu\n", offset);
        printf("\n The current chunk size is: %d \n", current_chunk_size);
        int ccs_inbytes = current_chunk_size * sizeof(int);
        rand_int(c_A, current_chunk_size);
        rand_int(c_B, current_chunk_size);

        cudaMemcpyAsync(d_A, c_A, ccs_inbytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_B, c_B, ccs_inbytes, cudaMemcpyHostToDevice, stream);

        vecadd<<< grid_size, block_size, 0, stream >>>(d_A, d_B, d_C, current_chunk_size);
        asyncerr();

        cudaMemcpyAsync(c_C, d_C, ccs_inbytes, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        for(int i= 0; i<5; i++){
        printf("\n %d + %d = %d \n ", c_A[i], c_B[i], c_C[i]);
    }

    }
    
    cudaEventRecord(stop);
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\n Time taken for execution is :%0.1f milliseconds \n", milliseconds);

    cudaFreeHost(c_A);
    cudaFreeHost(c_B);
    cudaFreeHost(c_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
