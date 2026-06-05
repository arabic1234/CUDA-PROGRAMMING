# CUDA Learning Projects

This repository contains CUDA C++ programs developed while learning GPU Computing, Parallel Programming, and GPU Performance Optimization.

## Projects

### Vector Addition
- Parallel vector addition using CUDA kernels
- Grid and block configuration
- Thread indexing

### Matrix Addition
- Parallel matrix addition using CUDA kernels
- 2D thread organization
- 2D matrix-to-linear memory mapping
- Row-major memory layout

### CUDA Streams
- Concurrent execution using multiple CUDA streams
- Overlapping computation and data transfers
- Understanding asynchronous execution

### Memory Transfer Optimization
- Host-to-device and device-to-host transfers
- Pinned (page-locked) memory allocation
- Asynchronous memory transfers using `cudaMemcpyAsync`
- Measuring performance improvements over synchronous transfers
