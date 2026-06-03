#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__global__ void matmul_conv_fused(const float *__restrict__ mask,
                                  const float *__restrict__ input,
                                  float *__restrict__ output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    size_t Height_out = Height - K + 1;
    size_t Width_out = Width - K + 1;

    __shared__ float tile_A[TILE_M][TILE_K];
    __shared__ float tile_B[TILE_K][TILE_N];

    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    size_t row = blockIdx.y * TILE_M + ty;
    size_t col = blockIdx.x * TILE_N + tx;

    float acc = 0.0f;

    for (size_t tile_k = 0; tile_k < (size_t)Channel * K * K; tile_k += TILE_K) {
        if (row < (size_t)Map_out && tile_k + tx < (size_t)Channel * K * K) {
            tile_A[ty][tx] = mask[row * ((size_t)Channel * K * K) + (tile_k + tx)];
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        if (col < (size_t)Batch * Height_out * Width_out && tile_k + ty < (size_t)Channel * K * K) {
            size_t tmp = col;
            size_t w_out = tmp % Width_out;
            tmp = tmp / Width_out;
            size_t h_out = tmp % Height_out;
            size_t b = tmp / Height_out;

            size_t k = tile_k + ty;
            size_t c = k / (K * K);
            size_t rem = k % (K * K);
            size_t p = rem / K;
            size_t q = rem % K;

            tile_B[ty][tx] = input[(b * Channel * Height * Width) + (c * Height * Width) +
                                   (h_out + p) * Width + (w_out + q)];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (size_t kk = 0; kk < TILE_K; kk++) {
            acc += tile_A[ty][kk] * tile_B[kk][tx];
        }

        __syncthreads();
    }

    if (row < (size_t)Map_out && col < (size_t)Batch * Height_out * Width_out) {
        size_t tmp = col;
        size_t w_out = tmp % Width_out;
        tmp = tmp / Width_out;
        size_t h_out = tmp % Height_out;
        size_t b = tmp / Height_out;

        output[(b * Map_out * Height_out * Width_out) + (row * Height_out * Width_out) +
               (h_out * Width_out) + w_out] = acc;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
                                                     float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
                                                     const int Batch, const int Map_out, const int Channel, const int Height,
                                                     const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    (void)host_output;

    int out_bytes = Batch * Map_out * Height_out * Width_out * sizeof(float);
    int in_bytes = Batch * Channel * Height * Width * sizeof(float);
    int mask_bytes = Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void**)device_output_ptr, out_bytes);
    cudaMalloc((void**)device_input_ptr, in_bytes);
    cudaMalloc((void**)device_mask_ptr, mask_bytes);

    cudaMemcpy(*device_input_ptr, host_input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_bytes, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                              const int Batch, const int Map_out, const int Channel, const int Height,
                                              const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    dim3 blockDim(TILE_N, TILE_M);
    dim3 gridDim((Batch * Height_out * Width_out + TILE_N - 1) / TILE_N,
                 (Map_out + TILE_M - 1) / TILE_M);
    matmul_conv_fused<<<gridDim, blockDim>>>(device_mask, device_input, device_output,
                                             Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input,
                                                     float *device_mask, const int Batch, const int Map_out,
                                                     const int Channel, const int Height, const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int out_bytes = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMemcpy(host_output, device_output, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
