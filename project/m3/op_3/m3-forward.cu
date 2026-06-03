#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// Parameter configurations for sweep
#ifndef SWEEP_CONFIG
#define SWEEP_CONFIG 3
#endif

#if SWEEP_CONFIG == 0
#define TILE_M 16
#define TILE_N 8
#define TILE_K 8
#define COARSEN_X 1
#elif SWEEP_CONFIG == 1
#define TILE_M 16
#define TILE_N 16
#define TILE_K 8
#define COARSEN_X 1
#elif SWEEP_CONFIG == 2
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16
#define COARSEN_X 1
#elif SWEEP_CONFIG == 3
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16
#define COARSEN_X 2
#elif SWEEP_CONFIG == 4
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16
#define COARSEN_X 4
#else
#error "Unsupported SWEEP_CONFIG value."
#endif

__global__ void matmul_conv_fused_sweep(const float *mask, const float *input, float *output,
                                        int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const size_t Height_out = Height - K + 1;
    const size_t Width_out = Width - K + 1;
    const size_t k_extent = (size_t)Channel * K * K;
    const size_t n_extent = (size_t)Batch * Height_out * Width_out;

    __shared__ float tile_A[TILE_M][TILE_K];
    __shared__ float tile_B[TILE_K][TILE_N * COARSEN_X];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;

    const size_t row = blockIdx.y * TILE_M + ty;
    const size_t col0 = blockIdx.x * (TILE_N * COARSEN_X) + tx;

    float acc[COARSEN_X];
    for (int c = 0; c < COARSEN_X; c++) {
        acc[c] = 0.0f;
    }

    for (size_t tile_k = 0; tile_k < k_extent; tile_k += TILE_K) {
        if (row < (size_t)Map_out && tile_k + tx < k_extent) {
            tile_A[ty][tx] = mask[row * k_extent + (tile_k + tx)];
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        const size_t k_idx = tile_k + ty;
        for (int c = 0; c < COARSEN_X; c++) {
            const size_t col = col0 + (size_t)c * TILE_N;
            if (col < n_extent && k_idx < k_extent) {
                size_t tmp = col;
                size_t w_out = tmp % Width_out;
                tmp = tmp / Width_out;
                size_t h_out = tmp % Height_out;
                size_t b = tmp / Height_out;

                size_t in_c = k_idx / (K * K);
                size_t rem = k_idx % (K * K);
                size_t p = rem / K;
                size_t q = rem % K;

                tile_B[ty][tx + c * TILE_N] =
                    input[(b * Channel * Height * Width) + (in_c * Height * Width) +
                          (h_out + p) * Width + (w_out + q)];
            } else {
                tile_B[ty][tx + c * TILE_N] = 0.0f;
            }
        }

        __syncthreads();

        for (int kk = 0; kk < TILE_K; kk++) {
            const float a = tile_A[ty][kk];
            for (int c = 0; c < COARSEN_X; c++) {
                acc[c] += a * tile_B[kk][tx + c * TILE_N];
            }
        }

        __syncthreads();
    }

    if (row < (size_t)Map_out) {
        for (int c = 0; c < COARSEN_X; c++) {
            const size_t col = col0 + (size_t)c * TILE_N;
            if (col < n_extent) {
                size_t tmp = col;
                size_t w_out = tmp % Width_out;
                tmp = tmp / Width_out;
                size_t h_out = tmp % Height_out;
                size_t b = tmp / Height_out;

                output[(b * Map_out * Height_out * Width_out) + (row * Height_out * Width_out) +
                       (h_out * Width_out) + w_out] = acc[c];
            }
        }
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
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    dim3 blockDim(TILE_N, TILE_M);
    dim3 gridDim((Batch * Height_out * Width_out + (TILE_N * COARSEN_X) - 1) / (TILE_N * COARSEN_X),
                 (Map_out + TILE_M - 1) / TILE_M);

    matmul_conv_fused_sweep<<<gridDim, blockDim>>>(device_mask, device_input, device_output,
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
