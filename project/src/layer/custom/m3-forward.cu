#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_M 8
#define TILE_N 16
#define TILE_K 16
#define COARSE 2

__global__ void conv_forward(const float *__restrict__ mask, const float *__restrict__ input, float *__restrict__ output, int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int HW_out = Height_out * Width_out;
    const int N = Batch * HW_out;
    const int K2 = K * K;
    const int Kdim = Channel * K2;

    __shared__ float tile_A[TILE_M * COARSE][TILE_K];
    __shared__ float tile_B[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row0 = blockIdx.y * (TILE_M * COARSE) + ty;
    int row1 = row0 + TILE_M;
    int col = blockIdx.x * TILE_N + tx;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int b = 0;
    int h_out = 0;
    int w_out = 0;

    if (col < N) {
        b = col / HW_out;
        int inner = col - b * HW_out;
        h_out = inner / Width_out;
        w_out = inner - h_out * Width_out;
    }

    for (int tile_k = 0; tile_k < Kdim; tile_k += TILE_K) {
        int k_a = tile_k + tx;

        if (row0 < Map_out && k_a < Kdim) {
            tile_A[ty][tx] = mask[row0 * Kdim + k_a];
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        if (row1 < Map_out && k_a < Kdim) {
            tile_A[ty + TILE_M][tx] = mask[row1 * Kdim + k_a];
        } else {
            tile_A[ty + TILE_M][tx] = 0.0f;
        }

        int k_b0 = tile_k + ty;
        int k_b1 = tile_k + ty + TILE_M;

        if (col < N && k_b0 < Kdim) {
            int c = k_b0 / K2;
            int rem = k_b0 - c * K2;
            int p = rem / K;
            int q = rem - p * K;

            tile_B[ty][tx] = input[
                b * Channel * Height * Width +
                c * Height * Width +
                (h_out + p) * Width +
                (w_out + q)
            ];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        if (col < N && k_b1 < Kdim) {
            int c = k_b1 / K2;
            int rem = k_b1 - c * K2;
            int p = rem / K;
            int q = rem - p * K;

            tile_B[ty + TILE_M][tx] = input[
                b * Channel * Height * Width +
                c * Height * Width +
                (h_out + p) * Width +
                (w_out + q)
            ];
        } else {
            tile_B[ty + TILE_M][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_K; i++) {
            float val = tile_B[i][tx];
            sum0 += tile_A[ty][i] * val;
            sum1 += tile_A[ty + TILE_M][i] * val;
        }

        __syncthreads();
    }

    if (col < N) {
        int out_base = b * Map_out * HW_out + h_out * Width_out + w_out;

        if (row0 < Map_out) {
            output[out_base + row0 * HW_out] = sum0;
        }

        if (row1 < Map_out) {
            output[out_base + row1 * HW_out] = sum1;
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    (void)host_output;

    size_t out_bytes = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);
    size_t in_bytes = (size_t)Batch * Channel * Height * Width * sizeof(float);
    size_t mask_bytes = (size_t)Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void**)device_output_ptr, out_bytes);
    cudaMalloc((void**)device_input_ptr, in_bytes);
    cudaMalloc((void**)device_mask_ptr, mask_bytes);

    cudaMemcpy(*device_input_ptr, host_input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_bytes, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int N = Batch * Height_out * Width_out;

    dim3 blockDim(TILE_N, TILE_M, 1);
    dim3 gridDim((N + TILE_N - 1) / TILE_N, (Map_out + TILE_M * COARSE - 1) / (TILE_M * COARSE), 1);

    conv_forward<<<gridDim, blockDim>>>(device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);

    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    size_t out_bytes = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);

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
