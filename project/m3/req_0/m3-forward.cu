#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256
#define THREADS_PER_BLOCK 256
#define STREAM_COUNT 4

namespace {
const float *g_host_input_ptr = nullptr;
float *g_host_output_ptr = nullptr;
bool g_input_host_registered = false;
bool g_output_host_registered = false;
}

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    const size_t Height_out = Height - K + 1;
    const size_t Width_out = Width - K + 1;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t Width_unrolled = (size_t)Width_out * Height_out * Batch;

    if (tid >= Width_unrolled) {
        return;
    }

    size_t w = tid % Width_out;
    size_t tmp = tid / Width_out;
    size_t h = tmp % Height_out;
    tmp /= Height_out;
    size_t b = tmp;

    for (size_t c = 0; c < (size_t)Channel; c++) {
        for (size_t p = 0; p < (size_t)K; p++) {
            for (size_t q = 0; q < (size_t)K; q++) {
                output[(c * K * K + p * K + q) * Width_unrolled + tid] = in_4d(b, c, h + p, w + q);
            }
        }
    }

    #undef in_4d
}

// Permutes the matmul result from Map_out x Batch x image_size
// into Batch x Map_out x image_size.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const size_t out_bytes = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);
    const size_t in_bytes = (size_t)Batch * Channel * Height * Width * sizeof(float);
    const size_t mask_bytes = (size_t)Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void**)device_output_ptr, out_bytes);
    cudaMalloc((void**)device_input_ptr, in_bytes);
    cudaMalloc((void**)device_mask_ptr, mask_bytes);

    cudaMemcpy(*device_mask_ptr, host_mask, mask_bytes, cudaMemcpyHostToDevice);

    g_host_input_ptr = host_input;
    g_host_output_ptr = const_cast<float*>(host_output);

    cudaError_t err = cudaHostRegister((void*)g_host_input_ptr, in_bytes, cudaHostRegisterDefault);
    g_input_host_registered = (err == cudaSuccess);
    if (!g_input_host_registered) {
        cudaGetLastError();
    }

    err = cudaHostRegister((void*)g_host_output_ptr, out_bytes, cudaHostRegisterDefault);
    g_output_host_registered = (err == cudaSuccess);
    if (!g_output_host_registered) {
        cudaGetLastError();
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int out_image_size = Height_out * Width_out;

    const int stream_count = std::max(1, std::min(STREAM_COUNT, Batch));
    const int chunk_batch = (Batch + stream_count - 1) / stream_count;

    std::vector<cudaStream_t> streams(stream_count);
    std::vector<float*> unrolled_matrix(stream_count, nullptr);
    std::vector<float*> matmul_output(stream_count, nullptr);

    const size_t max_width_unrolled = (size_t)chunk_batch * Height_out * Width_out;
    const size_t max_unrolled_elems = (size_t)Height_unrolled * max_width_unrolled;
    const size_t max_matmul_elems = (size_t)chunk_batch * Map_out * Height_out * Width_out;

    for (int s = 0; s < stream_count; s++) {
        cudaStreamCreate(&streams[s]);
        cudaMalloc((void**)&unrolled_matrix[s], max_unrolled_elems * sizeof(float));
        cudaMalloc((void**)&matmul_output[s], max_matmul_elems * sizeof(float));
    }

    for (int s = 0; s < stream_count; s++) {
        const int b_start = s * chunk_batch;
        if (b_start >= Batch) {
            continue;
        }
        const int curr_batch = std::min(chunk_batch, Batch - b_start);
        const size_t width_unrolled = (size_t)curr_batch * Height_out * Width_out;

        const size_t input_offset = (size_t)b_start * Channel * Height * Width;
        const size_t output_offset = (size_t)b_start * Map_out * Height_out * Width_out;
        const size_t input_bytes = (size_t)curr_batch * Channel * Height * Width * sizeof(float);
        const size_t output_bytes = (size_t)curr_batch * Map_out * Height_out * Width_out * sizeof(float);

        float *device_input_chunk = const_cast<float*>(device_input) + input_offset;
        float *device_output_chunk = device_output + output_offset;
        const float *host_input_chunk = g_host_input_ptr + input_offset;
        float *host_output_chunk = g_host_output_ptr + output_offset;

        cudaMemcpyAsync(device_input_chunk, host_input_chunk, input_bytes, cudaMemcpyHostToDevice, streams[s]);

        dim3 unroll_grid_dim((unsigned int)((width_unrolled + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 1, 1);
        dim3 unroll_block_dim(THREADS_PER_BLOCK, 1, 1);
        matrix_unrolling_kernel<<<unroll_grid_dim, unroll_block_dim, 0, streams[s]>>>(
            device_input_chunk, unrolled_matrix[s], curr_batch, Channel, Height, Width, K
        );

        dim3 matmul_grid_dim((unsigned int)((width_unrolled + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH),
                             (unsigned int)((Map_out + MATMUL_TILE_WIDTH - 1) / MATMUL_TILE_WIDTH),
                             1);
        dim3 matmul_block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
        matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim, 0, streams[s]>>>(
            device_mask, unrolled_matrix[s], matmul_output[s], Map_out, Height_unrolled,
            Height_unrolled, (int)width_unrolled, Map_out, (int)width_unrolled
        );

        dim3 permute_kernel_grid_dim((unsigned int)((out_image_size + PERMUTE_BLOCK_SIZE - 1) / PERMUTE_BLOCK_SIZE),
                                     (unsigned int)curr_batch, 1);
        matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE, 0, streams[s]>>>(
            matmul_output[s], device_output_chunk, Map_out, curr_batch, out_image_size
        );

        cudaMemcpyAsync(host_output_chunk, device_output_chunk, output_bytes, cudaMemcpyDeviceToHost, streams[s]);
    }

    for (int s = 0; s < stream_count; s++) {
        cudaStreamSynchronize(streams[s]);
    }

    for (int s = 0; s < stream_count; s++) {
        cudaFree(matmul_output[s]);
        cudaFree(unrolled_matrix[s]);
        cudaStreamDestroy(streams[s]);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    (void)host_output;
    (void)Batch;
    (void)Map_out;
    (void)Channel;
    (void)Height;
    (void)Width;
    (void)K;

    if (g_input_host_registered && g_host_input_ptr != nullptr) {
        cudaHostUnregister((void*)g_host_input_ptr);
        g_input_host_registered = false;
        g_host_input_ptr = nullptr;
    }
    if (g_output_host_registered && g_host_output_ptr != nullptr) {
        cudaHostUnregister((void*)g_host_output_ptr);
        g_output_host_registered = false;
        g_host_output_ptr = nullptr;
    }

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
