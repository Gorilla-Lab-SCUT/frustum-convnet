#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// #include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <cmath>
#include <vector>

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

template <typename T>
__global__ void query_depth_point_gpu(int b, int n, int m, float dis_z, int nsample,
                                       const T *__restrict__ xyz1, const T *__restrict__ xyz2,
                                       long *__restrict__ idx, int *__restrict__ pts_cnt)
{

    // xyz1 (b, n, 3)
    // xyz2 (b, m, 3)

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m)
        return;

    xyz1 += n * 3 * bs_idx;
    xyz2 += m * 3 * bs_idx + pt_idx * 3;

    idx += m * nsample * bs_idx + pt_idx * nsample;

    pts_cnt += m * bs_idx + pt_idx; // counting how many unique points selected in local region

    int cnt = 0;
    // float x2 = xyz2[0];
    // float y2 = xyz2[1];
    float z2 = xyz2[2];

    for (int k = 0; k < n; ++k)
    {
        if (cnt == nsample)
            break; // only pick the FIRST nsample points in the ball
        // float x1 = xyz1[k * 3 + 0];
        // float y1 = xyz1[k * 3 + 1];
        float z1 = xyz1[k * 3 + 2];
        // float d1 = fabs(x2 - x1);
        // float d2 = fabs(y2 - y1);
        float d3 = fabs(z2 - z1);

        if (d3 < dis_z)
        {
            if (cnt == 0)
            { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                for (int l = 0; l < nsample; ++l)
                    idx[l] = k;
            }
            idx[cnt] = k;
            cnt += 1;
        }
    }
    *pts_cnt = cnt;
}

//require 32*n working space
void query_depth_point_forward_cuda(int b, int n, int m, float dis_z, int nsample,
                                     const at::Tensor &xyz1, const at::Tensor &xyz2, at::Tensor &idx, at::Tensor &pts_cnt)
{
    // cudaError_t err;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // cudaStream_t stream = at::globalContext().getCurrentCUDAStream();

    dim3 blocks(DIVUP(m, 256), b); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(256);
    AT_DISPATCH_FLOATING_TYPES(xyz1.type(), "query_depth_point_forward", [&] {
        query_depth_point_gpu<scalar_t><<<blocks, threads, 0, stream>>>(
            b, n, m, dis_z, nsample,
            xyz1.data<scalar_t>(),
            xyz2.data<scalar_t>(),
            idx.data<long>(),
            pts_cnt.data<int>());
    });
    THCudaCheck(cudaGetLastError());
}
