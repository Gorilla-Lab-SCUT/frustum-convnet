#include <torch/torch.h>
//#include <torch/extension.h>
#include <vector>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// CUDA forward declarations
void query_depth_point_forward_cuda(
    int b,
    int n,
    int m,
    float dis_z,
    int nsample,
    const at::Tensor &xyz1,
    const at::Tensor &xyz2,
    at::Tensor &idx,
    at::Tensor &pts_cnt);

// C++ interface
void query_depth_point_forward(
    int b,
    int n,
    int m,
    float dis_z,
    int nsample,
    const at::Tensor &xyz1,
    const at::Tensor &xyz2,
    at::Tensor &idx,
    at::Tensor &pts_cnt)
{

  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  CHECK_INPUT(idx);
  CHECK_INPUT(pts_cnt);

  // cudaStream_t stream = THCState_getCurrentStream(state);

  query_depth_point_forward_cuda(b, n, m, dis_z, nsample, xyz1, xyz2, idx, pts_cnt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &query_depth_point_forward, "query_depth_point_forward");
}
