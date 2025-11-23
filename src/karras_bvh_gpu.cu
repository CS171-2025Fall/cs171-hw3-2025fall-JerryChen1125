#ifdef USE_CUDA

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "rdr/karras_bvh_gpu.h"
#include "rdr/math_aliases.h"

RDR_NAMESPACE_BEGIN

// Device helper: expand 10-bit integer into 30-bit Morton interleaving
__device__ inline uint32_t expandBitsDevice(uint32_t v) {
  uint32_t x = v & 0x3ffu;
  x          = (x | (x << 16)) & 0x30000FFu;
  x          = (x | (x << 8)) & 0x300F00Fu;
  x          = (x | (x << 4)) & 0x30C30C3u;
  x          = (x | (x << 2)) & 0x9249249u;
  return x;
}

__global__ void computeMortonKernel(const float *centers_x,
    const float *centers_y, const float *centers_z, int n, float bb_min_x,
    float bb_min_y, float bb_min_z, float extent_x, float extent_y,
    float extent_z, uint32_t *out_morton) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float nx      = extent_x <= 0 ? 0.5f : (centers_x[idx] - bb_min_x) / extent_x;
  float ny      = extent_y <= 0 ? 0.5f : (centers_y[idx] - bb_min_y) / extent_y;
  float nz      = extent_z <= 0 ? 0.5f : (centers_z[idx] - bb_min_z) / extent_z;
  uint32_t xi   = min(1023u, (uint32_t)(nx * 1023.0f));
  uint32_t yi   = min(1023u, (uint32_t)(ny * 1023.0f));
  uint32_t zi   = min(1023u, (uint32_t)(nz * 1023.0f));
  uint32_t code = (expandBitsDevice(xi) << 2) | (expandBitsDevice(yi) << 1) |
                  expandBitsDevice(zi);
  out_morton[idx] = code;
}

// Karras kernel: each thread handles one internal node
__global__ void karrasTopoKernel(
    const uint32_t *morton, int n, int *leftChild, int *rightChild) {
  int i             = blockIdx.x * blockDim.x + threadIdx.x;
  int internalCount = n - 1;
  if (i >= internalCount) return;

  auto delta = [&](int a, int b) {
    if (b < 0 || b >= n) return -1;
    uint32_t x = morton[a] ^ morton[b];
    if (x == 0) return 32;  // identical
    return __clz(x);
  };

  int d         = (delta(i, i + 1) - delta(i, i - 1) > 0) ? 1 : -1;
  int delta_min = delta(i, i - d);

  // find upper bound; start from larger number and multiply by 4
  int lmax = 128;
  while (delta(i, i + lmax * d) > delta_min) lmax *= 4;

  int l = 0;
  for (int t = lmax >> 2; t > 0; t >>= 2) {
    if (delta(i, i + (l + t) * d) > delta_min) l += t;
  }
  int j = i + l * d;

  int delta_node = delta(i, j);
  int s          = 0;
  int dl         = abs(j - i);
  int t          = 1;
  while (t <= dl) t <<= 1;
  for (int step = t >> 1; step > 0; step >>= 1) {
    if (s + step <= dl && delta(i, i + (s + step) * d) > delta_node) s += step;
  }
  int gamma = i + s * d + min(d, 0);

  int a = min(i, j);
  int b = max(i, j);
  if (a == gamma)
    leftChild[i] = (n - 1) + gamma;
  else
    leftChild[i] = gamma;

  if (b == gamma + 1)
    rightChild[i] = (n - 1) + (gamma + 1);
  else
    rightChild[i] = gamma + 1;
}

bool karras_bvh_gpu_build(const std::vector<Vec3f> &centers,
    const Vec3f &bb_min, const Vec3f &bb_extent,
    std::vector<uint32_t> &morton_out, std::vector<int> &indices_out,
    std::vector<int> &left_out, std::vector<int> &right_out) noexcept {
  int n = (int)centers.size();
  if (n == 0) return true;

  // device buffers for centers
  thrust::device_vector<float> d_x(n), d_y(n), d_z(n);
  for (int i = 0; i < n; ++i) {
    d_x[i] = centers[i].x;
    d_y[i] = centers[i].y;
    d_z[i] = centers[i].z;
  }

  thrust::device_vector<uint32_t> d_morton(n);
  int block = 256;
  int grid  = (n + block - 1) / block;
  computeMortonKernel<<<grid, block>>>(thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()),
      thrust::raw_pointer_cast(d_z.data()), n, bb_min.x, bb_min.y, bb_min.z,
      bb_extent.x, bb_extent.y, bb_extent.z,
      thrust::raw_pointer_cast(d_morton.data()));
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return false;
  }

  // indices
  thrust::device_vector<int> d_idx(n);
  thrust::sequence(d_idx.begin(), d_idx.end());

  // sort by morton key
  thrust::sort_by_key(d_morton.begin(), d_morton.end(), d_idx.begin());

  // allocate child arrays
  int internalCount = n - 1;
  thrust::device_vector<int> d_left(internalCount);
  thrust::device_vector<int> d_right(internalCount);

  int grid2 = (internalCount + block - 1) / block;
  karrasTopoKernel<<<grid2, block>>>(thrust::raw_pointer_cast(d_morton.data()),
      n, thrust::raw_pointer_cast(d_left.data()),
      thrust::raw_pointer_cast(d_right.data()));
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return false;
  }

  // copy back
  morton_out.resize(n);
  indices_out.resize(n);
  left_out.resize(internalCount);
  right_out.resize(internalCount);
  thrust::copy(d_morton.begin(), d_morton.end(), morton_out.begin());
  thrust::copy(d_idx.begin(), d_idx.end(), indices_out.begin());
  thrust::copy(d_left.begin(), d_left.end(), left_out.begin());
  thrust::copy(d_right.begin(), d_right.end(), right_out.begin());

  return true;
}

RDR_NAMESPACE_END

#else

#include "rdr/karras_bvh_gpu.h"
RDR_NAMESPACE_BEGIN
bool karras_bvh_gpu_build(const std::vector<Vec3f> &, const Vec3f &,
    const Vec3f &, std::vector<uint32_t> &, std::vector<int> &,
    std::vector<int> &, std::vector<int> &) noexcept {
  return false;
}
RDR_NAMESPACE_END

#endif
