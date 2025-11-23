#include "rdr/karras_bvh_gpu.h"

#if defined(USE_CUDA)

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

RDR_NAMESPACE_BEGIN

// Utility: expand 10 bits to interleaved Morton bits
__device__ __host__ inline uint32_t expandBits(uint32_t v) {
  // expand 10-bit integer into 30 bits with 2 zeros between bits
  v = (v | (v << 16)) & 0x030000FF;
  v = (v | (v << 8)) & 0x0300F00F;
  v = (v | (v << 4)) & 0x030C30C3;
  v = (v | (v << 2)) & 0x09249249;
  return v;
}

__device__ __host__ inline uint32_t morton3D(
    const float x, const float y, const float z) {
  // input x,y,z in [0,1]
  const float fx = fminf(fmaxf(x * 1023.0f, 0.0f), 1023.0f);
  const float fy = fminf(fmaxf(y * 1023.0f, 0.0f), 1023.0f);
  const float fz = fminf(fmaxf(z * 1023.0f, 0.0f), 1023.0f);
  uint32_t xx    = static_cast<uint32_t>(fx);
  uint32_t yy    = static_cast<uint32_t>(fy);
  uint32_t zz    = static_cast<uint32_t>(fz);
  return (expandBits(xx) << 2) | (expandBits(yy) << 1) | expandBits(zz);
}

// count leading zeros for 32-bit on device/host
__device__ __host__ inline int clz32(uint32_t x) {
#if defined(__CUDA_ARCH__)
  return __clz(x);
#else
  if (x == 0) return 32;
  return __builtin_clz(x);
#endif
}

// Kernel: compute Morton codes
__global__ void compute_morton_kernel(const float *centroid_x,
    const float *centroid_y, const float *centroid_z, const float minx,
    const float miny, const float minz, const float invsx, const float invsy,
    const float invsz, uint32_t *out_codes, uint32_t *out_indices, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  float nx         = (centroid_x[idx] - minx) * invsx;
  float ny         = (centroid_y[idx] - miny) * invsy;
  float nz         = (centroid_z[idx] - minz) * invsz;
  uint32_t code    = morton3D(nx, ny, nz);
  out_codes[idx]   = code;
  out_indices[idx] = idx;
}

// Device helper to compute common prefix length for two Morton codes
__device__ inline int common_prefix(
    const uint32_t *codes, int N, int i, int j) {
  if (j < 0 || j >= N) return -1;
  uint32_t a = codes[i];
  uint32_t b = codes[j];
  uint32_t x = a ^ b;
  if (x == 0)
    return 32 +
           clz32(i ^
                 j);  // tie-breaker: include index bits to order equal mortons
  return clz32(x);
}

// Kernel: build internal nodes following Karras algorithm
__global__ void build_internal_nodes_kernel(const uint32_t *codes, int N,
    int *out_left, int *out_right, int *out_parent) {
  // Each thread constructs one internal node in [0 .. N-2]
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N - 1) return;

  // helper lambda: delta(i, j) = commonPrefix(i, j)
  auto delta = [&](int i, int j) {
    uint32_t a = codes[i];
    uint32_t b = codes[j];
    uint32_t x = a ^ b;
    if (x == 0) {
      // identical morton codes: use index difference to break tie
      return 32 + clz32(static_cast<uint32_t>(i ^ j));
    } else {
      return clz32(x);
    }
  };

  // determine direction
  int d = (delta(idx, idx + 1) - delta(idx, idx - 1)) > 0 ? 1 : -1;

  // find upper bound for range length using exponential search
  int l    = 0;
  int step = 1;
  while (true) {
    int j = idx + d * step;
    if (j < 0 || j >= N) break;
    if (delta(idx, j) < delta(idx, idx - d)) break;
    l = step;
    step <<= 1;
  }

  // binary search to find other end
  int low  = 0;
  int high = l;
  while (low < high) {
    int mid = (low + high + 1) >> 1;
    int j   = idx + d * mid;
    if (j < 0 || j >= N) {
      high = mid - 1;
      continue;
    }
    if (delta(idx, j) >= delta(idx, idx - d))
      low = mid;
    else
      high = mid - 1;
  }

  int j = idx + d * low;

  // find split position between idx and j
  // find first k in (min(idx,j), max(idx,j)] such that delta(idx, k) >
  // delta(idx, k-1)
  int first = min(idx, j);
  int last  = max(idx, j);

  int split = first;
  int lo    = first;
  int hi    = last;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (delta(idx, mid) > delta(idx, mid + 1)) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  split = lo;

  // child assignment: left child covers [first, split]; right covers [split+1,
  // last]
  int left_child  = (split == first) ? ((N - 1) + first)
                                     : split;  // leaf index or internal node
  int right_child = (split + 1 == last) ? ((N - 1) + (split + 1)) : (split + 1);

  // store children (internal nodes indices are in [0..N-2], leaves in
  // [N-1..2N-2])
  out_left[idx]  = left_child;
  out_right[idx] = right_child;

  // set parent for children (atomic because multiple threads may write same
  // parent)
  atomicExch(&out_parent[left_child], idx);
  atomicExch(&out_parent[right_child], idx);

  // parent of this internal node will be set by whoever owns it; initialize if
  // unset (we may set root parent to -1 later)
}

bool karras_bvh_gpu_build(const std::vector<Vec3f> &centroids,
    const Vec3f &scene_min, const Vec3f &scene_max,
    std::vector<uint32_t> &out_sorted_indices, std::vector<int> &out_parent,
    std::vector<int> &out_left, std::vector<int> &out_right) noexcept {
  using uint = unsigned int;
  int N      = static_cast<int>(centroids.size());
  if (N == 0) {
    out_sorted_indices.clear();
    out_parent.clear();
    out_left.clear();
    out_right.clear();
    return true;
  }
  // allocate device arrays for centroid components
  thrust::host_vector<float> hx(N), hy(N), hz(N);
  for (int i = 0; i < N; ++i) {
    hx[i] = centroids[i].x;
    hy[i] = centroids[i].y;
    hz[i] = centroids[i].z;
  }
  thrust::device_vector<float> dx = hx;
  thrust::device_vector<float> dy = hy;
  thrust::device_vector<float> dz = hz;

  thrust::device_vector<uint32_t> d_codes(N);
  thrust::device_vector<uint32_t> d_indices(N);

  // compute normalization
  float minx = scene_min.x, miny = scene_min.y, minz = scene_min.z;
  float maxx = scene_max.x, maxy = scene_max.y, maxz = scene_max.z;
  float invsx = 1.0f / fmaxf(maxx - minx, 1e-9f);
  float invsy = 1.0f / fmaxf(maxy - miny, 1e-9f);
  float invsz = 1.0f / fmaxf(maxz - minz, 1e-9f);

  // launch compute morton kernel
  {
    const int block = 256;
    const int grid  = (N + block - 1) / block;
    compute_morton_kernel<<<grid, block>>>(thrust::raw_pointer_cast(dx.data()),
        thrust::raw_pointer_cast(dy.data()),
        thrust::raw_pointer_cast(dz.data()), minx, miny, minz, invsx, invsy,
        invsz, thrust::raw_pointer_cast(d_codes.data()),
        thrust::raw_pointer_cast(d_indices.data()), N);
    cudaDeviceSynchronize();
  }

  // sort by morton keys using thrust (stable)
  thrust::sort_by_key(d_codes.begin(), d_codes.end(), d_indices.begin());

  // copy sorted indices to host output
  out_sorted_indices.resize(N);
  thrust::copy(d_indices.begin(), d_indices.end(), out_sorted_indices.begin());

  // Prepare arrays for nodes:
  // total nodes M = 2*N - 1, internal 0..N-2, leaves N-1..2N-2
  int M = 2 * N - 1;
  out_parent.assign(M, -1);
  out_left.assign(M, -1);
  out_right.assign(M, -1);

  // copy codes to contiguous device array (thrust vector d_codes already
  // contains sorted codes) We need sorted codes present for build stage while
  // using the indices order d_codes already sorted after sort_by_key Launch
  // kernel to build internal nodes
  {
    // allocate device-side children/parent arrays
    thrust::device_vector<int> d_left(N - 1, -1);
    thrust::device_vector<int> d_right(N - 1, -1);
    thrust::device_vector<int> d_parent(M, -1);

    const int block = 256;
    const int grid  = ((N - 1) + block - 1) / block;
    build_internal_nodes_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_codes.data()), N,
        thrust::raw_pointer_cast(d_left.data()),
        thrust::raw_pointer_cast(d_right.data()),
        thrust::raw_pointer_cast(d_parent.data()));
    cudaDeviceSynchronize();

    // copy internal nodes children into out_left/out_right at positions
    // [0..N-2]
    thrust::copy(d_left.begin(), d_left.end(), out_left.begin());
    thrust::copy(d_right.begin(), d_right.end(), out_right.begin());

    // copy parents back: parent array includes parents for internal nodes &
    // leaves
    thrust::copy(d_parent.begin(), d_parent.end(), out_parent.begin());
  }

  // find root: root is the internal node whose parent == -1. Set parent[root]
  // == -1 already If parent array left -1 for internal nodes that belong to
  // root, it's already fine. Ensure leaves have their parent set: kernels wrote
  // parent for leaves via atomicExch.

  return true;
}

RDR_NAMESPACE_END

#else

// stub if not compiled with CUDA
RDR_NAMESPACE_BEGIN
bool karras_bvh_gpu_build(const std::vector<Vec3f> &, const Vec3f &,
    const Vec3f &, std::vector<uint32_t> &, std::vector<int> &,
    std::vector<int> &, std::vector<int> &) noexcept {
  return false;
}
RDR_NAMESPACE_END

#endif