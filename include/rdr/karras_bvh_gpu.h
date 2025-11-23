#pragma once
#ifdef USE_CUDA

#include <vector>

#include "rdr/math_types.h"  // assuming Vec3f is defined here; adjust include as needed
#include "rdr/platform.h"

RDR_NAMESPACE_BEGIN

bool karras_bvh_gpu_build(const std::vector<Vec3f> &centroids,
    const Vec3f &scene_min, const Vec3f &scene_max,
    std::vector<uint32_t> &out_sorted_indices, std::vector<int> &out_parent,
    std::vector<int> &out_left, std::vector<int> &out_right) noexcept;

RDR_NAMESPACE_END

#endif