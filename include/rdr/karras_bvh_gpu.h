#pragma once

#include <cstdint>
#include <vector>

#include "rdr/math_aliases.h"
#include "rdr/platform.h"


RDR_NAMESPACE_BEGIN

// Build LBVH topology on GPU using Karras (2012) algorithm.
// centers: per-primitive centroid positions (host)
// bb_min, bb_extent: bounding box for normalization
// Outputs:
//  - morton_out: morton code per primitive (sorted order)
//  - indices_out: permutation of primitive indices after sorting
//  - left_out/right_out: arrays of length n-1 for internal nodes, child indices
// Returns true on success (CUDA available and build succeeded).
bool karras_bvh_gpu_build(const std::vector<Vec3f> &centers,
    const Vec3f &bb_min, const Vec3f &bb_extent,
    std::vector<uint32_t> &morton_out, std::vector<int> &indices_out,
    std::vector<int> &left_out, std::vector<int> &right_out) noexcept;

RDR_NAMESPACE_END
