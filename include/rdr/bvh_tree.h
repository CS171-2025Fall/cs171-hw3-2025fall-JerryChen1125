// You might wonder why we need this while having an existing bvh_accel
// The abstraction level of these two are different. bvh_tree as a generic class
// can be used to implement bvh_accel, but bvh_accel itself can only encapsulate
// TriangleMesh thus cannot be used with bvhtree.
#ifndef __BVH_TREE_H__
#define __BVH_TREE_H__

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "rdr/accel.h"
#include "rdr/platform.h"
#include "rdr/primitive.h"
#include "rdr/ray.h"
#if defined(USE_CUDA)
#include "rdr/karras_bvh_gpu.h"
#endif

RDR_NAMESPACE_BEGIN

template <typename DataType_>
class BVHNodeInterface {
public:
  using DataType = DataType_;

  // The only two required interfaces
  virtual AABB getAABB() const            = 0;
  virtual const DataType &getData() const = 0;

protected:
  // Interface spec
  BVHNodeInterface()  = default;
  ~BVHNodeInterface() = default;

  BVHNodeInterface(const BVHNodeInterface &)            = default;
  BVHNodeInterface &operator=(const BVHNodeInterface &) = default;
};

// TODO: check derived class's type
template <typename NodeType_>
class BVHTree final {
public:
  using NodeType  = NodeType_;
  using IndexType = int;

  // Context-local
  constexpr static int INVALID_INDEX = -1;
  constexpr static int CUTOFF_DEPTH  = 22;

  enum class EHeuristicProfile {
    EMedianHeuristic      = 0,  ///<! use centroid[depth%3]
    ESurfaceAreaHeuristic = 1,  ///<! use SAH (see PBRT)
    EParallelKarras       = 2,  ///<! use Karras (2012) parallel build
  };

  // The actual node that represents the tree structure
  struct InternalNode {
    InternalNode() = default;
    InternalNode(IndexType span_left, IndexType span_right)
        : span_left(span_left), span_right(span_right) {}

    bool is_leaf{false};
    IndexType left_index{INVALID_INDEX};
    IndexType right_index{INVALID_INDEX};
    IndexType span_left{INVALID_INDEX};
    IndexType span_right{INVALID_INDEX};  // nodes[span_left, span_right)
    AABB aabb{};                          // The bounding box of the node
  };

  BVHTree()  = default;
  ~BVHTree() = default;

  /// General Interface
  size_t size() { return nodes.size(); }

  /// Nodes might be re-ordered
  void push_back(const NodeType &node) { nodes.push_back(node); }
  const AABB &getAABB() const { return internal_nodes[root_index].aabb; }

  /// reset build status
  void clear();

  /// *Can* be executed not only once
  void build();

  template <typename Callback>
  bool intersect(Ray &ray, Callback callback) const {
    if (!is_built) return false;
    return intersect(ray, root_index, callback);
  }

private:
  EHeuristicProfile hprofile{EHeuristicProfile::EParallelKarras};

  bool is_built{false};
  IndexType root_index{INVALID_INDEX};

  vector<NodeType> nodes{};               /// The data nodes
  vector<InternalNode> internal_nodes{};  /// The internal nodes

  /// Internal build
  IndexType build(
      int depth, const IndexType &span_left, const IndexType &span_right);

  /// Internal intersect
  template <typename Callback>
  bool intersect(
      Ray &ray, const IndexType &node_index, Callback callback) const;
};

/* ===================================================================== *
 *
 * Implementation
 *
 * ===================================================================== */

template <typename _>
void BVHTree<_>::clear() {
  nodes.clear();
  internal_nodes.clear();
  is_built = false;
}

template <typename _>
void BVHTree<_>::build() {
  if (is_built) return;
  // pre-allocate memory
  internal_nodes.reserve(2 * nodes.size());
  root_index = build(0, 0, nodes.size());
  is_built   = true;
}

template <typename _>
typename BVHTree<_>::IndexType BVHTree<_>::build(
    int depth, const IndexType &span_left, const IndexType &span_right) {
  if (span_left >= span_right) return INVALID_INDEX;

  // early calculate bound
  AABB prebuilt_aabb;
  for (IndexType span_index = span_left; span_index < span_right; ++span_index)
    prebuilt_aabb.unionWith(nodes[span_index].getAABB());

  // TODO(HW3): setup the stop criteria
  //
  // You should fill in the stop criteria here.
  //
  // You may find the following variables useful:
  //
  // @see CUTOFF_DEPTH: The maximum depth you would like to build
  // @see span_left: The left index of the current span
  // @see span_right: The right index of the current span
  //
  if (depth >= CUTOFF_DEPTH || span_right - span_left <= 1) {
    // create leaf node
    const auto &node = nodes[span_left];
    InternalNode result(span_left, span_right);
    result.is_leaf = true;
    result.aabb    = prebuilt_aabb;
    internal_nodes.push_back(result);
    return internal_nodes.size() - 1;
  }

  // You'll notice that the implementation here is different from the KD-Tree
  // ones, which re-use the node for both data-storing and organizing the real
  // tree structure. Here, for simplicity and generality, we use two different
  // types of nodes to ensure simplicity in interface, i.e. provided node does
  // not need to be aware of the tree structure.
  InternalNode result(span_left, span_right);

  // const int &dim = depth % 3;
  const int &dim  = ArgMax(prebuilt_aabb.getExtent());
  IndexType count = span_right - span_left;
  IndexType split = INVALID_INDEX;

  if (hprofile == EHeuristicProfile::EMedianHeuristic) {
use_median_heuristic:
    split = span_left + count / 2;
    // Sort the nodes
    // after which, all centroids in [span_left, split) are LT than right
    // clang-format off

    // TODO(HW3): implement the median split here
    //
    // You should sort the nodes in [span_left, span_right) according to
    // their centroid's `dim`-th dimension, such that all nodes in
    // [span_left, split) are less than those in [split, span_right)
    //
    // You may find `std::nth_element` useful here.

    // Partition nodes by median along chosen dimension using centroids
    auto mid_it = nodes.begin() + split;
    std::nth_element(nodes.begin() + span_left, mid_it, nodes.begin() + span_right,
        [dim](const NodeType &a, const NodeType &b) {
          return a.getAABB().getCenter()[dim] < b.getAABB().getCenter()[dim];
        });

    // clang-format on
  } else if (hprofile == EHeuristicProfile::ESurfaceAreaHeuristic) {
use_surface_area_heuristic:
    // See
    // https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
    // for algorithm details. In general, like *decision tree*, we evaluate our
    // split with some measures.
    // Briefly speaking, "for a convex volume A *contained* in another larger
    // convex volume B , the conditional probability that a uniformly
    // distributed random ray passing through B will also pass through A is the
    // ratio of their surface areas"

    // TODO (BONUS): implement Surface area heuristic here
    //
    // You can then set @see BVHTree::hprofile to ESurfaceAreaHeuristic to
    // enable this feature.
    UNIMPLEMENTED;
  }

  // If Karras parallel profile is selected, fallback to a dedicated
  // parallel builder which constructs a LBVH using Morton codes. We only
  // execute it at the top level (span covering all nodes).
  if (hprofile == EHeuristicProfile::EParallelKarras && span_left == 0 &&
      span_right == (IndexType)nodes.size()) {
    // Karras (2012) style LBVH builder (CPU-parallel version).
    const int n = (int)nodes.size();
    if (n == 0) return INVALID_INDEX;

    // 1) Compute centroid bounds and Morton codes
    AABB centroid_bb;
    vector<Vec3f> centroids;
    centroids.reserve(n);
    for (int i = 0; i < n; ++i) {
      Vec3f c = nodes[i].getAABB().getCenter();
      centroids.push_back(c);
      centroid_bb.unionWith(AABB(c, c));
    }

    // helper to compute leading zeros portable
#if defined(_MSC_VER)
    auto clz64 = [](uint64_t x) -> int {
      if (x == 0) return 64;
      unsigned long index = 0;
      _BitScanReverse64(&index, x);
      return 63 - (int)index;
    };
#else
    auto clz64 = [](uint64_t x) -> int {
      return x == 0 ? 64 : __builtin_clzll(x);
    };
#endif

    // map centroid to Morton code (10 bits per axis -> 30 bits)
    auto expandBits = [](uint32_t v) {
      uint32_t x = v & 0x3ff;  // 10 bits
      x          = (x | (x << 16)) & 0x30000FF;
      x          = (x | (x << 8)) & 0x300F00F;
      x          = (x | (x << 4)) & 0x30C30C3;
      x          = (x | (x << 2)) & 0x9249249;
      return x;
    };

    vector<uint32_t> morton(n);
    // Normalize and compute
    Vec3f extent = centroid_bb.getExtent();
    for (int i = 0; i < n; ++i) {
      Vec3f c = centroids[i];
      Vec3f norm;
      for (int k = 0; k < 3; ++k) {
        if (extent[k] <= 0)
          norm[k] = 0.5f;
        else
          norm[k] = (c[k] - centroid_bb.low_bnd[k]) / extent[k];
      }
      uint32_t xi = std::min<uint32_t>(1023u, (uint32_t)(norm.x * 1023.0f));
      uint32_t yi = std::min<uint32_t>(1023u, (uint32_t)(norm.y * 1023.0f));
      uint32_t zi = std::min<uint32_t>(1023u, (uint32_t)(norm.z * 1023.0f));
      uint32_t code =
          (expandBits(xi) << 2) | (expandBits(yi) << 1) | expandBits(zi);
      morton[i] = code;
    }

    // 2) Sort primitives by Morton code and build topology
    vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

#if defined(USE_CUDA)
    // Try GPU builder: it will compute Morton (we already did host morton),
    // sort and produce left/right child arrays. We pass centroids and bbox.
    vector<int> leftChildGPU;
    vector<int> rightChildGPU;
    vector<uint32_t> morton_out;
    vector<int> indices_out;
    bool gpu_ok = karras_bvh_gpu_build(centroids, centroid_bb.low_bnd,
        centroid_bb.getExtent(), morton_out, indices_out, leftChildGPU,
        rightChildGPU);
    if (gpu_ok) {
      // reorder nodes according to GPU-sorted indices
      vector<NodeType> nodes_sorted;
      nodes_sorted.reserve(n);
      for (int i = 0; i < n; ++i) nodes_sorted.push_back(nodes[indices_out[i]]);
      nodes.swap(nodes_sorted);

      // use GPU-produced children below
      const int internal_count = n - 1;
      const int total_nodes    = 2 * n - 1;
      internal_nodes.clear();
      internal_nodes.resize(total_nodes);

      // initialize leaves
      for (int k = 0; k < n; ++k) {
        int idx            = (n - 1) + k;
        InternalNode &leaf = internal_nodes[idx];
        leaf.is_leaf       = true;
        leaf.span_left     = k;
        leaf.span_right    = k + 1;
        leaf.aabb          = nodes[k].getAABB();
      }

      // set internal children indices
      for (int i = 0; i < internal_count; ++i) {
        InternalNode &it = internal_nodes[i];
        it.is_leaf       = false;
        it.left_index    = leftChildGPU[i];
        it.right_index   = rightChildGPU[i];
      }

      // build parent array to find the root and prepare for span/AABB
      vector<int> parent(total_nodes, -1);
      for (int i = 0; i < internal_count; ++i) {
        int lc = internal_nodes[i].left_index;
        int rc = internal_nodes[i].right_index;
        if (lc >= 0 && lc < total_nodes) parent[lc] = i;
        if (rc >= 0 && rc < total_nodes) parent[rc] = i;
      }

      int root = -1;
      for (int i = 0; i < internal_count; ++i)
        if (parent[i] == -1) {
          root = i;
          break;
        }
      if (root == -1) root = 0;

      // compute span_left/span_right for internal nodes via post-order
      // traversal starting from root. Leaves already have spans set.
      vector<int> stack;
      stack.reserve(total_nodes);
      stack.push_back(root);
      vector<char> visited(total_nodes, 0);
      while (!stack.empty()) {
        int cur = stack.back();
        if (visited[cur]) {
          stack.pop_back();
          if (!internal_nodes[cur].is_leaf) {
            const InternalNode &L =
                internal_nodes[internal_nodes[cur].left_index];
            const InternalNode &R =
                internal_nodes[internal_nodes[cur].right_index];
            internal_nodes[cur].span_left = std::min(L.span_left, R.span_left);
            internal_nodes[cur].span_right =
                std::max(L.span_right, R.span_right);
          }
        } else {
          visited[cur] = 1;
          if (!internal_nodes[cur].is_leaf) {
            stack.push_back(internal_nodes[cur].right_index);
            stack.push_back(internal_nodes[cur].left_index);
          }
        }
      }

      // compute AABB bottom-up using same post-order
      stack.clear();
      stack.push_back(root);
      std::fill(visited.begin(), visited.end(), 0);
      while (!stack.empty()) {
        int cur = stack.back();
        if (visited[cur]) {
          stack.pop_back();
          if (!internal_nodes[cur].is_leaf) {
            const InternalNode &L =
                internal_nodes[internal_nodes[cur].left_index];
            const InternalNode &R =
                internal_nodes[internal_nodes[cur].right_index];
            internal_nodes[cur].aabb = L.aabb;
            internal_nodes[cur].aabb.unionWith(R.aabb);
          }
        } else {
          visited[cur] = 1;
          if (!internal_nodes[cur].is_leaf) {
            stack.push_back(internal_nodes[cur].right_index);
            stack.push_back(internal_nodes[cur].left_index);
          }
        }
      }

      root_index = root;
      is_built   = true;
      return root_index;
    }
#endif

    // 3) allocate arrays for tree topology
    const int internal_count = n - 1;
    const int total_nodes    = 2 * n - 1;
    vector<int> leftChild(internal_count, -1);
    vector<int> rightChild(internal_count, -1);
    vector<int> parent(total_nodes, -1);

    // delta function: common prefix length
    auto delta = [&](int a, int b) -> int {
      if (b < 0 || b >= n) return -1;
      uint64_t x = (uint64_t)morton[a] ^ (uint64_t)morton[b];
      if (x == 0) return 64;  // identical codes
      return clz64(x);
    };

    // 4) parallel for each internal node i
#pragma omp parallel for schedule(static)
    for (int i = 0; i < internal_count; ++i) {
      // determine direction
      int d         = (delta(i, i + 1) - delta(i, i - 1) > 0) ? 1 : -1;
      int delta_min = delta(i, i - d);

      // find upper bound for range length
      int lmax = 1;
      while (delta(i, i + lmax * d) > delta_min) lmax <<= 1;

      // binary search to find other end j
      int l = 0;
      for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (delta(i, i + (l + t) * d) > delta_min) l += t;
      }
      int j = i + l * d;

      // find split position
      int delta_node = delta(i, j);
      int s          = 0;
      int dl         = std::abs(j - i);
      int t          = 1;
      while (t <= dl) t <<= 1;
      for (int step = t >> 1; step > 0; step >>= 1) {
        if (s + step <= dl && delta(i, i + (s + step) * d) > delta_node)
          s += step;
      }

      int gamma = i + s * d + std::min(d, 0);

      // assign children
      int a = std::min(i, j);
      int b = std::max(i, j);
      // left child
      if (a == gamma)
        leftChild[i] = (n - 1) + gamma;  // leaf index
      else
        leftChild[i] = gamma;  // internal index

      // right child
      if (b == gamma + 1)
        rightChild[i] = (n - 1) + (gamma + 1);
      else
        rightChild[i] = gamma + 1;

      // set parent pointers
      int lc     = leftChild[i];
      int rc     = rightChild[i];
      parent[lc] = i;
      parent[rc] = i;
    }

    // find root: internal node with no parent
    int root = -1;
    for (int i = 0; i < internal_count; ++i)
      if (parent[i] == -1) {
        root = i;
        break;
      }
    if (root == -1) root = 0;

    // 5) build internal_nodes array (size total_nodes)
    internal_nodes.clear();
    internal_nodes.resize(total_nodes);

    // initialize leaves
    for (int k = 0; k < n; ++k) {
      int idx            = (n - 1) + k;
      InternalNode &leaf = internal_nodes[idx];
      leaf.is_leaf       = true;
      leaf.span_left     = k;
      leaf.span_right    = k + 1;
      leaf.aabb          = nodes[k].getAABB();
    }

    // initialize internal children indices
    for (int i = 0; i < internal_count; ++i) {
      InternalNode &it = internal_nodes[i];
      it.is_leaf       = false;
      it.left_index    = leftChild[i];
      it.right_index   = rightChild[i];
    }

    // compute AABB bottom-up using stack
    // post-order traversal from root
    vector<int> stack;
    stack.reserve(total_nodes);
    stack.push_back(root);
    // iterative post-order using visited flag
    vector<char> visited(total_nodes, 0);
    while (!stack.empty()) {
      int cur = stack.back();
      if (visited[cur]) {
        stack.pop_back();
        if (!internal_nodes[cur].is_leaf) {
          const InternalNode &L =
              internal_nodes[internal_nodes[cur].left_index];
          const InternalNode &R =
              internal_nodes[internal_nodes[cur].right_index];
          internal_nodes[cur].aabb = L.aabb;
          internal_nodes[cur].aabb.unionWith(R.aabb);
        }
      } else {
        visited[cur] = 1;
        if (!internal_nodes[cur].is_leaf) {
          stack.push_back(internal_nodes[cur].right_index);
          stack.push_back(internal_nodes[cur].left_index);
        }
      }
    }

    root_index = root;
    is_built   = true;
    return root_index;
  }

  // Build the left and right subtree
  result.left_index  = build(depth + 1, span_left, split);
  result.right_index = build(depth + 1, split, span_right);

  // Iterative merge
  result.aabb = prebuilt_aabb;

  internal_nodes.push_back(result);
  return internal_nodes.size() - 1;
}

template <typename _>
template <typename Callback>
bool BVHTree<_>::intersect(
    Ray &ray, const IndexType &node_index, Callback callback) const {
  bool result      = false;
  const auto &node = internal_nodes[node_index];

  // Perform the actual pruning
  Float t_in, t_out;
  if (!node.aabb.intersect(ray, &t_in, &t_out)) return result;

  if (node.is_leaf) {
    for (IndexType span_index = node.span_left; span_index < node.span_right;
         ++span_index)
      result |= callback(ray, nodes[span_index].getData());
    return result;
  } else {
    // Recurse
    if (node.left_index != INVALID_INDEX)
      result |= intersect(ray, node.left_index, callback);
    if (node.right_index != INVALID_INDEX)
      result |= intersect(ray, node.right_index, callback);
    return result;
  }
}

RDR_NAMESPACE_END

#endif
