// You might wonder why we need this while having an existing bvh_accel
// The abstraction level of these two are different. bvh_tree as a generic class
// can be used to implement bvh_accel, but bvh_accel itself can only encapsulate
// TriangleMesh thus cannot be used with bvhtree.
#ifndef __BVH_TREE_H__
#define __BVH_TREE_H__

#include <algorithm>

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
  EHeuristicProfile hprofile{EHeuristicProfile::EMedianHeuristic};

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
#if defined(USE_CUDA)
  {
    const int N = nodes.size();
    if (N == 0) goto cpu_fallback;

    std::vector<Vec3f> centroids(N);
    AABB global;
    for (int i = 0; i < N; i++) {
      auto c       = nodes[i].getAABB().getCenter();
      centroids[i] = c;
      global.unionWith(nodes[i].getAABB());
    }

    // GPU LBVH output
    std::vector<int> left_idx, right_idx, parent_idx, sorted_index;

    bool ok = karras_bvh_gpu_build(centroids, global.pMin, global.pMax,
        sorted_index,  // permutation of primitives (size N)
        left_idx,      // size 2N-1
        right_idx, parent_idx);

    if (!ok) goto cpu_fallback;

    const int total_nodes = left_idx.size();
    internal_nodes.resize(total_nodes);

    // Build internal node topology
    for (int i = 0; i < total_nodes; i++) {
      auto &nd       = internal_nodes[i];
      nd.left_index  = left_idx[i];
      nd.right_index = right_idx[i];
      nd.parent      = parent_idx[i];

      // Leaf nodes: index >= N-1
      if (i >= N - 1) {
        nd.is_leaf    = true;
        int leaf_id   = i - (N - 1);
        nd.span_left  = sorted_index[leaf_id];
        nd.span_right = nd.span_left + 1;
      } else {
        nd.is_leaf = false;
      }
    }

    // Recompute AABB bottom-up
    for (int i = total_nodes - 1; i >= 0; i--) {
      auto &nd = internal_nodes[i];
      if (nd.is_leaf) {
        nd.aabb = nodes[nd.span_left].getAABB();
      } else {
        AABB box;
        if (nd.left_index >= 0)
          box.unionWith(internal_nodes[nd.left_index].aabb);
        if (nd.right_index >= 0)
          box.unionWith(internal_nodes[nd.right_index].aabb);
        nd.aabb = box;
      }
    }

    // Root = last internal node
    root_index = total_nodes - 1;
    is_built   = true;
    return;
  }
#endif
cpu_fallback:
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
