#ifndef POSEGRAPH_UNIQUE_ID_H_
#define POSEGRAPH_UNIQUE_ID_H_
#include <unordered_set>
#include <vector>

#include <aslam/common/unique-id.h>
#include <glog/logging.h>

namespace pose_graph {
UNIQUE_ID_DEFINE_ID(VertexId);
UNIQUE_ID_DEFINE_ID(EdgeId);
}  // namespace pose_graph

UNIQUE_ID_DEFINE_ID_HASH(pose_graph::VertexId);
UNIQUE_ID_DEFINE_ID_HASH(pose_graph::EdgeId);

#endif  // POSEGRAPH_UNIQUE_ID_H_
