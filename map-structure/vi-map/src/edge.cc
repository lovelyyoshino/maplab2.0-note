#include "vi-map/edge.h"

#include <aslam/common/memory.h>

#include "vi-map/cklam-edge.h"
#include "vi-map/loopclosure-edge.h"
#include "vi-map/transformation-edge.h"
#include "vi-map/vi_map.pb.h"
#include "vi-map/viwls-edge.h"

namespace vi_map {

void Edge::setId(const pose_graph::EdgeId& id) {
  CHECK(id.isValid());
  id_ = id;
}

void Edge::setFrom(const pose_graph::VertexId& from) {
  CHECK(from.isValid());
  from_ = from;
}

void Edge::setTo(const pose_graph::VertexId& to) {
  CHECK(to.isValid());
  to_ = to;
}

void Edge::serialize(vi_map::proto::Edge* proto) const {
  CHECK_NOTNULL(proto);

  if (getType() == pose_graph::Edge::EdgeType::kViwls) {
    const vi_map::ViwlsEdge& derived_edge =
        static_cast<const vi_map::ViwlsEdge&>(*this);
    derived_edge.serialize(proto->mutable_viwls());
  } else if (getType() == pose_graph::Edge::EdgeType::kOdometry) {
    const vi_map::TransformationEdge& derived_edge =
        static_cast<const vi_map::TransformationEdge&>(*this);
    derived_edge.serialize(proto->mutable_odometry());
  } else if (getType() == pose_graph::Edge::EdgeType::kLoopClosure) {
    const vi_map::LoopClosureEdge& derived_edge =
        static_cast<const vi_map::LoopClosureEdge&>(*this);
    derived_edge.serialize(proto->mutable_loopclosure());
  } else if (getType() == pose_graph::Edge::EdgeType::kWheelOdometry) {
    const vi_map::TransformationEdge& derived_edge =
        static_cast<const vi_map::TransformationEdge&>(*this);
    derived_edge.serialize(proto->mutable_wheel_odometry());
  } else {
    LOG(FATAL) << "Unknown edge type.";
  }
}

Edge::UniquePtr Edge::deserialize(
    const pose_graph::EdgeId& edge_id, const vi_map::proto::Edge& proto) {
  if (proto.has_viwls()) {
    vi_map::ViwlsEdge* edge(new vi_map::ViwlsEdge());
    edge->deserialize(edge_id, proto.viwls());
    return Edge::UniquePtr(edge);
  } else if (proto.has_odometry()) {
    vi_map::TransformationEdge* edge(
        new vi_map::TransformationEdge(vi_map::Edge::EdgeType::kOdometry));
    edge->deserialize(edge_id, proto.odometry());
    return Edge::UniquePtr(edge);
  } else if (proto.has_loopclosure()) {
    vi_map::LoopClosureEdge* edge(new vi_map::LoopClosureEdge());
    edge->deserialize(edge_id, proto.loopclosure());
    return Edge::UniquePtr(edge);
  } else if (proto.has_wheel_odometry()) {
    vi_map::TransformationEdge* edge(
        new vi_map::TransformationEdge(vi_map::Edge::EdgeType::kWheelOdometry));
    edge->deserialize(edge_id, proto.wheel_odometry());
    return Edge::UniquePtr(edge);
  } else {
    LOG(FATAL) << "Unknown edge type.";
    return nullptr;
  }
}

void Edge::copyEdgeInto(Edge** new_edge) const {
  CHECK_NOTNULL(new_edge);

  switch (edge_type_) {
    // TODO(ben): is kOdometry being used at alL?
    case pose_graph::Edge::EdgeType::kOdometry:
    case pose_graph::Edge::EdgeType::kWheelOdometry: {
      copyEdge<TransformationEdge>(new_edge);
      break;
    }
    case pose_graph::Edge::EdgeType::kLoopClosure: {
      copyEdge<LoopClosureEdge>(new_edge);
      break;
    }
    case pose_graph::Edge::EdgeType::kViwls: {
      copyEdge<ViwlsEdge>(new_edge);
      break;
    }
    case pose_graph::Edge::EdgeType::kCklamImuLandmark: {
      copyEdge<CklamEdge>(new_edge);
      break;
    }
    case pose_graph::Edge::EdgeType::kUndefined:
    default: {
      LOG(FATAL) << "Edge type " << static_cast<int>(edge_type_)
                 << " is not supported by deep copy.";
    }
  }

  CHECK(new_edge != nullptr);
}

}  // namespace vi_map
