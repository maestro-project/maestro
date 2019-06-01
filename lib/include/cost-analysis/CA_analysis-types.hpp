#ifndef MAESTRO_CA_ANALYSIS_TYPES_HPP_
#define MAESTRO_CA_ANALYSIS_TYPES_HPP_


namespace maestro {
  namespace CA {

    enum class IterationStatus {Init, Steady, Edge, Unroll, InitEdge, InitUnroll, InitEdgeSpEdge, EdgeSpEdge, NumIterStatus};
    enum class EstimationType {Min, Max, Exact};
    const int num_iter_status = 3;

  };
};

#endif
