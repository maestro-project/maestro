#ifndef MAESTRO_AHW_ACCELERATOR_HPP_
#define MAESTRO_AHW_ACCELERATOR_HPP_

#include <vector>
#include <string>

#include "DSE_design-options.hpp"

namespace maestro {

  namespace AHW {

    class Accelerator {
      public:


      protected:
        int num_pes_;
        int pe_vector_width_;
        DSE::OpType mult_op_type_;
        int mult_precision_;
        DSE::OpType add_op_type_;
        int add_precision_;

        std::vector<std::string> buffer_name_;
        std::vector<int> buffer_sizes_;
        std::vector<std::string> noc_name_;
        std::vector<int> noc_bandwidth_;
        std::vector<int> noc_latency_;


    }; // End of class Accelerator


  }; // End of namespace AHW
}; // End of namespace maestro


#endif
