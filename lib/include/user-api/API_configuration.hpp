/******************************************************************************
Copyright (c) 2019 Georgia Instititue of Technology
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author : Hyoukjun Kwon (hyoukjun@gatech.edu)
*******************************************************************************/


#ifndef API_CONFIGURATION_HPP_
#define API_CONFIGURATION_HPP_

#include <memory>
#include <vector>

//#include "option.hpp"

#include "BASE_base-objects.hpp"

#include "DFA_tensor.hpp"
#include "DFA_neural-network.hpp"
#include "DFA_cluster-analysis.hpp"

#include "DSE_hardware_modules.hpp"

#include "AHW-noc-model.hpp"



namespace maestro {

  class Configuration {

    public:
      Configuration(int num_pes, int vector_width, int top_noc_bw, int l1_sram_byte_size, int l2_sram_byte_size)
      : num_pes_(num_pes) {
        network_= std::make_shared<DFA::NeuralNetwork>();
        tensors_ = std::make_shared<DFA::TensorTable>();
        nocs_ = std::make_shared<std::vector<std::shared_ptr<abstract_hw::NetworkOnChipModel>>>();
        cluster_analysis_= std::make_shared<std::vector<std::shared_ptr<DFA::ClusterAnalysis>>>();
        target_accelerator_ = std::make_shared<DSE::Accelerator>
                                (num_pes, vector_width, top_noc_bw, l1_sram_byte_size, l2_sram_byte_size);


      }
/*
      Accelerator (int num_pes, int vector_width, int noc_bw, int l1_sram_byte_size, int l2_sram_byte_size)
        : num_pes_(num_pes), vector_width_(vector_width), noc_bw_(noc_bw) {
        ReconstructAccelerator(num_pes, vector_width, noc_bw, l1_sram_byte_size, l2_sram_byte_size);
      }
*/
      std::shared_ptr<DFA::NeuralNetwork> network_;
      std::shared_ptr<DFA::TensorTable> tensors_;
      std::shared_ptr<std::vector<std::shared_ptr<DFA::ClusterAnalysis>>> cluster_analysis_;

      //Hardware
      std::shared_ptr<DSE::Accelerator> target_accelerator_;
      std::shared_ptr<std::vector<std::shared_ptr<abstract_hw::NetworkOnChipModel>>> nocs_;

      int num_pes_;
  }; // End of class Configuration

}; // End of namespace maestro


#endif
