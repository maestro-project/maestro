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

#ifndef MAESTRO_CA_ITERATION_HPP_
#define MAESTRO_CA_ITERATION_HPP_

#include <string>
#include <memory>
#include <cmath>
#include <vector>
#include <map>
#include <list>
#include <utility>

#include "BASE_maestro-class.hpp"

#include "DFA_directives.hpp"
#include "DFA_tensor.hpp"

#include "CA_analysis-types.hpp"

namespace maestro {
  namespace CA {


    class IterationStatusInfo : public MAESTROClass {
      public:
        IterationStatusInfo() :
          MAESTROClass("IterationStatusInfo") {
        }

        void AddIterationInfo(std::pair<std::string, std::shared_ptr<std::list<CA::IterationStatus>>> iteration_info) {
          possible_status_.push_back(iteration_info);
        }

      protected:
        std::vector<std::pair<std::string, std::shared_ptr<std::list<CA::IterationStatus>>>> possible_status_;

    };


    class IterationStatusInstance : public MAESTROClass {
      public:
        IterationStatusInstance() :
          MAESTROClass("IterationStatusInstance") {
        }



      protected:
        std::map<std::string, CA::IterationStatus> status_;

    };


    class UnitIterationAnalysis {
      public:
        UnitIterationAnalysis(std::shared_ptr<DFA::directive::Directive> dataflow_directive, int dim_bound, int num_clusters)
          : dataflow_directive_(dataflow_directive), dimension_bound_(dim_bound), num_targ_clusters_(num_clusters) {
        }

        int GetNumTemporalIteration() {
          int ret = 1;
          if(dataflow_directive_->GetClass() == DFA::directive::DirectiveClass::TemporalMap) {
            ret = ceil(static_cast<double>(dimension_bound_ - dataflow_directive_->GetSize() +1 ) / static_cast<double>(dataflow_directive_->GetOfs()));
          }
          return ret;
        }

        int GetNumSpatialIteration() {
          int ret = 1;
          if(dataflow_directive_->GetClass() == DFA::directive::DirectiveClass::SpatialMap) {
            ret = ceil(static_cast<double>(dimension_bound_ - dataflow_directive_->GetSize() +1 ) / static_cast<double>(dataflow_directive_->GetOfs() * num_targ_clusters_));
          }
          return ret;
        }

        int GetNumSpatialEdgeClusters() {
          return (dimension_bound_ - dataflow_directive_->GetSize() +1 ) % (dataflow_directive_->GetOfs() * num_targ_clusters_);
        }

      protected:
        std::shared_ptr<DFA::directive::Directive> dataflow_directive_;
        int dimension_bound_;
        int num_targ_clusters_;
    }; // End of class UnitIterationAnalysis


  }; // End of namespace CA
}; // End of namespace maestro

#endif
