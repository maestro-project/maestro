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


#ifndef CA_REUSE_ANALYSIS_HPP_
#define CA_REUSE_ANALYSIS_HPP_

//#define DEBUG_REUSE_ANALYSIS

#include <memory>
#include <map>
#include <cmath>

#include "BASE_maestro-class.hpp"
#include "TL_error-handler.hpp"

#include "DFSL_syntax_tokens.hpp"

#include "DFA_directives.hpp"
#include "DFA_tensor.hpp"
#include "DFA_cluster-unit.hpp"

#include "CA_analysis-types.hpp"


namespace  maestro {
  namespace CA {

    class ReuseAnalysis : public MAESTROClass {
      public:
        ReuseAnalysis (std::shared_ptr<DFA::ClusterUnit> target_cluster) :
          MAESTROClass("Reuse Analysis"),
          target_cluster_(target_cluster) {
          num_mapped_elements_ = std::make_unique<std::map<std::string, int>>();
          num_mapped_elements_tp_edge_ = std::make_unique<std::map<std::string, int>>();

          num_unique_elements_ = std::make_unique<std::map<std::string, int>>();
          num_unique_elements_tp_edge_ = std::make_unique<std::map<std::string, int>>();

          num_reused_elements_ = std::make_unique<std::map<std::string, int>>();
          num_reused_elements_tp_edge_ = std::make_unique<std::map<std::string, int>>();

          AnalyzeMappingSizes(target_cluster);
#ifdef DEBUG_REUSE_ANALYSIS
          for(auto it : *num_mapped_elements_ ){
            std::cout << "NumMapped_elements[" << it.first << "] = " <<  it.second << std::endl;
          }
          for(auto it : *num_mapped_elements_tp_edge_ ){
            std::cout << "NumMapped_elements_tp_edge[" << it.first << "] = " <<  it.second << std::endl;
          }

          for(auto it : *num_unique_elements_ ){
            std::cout << "num_unique_elements_[" << it.first << "] = " <<  it.second << std::endl;
          }

          for(auto it : *num_unique_elements_tp_edge_){
            std::cout << "num_unique_elements_tp_edge[" << it.first << "] = " <<  it.second << std::endl;
          }
#endif
        }

        long GetMappedVolume(std::shared_ptr<DFA::Tensor> tensor) {
          long ret = 1;

          auto corr_var_list = tensor->GetCoupledVariables();
          for(auto var: *corr_var_list){
            if(num_mapped_elements_->find(var) != num_mapped_elements_->end()) {
              ret *= num_mapped_elements_->at(var);
            }
          }

          return ret;
        }

        long GetSpatialLoadVolume(std::shared_ptr<DFA::Tensor> tensor,
                                  std::shared_ptr<std::vector<IterationStatus>> iter_status,
                                  CA::EstimationType estimation_type = CA::EstimationType::Exact) {

          bool is_sp_edge = IsSpatialEdge(iter_status);
          long num_clusters = target_cluster_->GetNumClusters(false);
          long num_sub_clusters = target_cluster_->GetNumClusters(is_sp_edge);
          long ret =  0;

          auto coupled_dims = tensor->GetCoupledVariables();

          bool is_sp_mapped = false;

          for(auto& dim : *coupled_dims) {
            auto dataflow = target_cluster_->GetDataflow();
            auto directive = dataflow->FindDirective(dim);
            if(directive != nullptr) {
              if(directive->GetClass() == DFA::directive::DirectiveClass::SpatialMap) {
                is_sp_mapped = true;
                break;
              }
            }
          }

          ret += GetPELoadVolume(tensor, iter_status, true);
          switch(estimation_type) {
            case CA::EstimationType::Exact: {
              if(num_sub_clusters > 1 && is_sp_mapped) { // && if tensor is spatially mapped
                ret += (num_sub_clusters-1) * GetPELoadVolume(tensor, iter_status, false);
              }
              break;
            }
            case CA::EstimationType::Max: {
              ret += (num_clusters-1) * GetPELoadVolume(tensor, iter_status, false);
              break;
            }
            default: {
              //Add nothing
            }
          }

          return ret;
        }

        long GetNumOutputs(
            std::shared_ptr<DFA::Tensor> output_tensor,
            std::shared_ptr<std::vector<CA::IterationStatus>> iter_status,
            bool get_num_partial_sum = false,
            bool is_sp_edge_pe = false)
        {
          long num_outputs = 1;

          auto dataflow = target_cluster_->GetDataflow();
          auto dimensions = target_cluster_->GetDimensions();
          auto coupled_var_list = output_tensor->GetCoupledVariables();


          /******/
          //New Parts
          for(auto& var: *coupled_var_list) {
            if(!get_num_partial_sum) {

            }
          }
          /******/


          int directive_id = 0;
          for(auto directive : *dataflow) {
            auto directive_var = directive->GetVariable();
            int original_map_size = directive->GetSize();
            int actual_map_size = original_map_size;
            auto iter_status_this_var = iter_status->at(directive_id);
/*

            if(dimensions->IsOverlapped(directive_var)) {
              if(dimensions->IsSlidingDim(directive_var)) {
                auto overlapping_dim = dimensions->GetOverlappingDim(directive_var);
                int reference_dim_size = dataflow->FindDirective(overlapping_dim)->GetSize();
                int sliding_dim_size = original_map_size;

                actual_map_size = (reference_dim_size < sliding_dim_size)? actual_map_size : 1;
              }
              else { // If directive var dimension is reference dimension, not sliding dimension
                auto overlapping_dim = dimensions->GetOverlappingDim(directive_var);
                int sliding_dim_size = dataflow->FindDirective(overlapping_dim)->GetSize();
                int reference_dim_size = original_map_size;
                int actual_reference_dim_size = reference_dim_size;
                //TODO: I don't think we need to deal with edge case on sliding dim size. Verify this
                actual_map_size = (reference_dim_size < sliding_dim_size)? actual_map_size : reference_dim_size - sliding_dim_size + 1;
              }
            }


            if(!get_num_partial_sum) {
              //If is coupled
              if(std::find(coupled_var_list->begin(), coupled_var_list->end(), directive_var) != coupled_var_list->end() ) {

                if((directive->GetClass() == DFA::directive::DirectiveClass::TemporalMap)) {
                  switch(iter_status_this_var) {
                    case IterationStatus::Edge: {
                      actual_map_size = dimensions->GetSize(directive_var) % directive->GetOfs();
                      break;
                    }
                    case IterationStatus::Init: {
                      actual_map_size = (target_cluster_->IsInitEdge(directive_var))? dimensions->GetSize(directive_var) : original_map_size;
                      break;
                    }
                    case IterationStatus::Unroll: {
                      actual_map_size = (target_cluster_->IsInitEdge(directive_var))? dimensions->GetSize(directive_var) : original_map_size;
                      break;
                    }
                    default: {
                      actual_map_size = original_map_size;
                      break;
                    }
                  }
                }
                else {
                  assert(directive->GetClass() == DFA::directive::DirectiveClass::SpatialMap);


                }

              }
            }

*/


              if((directive->GetClass() == DFA::directive::DirectiveClass::TemporalMap)) {
                if(iter_status_this_var == IterationStatus::Edge) {
//                  actual_map_size = dimensions->GetSize(directive_var) % directive->GetOfs(); // TODO: Verify this change
                  actual_map_size = dimensions->GetSize(directive_var) - (dimensions->GetSize(directive_var)/directive->GetOfs()) * directive->GetOfs();
                }
                else if(iter_status_this_var == IterationStatus::Init && target_cluster_->IsInitEdge(directive_var)) {
                  actual_map_size = dimensions->GetSize(directive_var);
                }
                else if(iter_status_this_var == IterationStatus::Unroll && target_cluster_->IsInitEdge(directive_var)) {
                  actual_map_size = dimensions->GetSize(directive_var);
                }
              }
              else { // if it is spatially mapped
                if((iter_status_this_var == IterationStatus::Edge)
                    || (iter_status_this_var == IterationStatus::Init && target_cluster_->IsInitEdge(directive_var)) ) {
                  if(dimensions->GetSize(directive_var) < original_map_size) {
                    actual_map_size = dimensions->GetSize(directive_var);
                  }
                  else if(is_sp_edge_pe) {
                    actual_map_size = dimensions->GetSize(directive_var) % directive->GetOfs();
                    actual_map_size = (actual_map_size == 0)? original_map_size : actual_map_size;
                  }
                  else {
                    actual_map_size = original_map_size;
                  }
                }
              }

              directive_id++;

              if(!get_num_partial_sum) {
                //TODO: Verify this part
                if(dimensions->IsOverlapped(directive_var)) {
                  if(dimensions->IsSlidingDim(directive_var)) {
                    auto overlapping_dim = dimensions->GetOverlappingDim(directive_var);
                    int reference_dim_size = dataflow->FindDirective(overlapping_dim)->GetSize();
                    int sliding_dim_size = original_map_size;

                    actual_map_size = (reference_dim_size < sliding_dim_size)? actual_map_size : 1;
                  }
                  else { // If directive var dimension is reference dimension, not sliding dimension
                    auto overlapping_dim = dimensions->GetOverlappingDim(directive_var);
                    int sliding_dim_size = dataflow->FindDirective(overlapping_dim)->GetSize();
                    int reference_dim_size = original_map_size;
                    int actual_reference_dim_size = reference_dim_size;
                    //TODO: I don't think we need to deal with edge case on sliding dim size. Verify this
                    actual_map_size = (reference_dim_size < sliding_dim_size)? actual_map_size : reference_dim_size - sliding_dim_size + 1;
                  }
                }
              }
              else { // Per sub-cluster

                if(dimensions->IsOverlapped(directive_var)) {
                  if(dimensions->IsSlidingDim(directive_var)) {
                    auto overlapping_dim = dimensions->GetOverlappingDim(directive_var);
                    int reference_dim_size = dataflow->FindDirective(overlapping_dim)->GetSize();
                    int sliding_dim_size = original_map_size;

                    actual_map_size = sliding_dim_size;
                  }
                  else { // If directive var dimension is reference dimension, not sliding dimension
                    auto overlapping_dim = dimensions->GetOverlappingDim(directive_var);
                    int sliding_dim_size = dataflow->FindDirective(overlapping_dim)->GetSize();
                    int reference_dim_size = actual_map_size;

                    actual_map_size = reference_dim_size - sliding_dim_size + 1;
                    if(actual_map_size < 1) actual_map_size = 1;
//                    std::cout << "Dim: " << directive_var << std::endl;
//                    std::cout << "reference_dim_size - sliding_dim_size + 1 = " << reference_dim_size << " - " <<  sliding_dim_size << " + 1 = " << actual_map_size << std::endl;

                  }
                }
              }


              num_outputs *= actual_map_size;
//            } // End of if(directive var is coupled with the target tensor)

          } // End of for each directive in dataflow

          return num_outputs;
        }

      protected:
        std::shared_ptr<DFA::ClusterUnit> target_cluster_;

        std::unique_ptr<std::map<std::string, int>> num_mapped_elements_;
        std::unique_ptr<std::map<std::string, int>> num_mapped_elements_tp_edge_;

        std::unique_ptr<std::map<std::string, int>> num_unique_elements_;
        std::unique_ptr<std::map<std::string, int>> num_unique_elements_tp_edge_;

        std::unique_ptr<std::map<std::string, int>> num_reused_elements_;
        std::unique_ptr<std::map<std::string, int>> num_reused_elements_tp_edge_;

      private:
        /* This is a critical function that actually estimates reuse based on various cases */
        /* And also, this is actually the most complicated function! (Although it seems pretty short)*/
        long GetPELoadVolume(std::shared_ptr<DFA::Tensor> tensor,
                             std::shared_ptr<std::vector<IterationStatus>> iter_status,
                             bool is_first_pe,
                             CA::EstimationType estimation_type = CA::EstimationType::Exact) {
          auto dataflow = target_cluster_->GetDataflow();
          auto dimensions = target_cluster_->GetDimensions();
          auto coupled_dims = tensor->GetCoupledVariables();

          /* Identify which tensor is changing at the given iter_status*/
          // Prime change id: The inner-most Steady or Edge dimension;
          //   when a variable is incremented, all the sub dimensions are init
          int prime_changing_dim_id = GetPrimeChangingDimDirectiveID(iter_status);
          int tensor_prime_changing_dim_id = GetPrimeChangingDimDirectiveIDTensor(tensor, iter_status, prime_changing_dim_id);

          bool is_all_init_case = (prime_changing_dim_id == -1); // Maybe redundant now. Let's check it later
          long load_volume_with_no_tp_reuse = GetFullMappedVolume(tensor, iter_status, is_first_pe);
          if(is_all_init_case) {
            return load_volume_with_no_tp_reuse;
          }

          bool is_full_unroll = IsTensorFullUnroll(tensor, iter_status);
          if(is_full_unroll) {
            return 0;
          }

          //This must come after full unroll check because it contains all of the followings:
          // 1) full init
          // 2) full unroll
          // 3) init + unroll mixed
          // Except for 2, we need full load, which is manged below
          bool tensor_require_full_load = (tensor_prime_changing_dim_id == -1); // Full init
          if(tensor_require_full_load) {
            return load_volume_with_no_tp_reuse;
          }


          // When we have something changing, check if it is also globally changing among tensors
          bool is_tensor_changing = tensor_prime_changing_dim_id == prime_changing_dim_id;
          // If this tensor is not actually globally changing, it is stationary; no load
          if(!is_tensor_changing) {
            return 0;
          }

          //If this tensor is changing, compute the load volume considering the "partial" temporal reuse
          long load_voulme_with_tp_reuse = 1;

          for(auto& dim : *coupled_dims) {
            auto directive = dataflow->FindDirective(dim);
            auto directive_class = directive->GetClass();
            int directive_idx = dataflow->GetDirectiveIdx(dim);

            bool is_changing_dim = (directive_idx == prime_changing_dim_id);

            if(directive_class == DFA::directive::DirectiveClass::SpatialMap) {
              //Spatial reuse check is always there
              if(is_first_pe) {
                load_voulme_with_tp_reuse *= (*num_mapped_elements_)[dim];
              }
              else {
                load_voulme_with_tp_reuse *= (*num_unique_elements_)[dim];
              }
            }
            else {
              if(is_changing_dim) {
                load_voulme_with_tp_reuse *= (*num_unique_elements_)[dim];
              }
              else {
                load_voulme_with_tp_reuse *= (*num_mapped_elements_)[dim];
              }
            }
          }

          return load_voulme_with_tp_reuse;

        }

        int GetPrimeChangingDimDirectiveID(std::shared_ptr<std::vector<IterationStatus>> iter_status) {
          int changing_dim_directive_id = -1;
          int directive_id = -1;

          for(auto& dim_iter_status : *iter_status) {
            directive_id++;
            if(dim_iter_status != CA::IterationStatus::Unroll && dim_iter_status != CA::IterationStatus::Init) {
              if(directive_id > changing_dim_directive_id) {
                changing_dim_directive_id = directive_id;
              }
            }
          }

          return changing_dim_directive_id;
        }


        int GetPrimeChangingDimDirectiveIDTensor(
            std::shared_ptr<DFA::Tensor> tensor,
            std::shared_ptr<std::vector<IterationStatus>> iter_status,
            int prime_changing_dim_id)
        {
          auto dataflow = target_cluster_->GetDataflow();
          auto dimensions = target_cluster_->GetDimensions();
          auto coupled_dims = tensor->GetCoupledVariables();

          int changing_dim_directive_id = -1;
          int directive_id = -1;

          for(auto& dim_iter_status : *iter_status) {
            directive_id++;
            auto directive = dataflow->at(directive_id);
            auto directive_dim = directive->GetVariable();

            bool found = false;
            for(auto& dim: *coupled_dims) {
              if(dim == directive_dim) {
                found =true;
                break;
              }
            }

            if(!found) {
              continue; //To next directive
            }

            if(dim_iter_status != CA::IterationStatus::Unroll) {
              if(dim_iter_status != CA::IterationStatus::Init || (dim_iter_status == CA::IterationStatus::Init && prime_changing_dim_id > directive_id)) {
                if(directive_id > changing_dim_directive_id) {
                  changing_dim_directive_id = directive_id;
                }
              }
            }
          }

          return changing_dim_directive_id;
        }

        bool IsTensorFullUnroll(
            std::shared_ptr<DFA::Tensor> tensor,
            std::shared_ptr<std::vector<IterationStatus>> iter_status) {
          auto dataflow = target_cluster_->GetDataflow();
          auto dimensions = target_cluster_->GetDimensions();
          auto coupled_dims = tensor->GetCoupledVariables();

          bool ret = true;

          int directive_id = -1;
          for(auto& dim_iter_status : *iter_status) {
            directive_id++;
            auto directive = dataflow->at(directive_id);
            auto directive_dim = directive->GetVariable();

            bool found = false;
            for(auto& dim: *coupled_dims) {
              if(dim == directive_dim) {
                found =true;
                break;
              }
            }

            if(!found) {
              continue; //To next directive
            }

            if(dim_iter_status != CA::IterationStatus::Unroll) {
              ret = false;
            }
          }

          return ret;
        }

          long GetFullMappedVolume(
              std::shared_ptr<DFA::Tensor> tensor,
              std::shared_ptr<std::vector<IterationStatus>> iter_status,
              bool is_first_pe) {
            auto dataflow = target_cluster_->GetDataflow();
            auto dimensions = target_cluster_->GetDimensions();
            auto coupled_vars = tensor->GetCoupledVariables();

            bool has_sp_map = false;
            for(auto& var : * coupled_vars) {
              auto directive = dataflow->FindDirective(var);
              if(directive->GetClass() == DFA::directive::DirectiveClass::SpatialMap) {
                has_sp_map = true;
                break;
              }
            }

            long load_volume_with_no_reuse = 1;

            for(auto& var : * coupled_vars) {
              auto directive = dataflow->FindDirective(var);

              if(directive->GetClass() == DFA::directive::DirectiveClass::SpatialMap) {
                if(is_first_pe) {
                  load_volume_with_no_reuse *= (*num_mapped_elements_)[var];
                }
                else {
                  load_volume_with_no_reuse *= (*num_unique_elements_)[var]; //Spatial reuse
                }
              }
              else { // Temporal Map
                if(has_sp_map) {
                  if(target_cluster_->IsInitEdge(var)) {
                    load_volume_with_no_reuse *= (*num_mapped_elements_tp_edge_)[var];
                  }
                  else {
                    load_volume_with_no_reuse *= (*num_mapped_elements_)[var];
                  }
                }
                else {
                  if(is_first_pe && target_cluster_->IsInitEdge(var)) {
                    load_volume_with_no_reuse *= (*num_mapped_elements_tp_edge_)[var];
                  }
                  else if(is_first_pe && !target_cluster_->IsInitEdge(var)) {
                    load_volume_with_no_reuse *= (*num_mapped_elements_)[var];
                  }
                  else {
                    load_volume_with_no_reuse = 0;
                  }
                }

              }
            }

            return load_volume_with_no_reuse;

          }

        void AnalyzeMappingSizes(std::shared_ptr<DFA::ClusterUnit> target_cluster) {
          auto dataflow = target_cluster->GetDataflow();
          auto dimensions = target_cluster->GetDimensions();

          int idx = 0;
          //For each data class (tensor)
          for(auto& directive : *dataflow) {
            auto loop_var = directive->GetVariable();
            auto directive_class = directive->GetClass();
            auto map_size = directive->GetSize();
            auto ofs_size = directive->GetOfs();

            auto dim_size = dimensions->GetSize(loop_var);
            auto edge_map_size = dim_size % ofs_size;
            auto edge_out_of_bound_size = map_size - edge_map_size;

            bool is_fully_tp_unrolled = false;

            (*num_mapped_elements_)[loop_var] = map_size;

            switch(directive_class) {
              case DFA::directive::DirectiveClass::TemporalMap:{
                is_fully_tp_unrolled = (dim_size == map_size);
                (*num_unique_elements_)[loop_var] = (map_size > ofs_size)? ofs_size : map_size;
                (*num_reused_elements_)[loop_var] = (*num_mapped_elements_)[loop_var] - (*num_unique_elements_)[loop_var];

                (*num_mapped_elements_tp_edge_)[loop_var] = edge_map_size;
                (*num_unique_elements_tp_edge_)[loop_var] = (edge_out_of_bound_size > ofs_size)? 0 : ofs_size - edge_out_of_bound_size; //TODO
                (*num_reused_elements_tp_edge_)[loop_var] = (*num_mapped_elements_tp_edge_)[loop_var] - (*num_unique_elements_tp_edge_)[loop_var];

                break;
              }
              case DFA::directive::DirectiveClass::SpatialMap: {
                (*num_unique_elements_)[loop_var] =(map_size > ofs_size)? ofs_size : map_size;
                (*num_reused_elements_)[loop_var] =(map_size > ofs_size)? map_size-ofs_size : 0;
                break;
              }
              default: {
                error_handler->PrintErrorMsg(TL::ErrorCode::InvalidCluster, std::to_string(target_cluster_->GetClusterLevel()), this->instance_name_);
              }
            } // End of switch(directive_class)
/*
            if(is_fully_tp_unrolled) {
              (*num_unique_elements_)[loop_var] = 0;
              (*num_reused_elements_)[loop_var] = map_size;
            }
            */
          } // End of for(auto directive : dataflow)
        }

        long GetNumTempIters(std::string var) {
          auto dataflow = target_cluster_->GetDataflow();
          auto dimensions = target_cluster_->GetDimensions();
          auto directive = dataflow->FindDirective(var);

          auto directive_class = directive->GetClass();
          auto map_size = directive->GetSize();
          auto ofs_size = directive->GetOfs();

          auto dim_size = dimensions->GetSize(var);

          long ret = 1;

          if(directive_class == DFA::directive::DirectiveClass::TemporalMap) {
            ret = static_cast<long>(std::ceil(static_cast<double>(dim_size - map_size) / static_cast<double>(ofs_size)));
          }

          return ret;
        }

        bool IsSpatialEdge(std::shared_ptr<std::vector<IterationStatus>> iter_status) {
          auto dataflow = target_cluster_->GetDataflow();
          auto dimensions = target_cluster_->GetDimensions();

          int directive_id = 0;
          for(auto& dim_iter_status : *iter_status) {
            auto directive = dataflow->at(directive_id);
            auto directive_dim = directive->GetVariable();
            auto directive_class = directive->GetClass();

            if(directive_class == DFA::directive::DirectiveClass::SpatialMap) {
              if(dim_iter_status == CA::IterationStatus::Edge) {
                return true;
              }
              else if(dim_iter_status == CA::IterationStatus::Init && target_cluster_->IsInitEdge(directive_dim)) {
                return true;
              }
              else if(dim_iter_status == CA::IterationStatus::Unroll && target_cluster_->IsInitEdge(directive_dim)) {
                return true;
              }
              else {
                return false;
              }
            }
            directive_id++;
          }

          return false;
        }


    }; // End of class TemporalReuseAnalysis
  }; // End of namespace CA
}; // End of namespace maestro
#endif
