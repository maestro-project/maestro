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

#ifndef MAESTRO_DFA_CLUSTER_UNIT_HPP_
#define MAESTRO_DFA_CLUSTER_UNIT_HPP_

//#define DEBUG_CLUSTER_UNIT

#include <memory>
#include <vector>
#include <list>
#include <map>
#include <string>

#include "BASE_maestro-class.hpp"
#include "TL_error-handler.hpp"

#include "AHW-noc-model.hpp"
#include "DFA_dimension-table.hpp"
#include "DFA_directives.hpp"
#include "DFA_directive-table.hpp"

#include "DFA_tensor.hpp"
#include "DFA_tensor-table.hpp"


namespace maestro {
  namespace DFA {

    const int invalid_map_pos = -1;

    class ClusterUnit : public MAESTROClass {
      public:
        ClusterUnit(int cluster_level, int cluster_size,
                    std::shared_ptr<DFA::DirectiveTable> dataflow,
                    std::shared_ptr<DFA::DimensionTable> dimensions,
                    std::shared_ptr<DFA::TensorTable> tensors,
                    std::shared_ptr<abstract_hw::NetworkOnChipModel> noc) :
              cluster_level_(cluster_level),
              cluster_size_(cluster_size),
              noc_(noc),
              dimensions_(dimensions),
              dataflow_(dataflow),
              MAESTROClass("ClusterUnitAnalysis_Lv"+ std::to_string(cluster_level))
        {
          num_mapped_elements_ = std::make_unique<std::map<std::string, int>>();
          sp_mapped_unique_elements_ = std::make_unique<std::map<std::string, int>>();
          tp_mapped_unique_elements_ = std::make_unique<std::map<std::string, int>>();
          sp_mapped_reused_elements_ = std::make_unique<std::map<std::string, int>>();
          tp_mapped_reused_elements_ = std::make_unique<std::map<std::string, int>>();
          inner_temporal_map_indices_ = std::make_unique<std::map<std::string, int>>();
          dataflow->ConvertToInputCentric();
          Preprocess();
        }

        int GetClusterLevel() {
          return cluster_level_;
        }

        int GetNumOuterTemporalIterations() {
          return num_outer_temporal_iterations_;
        }

        int GetNumInnerTemporalIterations() {
          return num_innner_temporal_iterations_;
        }

        std::shared_ptr<abstract_hw::NetworkOnChipModel> GetNoCModel() {
          return noc_;
        }

        std::shared_ptr<DFA::DimensionTable> GetDimensions() {
          return dimensions_;
        }

        std::shared_ptr<DFA::DirectiveTable> GetDataflow() {
          return dataflow_;
        }

        bool IsUnrolled(std::string dim) {
          bool ret = false;
          auto directive = dataflow_->FindDirective(dim);
          int dim_size =  dimensions_->GetSize(dim);
          int map_size = directive->GetSize();
          int ofs_size = directive->GetOfs();

          if(directive != nullptr) {
            if(directive->GetClass() == DFA::directive::DirectiveClass::TemporalMap) {
              ret = (map_size >= dim_size);
            }
            else { // Spatial Map case
              int num_pes = cluster_size_;
              int sp_covered_size = num_pes * ofs_size + (map_size - ofs_size); //TODO: double-check this
              ret = (sp_covered_size >= dim_size);
            }
          }

          return ret;
        }

        bool IsInitEdge(std::string var) {
          bool ret = false;
          auto directive = dataflow_->FindDirective(var);
          if(directive != nullptr) {
            auto directive_class = directive->GetClass();
            if(directive_class == DFA::directive::DirectiveClass::TemporalMap) {
              ret = (directive->GetSize() > dimensions_->GetSize(var));
            }
            else {
              ret = ( (directive->GetOfs() * cluster_size_) > dimensions_->GetSize(var));
            }
          }
          return ret;
        }

        bool IsInitUnroll(std::string var) {
          bool ret = false;
          auto directive = dataflow_->FindDirective(var);
          if(directive != nullptr) {
            auto directive_class = directive->GetClass();
            if(directive_class == DFA::directive::DirectiveClass::TemporalMap) {
              ret = (directive->GetSize() == dimensions_->GetSize(var));
            }
            else {
              ret = ( (directive->GetOfs() * cluster_size_) == dimensions_->GetSize(var));
            }
          }
          return ret;
        }


        bool HasSpatialEdgeCase() {
          return (num_edge_spatial_iterations_ != 0);
        }

        bool HasTemporalEdgeCase(std::string var) {
          auto directive = dataflow_->FindDirective(var);
          if(directive == nullptr) {
            this->error_handler_->PrintErrorMsg(TL::ErrorCode::InvalidDimension, var);
          }

          if(directive->GetClass() ==  directive::DirectiveClass::SpatialMap) {
            return false; // Treats entire spatial iterations as one temporal iteration
          }

          int dim_size = 0;
          int offset = directive->GetOfs();
          bool is_overlapping = dimensions_->IsOverlapped(var);

          if(is_overlapping) {
            auto overlapping_var = dimensions_->GetOverlappingDim(var);
            bool is_sliding_dim = dimensions_->IsSlidingDim(var);

            if(is_sliding_dim) {
              int sliding_dim_size = dimensions_->GetSize(var);
              dim_size = sliding_dim_size;
            }
            else {
              int reference_dim_size = dimensions_->GetSize(var);
              int sliding_dim_size = dimensions_->GetSize(overlapping_var);

              dim_size = reference_dim_size - sliding_dim_size + 1;
            }
          }
          else {
            dim_size = dimensions_->GetSize(var);
          }

          return (dim_size % offset != 0);
        }

        long GetNumTemporalIterations(std::string var) {
          long ret = 0;

          auto directive = dataflow_->FindDirective(var);
          if(directive == nullptr) {
            this->error_handler_->PrintErrorMsg(TL::ErrorCode::InvalidDimension, var);
          }

          if(directive->GetClass() ==  directive::DirectiveClass::SpatialMap) {
            return 1; // Treats entire spatial iterations as one temporal iteration
          }

          int dim_size = 0;
          int offset = directive->GetOfs();
          bool is_overlapping = dimensions_->IsOverlapped(var);

          if(is_overlapping) {
            auto overlapping_var = dimensions_->GetOverlappingDim(var);
            bool is_sliding_dim = dimensions_->IsSlidingDim(var);

            if(is_sliding_dim) {
              int sliding_dim_size = dimensions_->GetSize(var);
              dim_size = sliding_dim_size;
            }
            else {
              int reference_dim_size = dimensions_->GetSize(var);
              int sliding_dim_size = dimensions_->GetSize(overlapping_var);

              dim_size = reference_dim_size - sliding_dim_size + 1;
            }
          }
          else {
            dim_size = dimensions_->GetSize(var);
          }

          ret = dim_size / offset;
          if(dim_size % offset != 0) {
            ret++;
          }

          return ret;
        }

        long GetNumSpatialIterations(std::string dim) {
          long ret = 0;

          auto directive = dataflow_->FindDirective(dim);
          if(directive == nullptr) {
            this->error_handler_->PrintErrorMsg(TL::ErrorCode::InvalidDimension, dim);
          }

          if(directive->GetClass() ==  directive::DirectiveClass::TemporalMap) {
            return 1; // Treats entire spatial iterations as one temporal iteration
          }

          int dim_size = 0;
          int offset = directive->GetOfs();
          bool is_overlapping = dimensions_->IsOverlapped(dim);

          if(is_overlapping) {
            auto overlapping_var = dimensions_->GetOverlappingDim(dim);
            bool is_sliding_dim = dimensions_->IsSlidingDim(dim);

            if(is_sliding_dim) {
              int sliding_dim_size = dimensions_->GetSize(dim);
              dim_size = sliding_dim_size;
            }
            else {
              int reference_dim_size = dimensions_->GetSize(dim);
              int sliding_dim_size = dimensions_->GetSize(overlapping_var);

              dim_size = reference_dim_size - sliding_dim_size + 1;
            }
          }
          else {
            dim_size = dimensions_->GetSize(dim);
          }

          int full_spatial_coverage = offset * cluster_size_;

          ret = dim_size / full_spatial_coverage;
          if(dim_size % full_spatial_coverage != 0) {
            ret++;
          }

          return ret;
        }


        std::vector<std::string> GetOuterTemporalIterationVars () {
          std::vector<std::string> ret;

          for(int idx = 0; idx < spatial_map_idx_; idx++) {
            if(dataflow_->at(idx)->GetClass() == directive::DirectiveClass::TemporalMap) {
              ret.push_back(dataflow_->at(idx)->GetVariable());
            }
          }

          return ret;
        }

        std::vector<std::string> GetInnerTemporalIterationVars () {
          std::vector<std::string> ret;

          for(int idx = spatial_map_idx_; idx < dataflow_->size(); idx++) {
            if(dataflow_->at(idx)->GetClass() == directive::DirectiveClass::TemporalMap) {
              ret.push_back(dataflow_->at(idx)->GetVariable());
            }
          }

          return ret;
        }

        std::shared_ptr<std::vector<std::string>> GetAllVarsInDirectiveClass (DFA::directive::DirectiveClass target_class) {
          std::shared_ptr<std::vector<std::string>> ret = std::make_shared<std::vector<std::string>>();

          for(auto directive : *dataflow_) {
            if(target_class == DFA::directive::DirectiveClass::Invalid
                               || directive->GetClass() == target_class) {
              ret->push_back(directive->GetVariable());
            }
          }

          return ret;
        }

        std::shared_ptr<std::vector<std::string>> GetAllDimensionVars () {
          std::shared_ptr<std::vector<std::string>> ret = std::make_shared<std::vector<std::string>>();

          for(auto directive : *dataflow_) {
            ret->push_back(directive->GetVariable());
          }

          return ret;
        }



        long GetNumTemporalIterations_SingleDim(std::string var, bool edge_case) {
          long ret = 0;

          auto dim_size = dimensions_->GetSize(var);
          auto directive = dataflow_->FindDirective(var);

          /*
          bool is_full_tp_unrolled = (dim_size == directive->GetSize());

          if(is_full_tp_unrolled) {
            ret = 0;
          }
          */

          //TODO: Fix this hard-coded part
          if(var == "X") {
            dim_size = dim_size - dimensions_->GetSize("S") +1;
          }
          else if(var == "Y") {
            dim_size = dim_size - dimensions_->GetSize("R") +1;
          }

          if(directive->GetClass() ==  directive::DirectiveClass::TemporalMap) {
            if(!edge_case) {
              ret = dim_size / directive->GetOfs();
            }
            else {
              ret = dim_size % directive->GetOfs(); //TODO: Fix this
            }
          }

          return ret;
        }


        long GetNumSpatialSparseIterations(int blk_sparsity_size, int num_sparse_blks) {
          long ret;

          auto spatial_map_directive = dataflow_->at(spatial_map_idx_);
          auto spatially_mapped_dimension = spatial_map_directive->GetVariable();

          auto sp_dim_size = dimensions_->GetSize(spatially_mapped_dimension);
          auto sp_map_ofs = spatial_map_directive->GetOfs();

          sp_dim_size = sp_dim_size - blk_sparsity_size * num_sparse_blks;

          ret = sp_dim_size / (sp_map_ofs * cluster_size_);

          if(sp_dim_size % (sp_map_ofs * cluster_size_) != 0) {
            ret++;
          }

          return ret;
        }

        long GetNumTotalIterations() {
          long ret = 1;

          for(int idx = 0; idx < dataflow_->size();idx ++) {

            auto directive = dataflow_->at(idx);
            auto mapped_dim = directive->GetVariable();
            auto dim_size = dimensions_->GetSize(mapped_dim);
            int num_iter_this_dim = 1;

            if(mapped_dim == "X") {
              dim_size = dim_size - dimensions_->GetSize("S") +1;
            }
            else if(mapped_dim == "Y") {
              dim_size = dim_size - dimensions_->GetSize("R") +1;
            }

            if (directive->GetClass() == directive::DirectiveClass::TemporalMap) {
              num_iter_this_dim = dim_size/directive->GetOfs();
              if(dim_size % directive->GetOfs() != 0) {
                num_iter_this_dim++;
              }
            }
            else if(directive->GetClass() == directive::DirectiveClass::SpatialMap) {
              num_iter_this_dim = dim_size/(directive->GetOfs() * cluster_size_);
              if(dim_size % (directive->GetOfs() * cluster_size_) != 0) {
                num_iter_this_dim++;
              }
            }
            else {
              //TODO: Handle this error
            }

            ret *= num_iter_this_dim;
          } //End of for(idx)
          return ret;
        }

        bool isInnerTemporalMapped(std::string var) {
          auto directive = dataflow_->FindDirective(var);
          auto inner_temporal_map_directive = dataflow_->at(inner_temporal_map_idx_);

          return (directive == inner_temporal_map_directive);
        }


        long GetNumClusters(bool is_spatial_edge = false) {
          if(!is_spatial_edge) {
            return cluster_size_;
          }
          else {
            return num_spatial_edge_clusters_;
          }
        }

        long GetNumPartialOutputs() {
          return num_pouts_;
        }
/*
        long GetNumSpatialIterations(bool is_edge = false) {
          if(is_edge) {
            return num_edge_spatial_iterations_;
          }
          else {
            return num_steady_spatial_iterations_;
          }
        }
*/
        void SetDataflow(std::shared_ptr<DFA::DirectiveTable> new_dataflow) {
          dataflow_ = new_dataflow;
        }

        void Reset() {
          num_mapped_elements_->clear();
          sp_mapped_unique_elements_->clear();
          tp_mapped_unique_elements_->clear();
          sp_mapped_reused_elements_->clear();
          tp_mapped_reused_elements_->clear();

          cluster_size_ = 1;

          spatial_map_idx_ = invalid_map_pos;
          outer_temporal_map_idx_ = invalid_map_pos;
          inner_temporal_map_idx_ = invalid_map_pos;

          num_spatial_iterations_ = 1;
          num_outer_temporal_iterations_ = 1;
          num_innner_temporal_iterations_ = 1;
          num_spatial_edge_clusters_ = 1;

          num_pouts_ = 0;
        }

        void Reanalyze() {
          Preprocess();
        }


      protected:
//        directive::ClusterType type_;

        int cluster_level_ = -1;
        int cluster_size_ = 1;

        int spatial_map_idx_ = invalid_map_pos;
        int outer_temporal_map_idx_ = invalid_map_pos;
        int inner_temporal_map_idx_ = invalid_map_pos;

        int num_spatial_iterations_ = 1;
        int num_outer_temporal_iterations_ = 1;
        int num_innner_temporal_iterations_ = 1;

        int num_spatial_edge_clusters_ = 1;
        int num_steady_spatial_iterations_ = 1;
        int num_edge_spatial_iterations_ = 0;

        long num_pouts_ = 0;

        bool has_double_spatial_maps_ = false;

        std::shared_ptr<DFA::DimensionTable> dimensions_;
        std::shared_ptr<DFA::DirectiveTable> dataflow_;
        std::shared_ptr<abstract_hw::NetworkOnChipModel> noc_;

        std::unique_ptr<std::map<std::string, int>> num_mapped_elements_; //TSz
        std::unique_ptr<std::map<std::string, int>> sp_mapped_unique_elements_; //TUSz
        std::unique_ptr<std::map<std::string, int>> tp_mapped_unique_elements_; //TUSz

        std::unique_ptr<std::map<std::string, int>> sp_mapped_reused_elements_; //
        std::unique_ptr<std::map<std::string, int>> tp_mapped_reused_elements_; //

        std::unique_ptr<std::map<std::string, int>> inner_temporal_map_indices_;

      private:
        //Get the index of the inner-most spatial map
        void AnalyzeSpatialMapIdx() {
          int spatial_map_index = -1;
          int idx = 0;

          for(auto& it : *dataflow_) {
            if(it->GetClass() == directive::DirectiveClass::SpatialMap) {
              spatial_map_index = idx;
            }
            idx++;
          }

          spatial_map_idx_ = spatial_map_index;

          if(spatial_map_idx_ != 0 && dataflow_->at(spatial_map_idx_-1)->GetClass() == directive::DirectiveClass::SpatialMap) {
            has_double_spatial_maps_ = true;
          }
        }

        void AnalyzeOutterTemporalMapIdx() {
          int spatial_map_index = spatial_map_idx_;
          int outer_temporal_map_index = spatial_map_index;

          for(int idx = spatial_map_index; idx >= 0; idx--) {
            if(dataflow_->at(idx)->GetClass() == directive::DirectiveClass::TemporalMap) {
              outer_temporal_map_index = idx;
              break;
            }
          }

          outer_temporal_map_idx_ = outer_temporal_map_index;
        }

        //Get the index of the inner-most temporal map under the inner-most spatial map.
        // If no temporal map is under the inner-most spatial map, it returns the index to the inner-most spatial map
        void AnalyzeInnerTemporalMapIdx() {
          int inner_temporal_map_index = -1;
          int idx = 0;

          int spatial_map_index = spatial_map_idx_;
          inner_temporal_map_index = spatial_map_index;

          for(int idx = spatial_map_index; idx < dataflow_->size() ; idx++) {
            if(dataflow_->at(idx)->GetClass() == directive::DirectiveClass::TemporalMap) {
              bool isUnrolled = dataflow_->at(idx)->GetSize() == dimensions_->GetSize(dataflow_->at(idx)->GetVariable());
              if(!isUnrolled) {
                inner_temporal_map_index = idx;
              }
            }
            idx++;
          }

          inner_temporal_map_idx_ = inner_temporal_map_index;
        }


        void AnalyzeNumSpatialIterations() {
          auto spatial_map_directive = dataflow_->at(spatial_map_idx_);
          auto spatially_mapped_dimension = spatial_map_directive->GetVariable();

          auto sp_dim_size = dimensions_->GetSize(spatially_mapped_dimension);
          auto sp_map_ofs = spatial_map_directive->GetOfs();
          num_spatial_iterations_ = sp_dim_size / (sp_map_ofs * cluster_size_);

          if(sp_dim_size % (sp_map_ofs * cluster_size_) != 0) {
            num_spatial_iterations_++;
          }
        }

        void AnalyzeNumOutterTemporalIterations() {
          for(int idx = 0; idx < spatial_map_idx_;idx ++) {

            auto directive = dataflow_->at(idx);
            auto mapped_dim = directive->GetVariable();
            auto dim_size = dimensions_->GetSize(mapped_dim);
            int num_iter_this_dim;

            if (directive->GetClass() == directive::DirectiveClass::TemporalMap) {
              num_iter_this_dim = dim_size/directive->GetOfs();
              if(dim_size % directive->GetOfs() != 0) {
                num_iter_this_dim++;
              }
            }
            else if(directive->GetClass() == directive::DirectiveClass::SpatialMap) {
              num_iter_this_dim = dim_size/(directive->GetOfs() * cluster_size_);
              if(dim_size % (directive->GetOfs() * cluster_size_) != 0) {
                num_iter_this_dim++;
              }
            }
            else {
              //TODO: Handle this error
            }

            num_outer_temporal_iterations_ *= num_iter_this_dim;
          } //End of for(idx)
        } // End function

        void AnalyzeNumInnerTemporalIterations() {
          for(int idx = spatial_map_idx_; idx < dataflow_->size(); idx ++) {
            auto directive = dataflow_->at(idx);

            if (directive->GetClass() == directive::DirectiveClass::TemporalMap) {

              auto mapped_dim = directive->GetVariable();
              auto dim_size = dimensions_->GetSize(mapped_dim);

              int num_iter_this_dim = dim_size / directive->GetOfs();

              //This also handles cases where num_ter_this_dim == 0
              if (dim_size % directive->GetOfs() != 0) {
                num_iter_this_dim++;
              }

              num_innner_temporal_iterations_ *= num_iter_this_dim;
            } // End of if(directive class == TemporalMap)
          } // End of for (idx)
        } // End function

        void AnalyzeMappingSizes() {
          int idx = 0;
          //For each data class (tensor)
          for(auto& directive : *dataflow_) {
            auto loop_var = directive->GetVariable();

            (*num_mapped_elements_)[loop_var] = directive->GetSize();

            //1. Spatial reuse analysis
            // Handles two cases;
              // case 1: Have only one spatial map within this cluster
              // case 2: Two spatial maps are placed together. For legality, their variables need to belong to different tensors
            if(idx == spatial_map_idx_ || (idx == spatial_map_idx_-1 && directive->GetClass() == directive::DirectiveClass::SpatialMap)) {
              (*sp_mapped_unique_elements_)[loop_var] = directive->GetOfs();
            }
            else {
              (*sp_mapped_unique_elements_)[loop_var] = directive->GetSize();
            }

            (*sp_mapped_reused_elements_)[loop_var] = (*num_mapped_elements_)[loop_var] - (*sp_mapped_unique_elements_)[loop_var];


            //2. Temporal reuse analysis
            if(idx == inner_temporal_map_idx_) {
              (*tp_mapped_unique_elements_)[loop_var] = directive->GetOfs();
            }
            else {
              (*tp_mapped_unique_elements_)[loop_var] = directive->GetSize();
            }

            (*tp_mapped_reused_elements_)[loop_var] = (*num_mapped_elements_)[loop_var] - (*tp_mapped_unique_elements_)[loop_var];

          } // End of for_each (directive)
        }

        void AnalyzeSpatialEdgeCase() {
          auto sp_map_directive = dataflow_->at(spatial_map_idx_);
          auto sp_dim_sz = dimensions_->GetSize(sp_map_directive->GetVariable());

          /*********************************************/
          //TODO: fix this conv-specific hard-coded part
          auto sp_var = sp_map_directive->GetVariable();
          if(sp_var == "X") {
            sp_dim_sz = dimensions_->GetSize("X'");
          }
          else if(sp_var == "Y") {
            sp_dim_sz = dimensions_->GetSize("Y'");
          }

          /**********************************************/
          num_spatial_edge_clusters_ = sp_dim_sz % (sp_map_directive->GetOfs() * cluster_size_);
          if(sp_dim_sz <= sp_map_directive->GetSize()) {
            num_spatial_edge_clusters_ = 1;
          }
          num_steady_spatial_iterations_ = sp_dim_sz / (sp_map_directive->GetOfs() * cluster_size_);
          num_edge_spatial_iterations_ = (sp_dim_sz % (sp_map_directive->GetOfs() * cluster_size_) == 0)? 0 : 1;
          num_steady_spatial_iterations_ -= num_edge_spatial_iterations_;

#ifdef DEBUG_CLUSTER_UNIT
          std::cout << "Cluster lv: " << cluster_level_ << std::endl;
          std::cout << "Cluster size: " << cluster_size_ << std::endl;
          std::cout << "num_steady_spatial_iterations_ size: " << num_steady_spatial_iterations_ << std::endl;
          std::cout << "Cluster num_edge_spatial_iterations_: " << num_edge_spatial_iterations_ << std::endl;
#endif
        }





        //TODO: Fix this convolution-specific hard-coded parts
        void AnalyzeNumPartialOutputs() {
          long num_pouts = 1;
/*
 * Need overlapping dimension information
          std::shared_ptr<std::list<std::string>> var_list = std::make_shared<std::list<std::string>>();
          for(auto& tensor : *tensors_) {
            if(tensor->GetTensorClass() == TensorClass::InputTensor) {
              auto vars = tensor->GetCorrelatedVariables();
              for(auto& var : *vars) {
                var_list->push_back(var);
              }
            }
          }

          var_list->sort();
          var_list->unique();

          for(auto& var : *var_list) {
            num_pouts *= num_mapped_elements_->at(var);
          }
*/
          int filter_channel = dataflow_->FindDirective("K")->GetSize();
          int filter_height = dataflow_->FindDirective("R")->GetSize();
          int filter_width = dataflow_->FindDirective("S")->GetSize();

          int input_channel = dataflow_->FindDirective("C")->GetSize();
          int input_height = dataflow_->FindDirective("Y")->GetSize();
          int input_width = dataflow_->FindDirective("X")->GetSize();

          int output_height = input_height - filter_height + 1;
          int output_width = input_width - filter_width + 1;

          output_height = (output_height <= 0) ? 1 : output_height;
          output_width = (output_width <= 0) ? 1 : output_width;

          if(input_height >= filter_height && input_width >= filter_width) {
            num_pouts_ = filter_channel * input_channel * output_height * output_width * filter_height * filter_width;
          }
          else if(input_height < filter_height && input_width < filter_width) {
            num_pouts_ = filter_channel * input_channel * filter_height * filter_width;

          }
          else if(input_height >= filter_height && input_width < filter_width) {
            num_pouts_ = filter_channel * input_channel * output_height * filter_height * filter_width;
          }
          else if(input_height < filter_height && input_width >= filter_width) {
            num_pouts_ = filter_channel * input_channel * output_width * filter_height * filter_width;
          }
          else {
            std::cout << "Not supported case" << std::endl;
            exit(-1);
          }
        }

        void Preprocess() {
          //Functions regarding spatial mapping always need to be called first
          AnalyzeSpatialMapIdx();
          AnalyzeInnerTemporalMapIdx();
          AnalyzeOutterTemporalMapIdx();

          AnalyzeNumSpatialIterations();
          AnalyzeNumOutterTemporalIterations();
          AnalyzeNumInnerTemporalIterations();

          AnalyzeSpatialEdgeCase();

          AnalyzeMappingSizes();

          AnalyzeNumPartialOutputs();

        }

    }; // End of class ClusterUnit
  }; // End of namespace DFA
}; // End of namespace maestro


#endif
