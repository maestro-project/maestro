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


#ifndef MAESTRO_DFA_CONV_CLUSTER_ANALYSIS_HPP_
#define MAESTRO_DFA_CONV_CLUSTER_ANALYSIS_HPP_

//#define DEBUG_CLUSTER_ANALYSIS

#include <vector>
#include <memory>

#include "BASE_constants.hpp"
#include "BASE_maestro-class.hpp"
#include "TL_error-handler.hpp"

#include "DFA_directives.hpp"
#include "DFA_directive-table.hpp"
#include "DFA_layer.hpp"
#include "DFA_tensor.hpp"

#include "DFA_cluster-unit.hpp"
#include "DFA_cluster-table.hpp"

#include "DFA_cluster-analysis.hpp"

#include "AHW-noc-model.hpp"


namespace maestro {
	namespace DFA {

	  /* Class ClusterAnalysis
	   * Receive full dataflow description and split it into clusters
	   */


		class ConvClusterAnalysis : public ClusterAnalysis {
			public:
		    ConvClusterAnalysis(int num_pes,
		                    std::shared_ptr<DFA::TensorTable> tensors,
		                    std::shared_ptr<DFA::DimensionTable> full_dimensions,
		                    std::shared_ptr<DFA::DirectiveTable> full_dataflow,
		                    std::shared_ptr<std::vector<std::shared_ptr<abstract_hw::NetworkOnChipModel>>> nocs) :
		      MAESTROClass("ClusterAnalysis"),
		      num_unit_clusters_(num_pes),
		      tensors_(tensors),
		      full_dimensions_(full_dimensions),
		      full_dataflow_(full_dataflow),
		      nocs_(nocs) {
		      clusters_ = std::make_shared<DFA::ClusterTable>();
		      AnalyzeClusters();
		    }

		    std::shared_ptr<DFA::ClusterTable> GetClusters() {
		      return clusters_;
		    }

			protected:
		    int num_unit_clusters_;

		    // Dimension and dataflow information of entire accelerator(s)
		    std::shared_ptr<DFA::TensorTable> tensors_;
        std::shared_ptr<DFA::DimensionTable> full_dimensions_;
		    std::shared_ptr<DFA::DirectiveTable> full_dataflow_;
		    std::shared_ptr<std::vector<std::shared_ptr<abstract_hw::NetworkOnChipModel>>> nocs_;


		    std::shared_ptr<DFA::ClusterTable> clusters_;

			private:
        void AnalyzeClusters() { //TODO: Fix this
          std::vector<int> cluster_sizes;

          int num_cluster_lvs = 0;
          int uppermost_cluster_unit_size = 1;
          //Count the number of clusters
          for(auto directive: *full_dataflow_) {
            if(directive->GetClass() == DFA::directive::DirectiveClass::Cluster) {
              cluster_sizes.push_back(directive->GetSize());
              num_cluster_lvs++;
              uppermost_cluster_unit_size *= directive->GetSize();
            }
          }

          if(uppermost_cluster_unit_size == 1) uppermost_cluster_unit_size = num_unit_clusters_;

          int IY_size, IX_size, R_size, S_size;

          for(auto dim : *full_dimensions_) {
            if(dim->GetName() == "Y") {
              IY_size = dim->GetSize();
            }
            else if(dim->GetName() == "X") {
              IX_size = dim->GetSize();
            }
            else if(dim->GetName() == "R") {
              R_size = dim->GetSize();
            }
            else if(dim->GetName() == "S") {
              S_size = dim->GetSize();
            }
          }

          int current_cluster_level = num_cluster_lvs;
          int num_unit_clusters = uppermost_cluster_unit_size * (num_unit_clusters_/uppermost_cluster_unit_size); //Normalizing the number
          int current_cluster_size = num_unit_clusters;
          int last_cluster_size = num_unit_clusters;

          auto cluster_dataflow = std::make_shared<DFA::DirectiveTable>();
          std::shared_ptr<DFA::DirectiveTable> prev_dataflow = nullptr;
          auto cluster_dimensions =full_dimensions_;
          std::shared_ptr<DFA::DimensionTable> prev_cluster_dimensions = nullptr;

          for(auto directive: *full_dataflow_) {

            switch(directive->GetClass()) {
              case directive::DirectiveClass::SpatialMap: {
                cluster_dataflow->AddDirective(directive);
                break;
              } // End of case SpatialMap

              case directive::DirectiveClass::TemporalMap: {
                cluster_dataflow->AddDirective(directive);
                break;
              } // End of case TemporalMap

              case directive::DirectiveClass::Cluster: {
                std::shared_ptr<abstract_hw::NetworkOnChipModel> noc;

                if(nocs_->size() < current_cluster_level) {
                  noc = nocs_->at(0);
                }
                else {
                  noc = nocs_->at(current_cluster_level);
                }

                std::list<std::string> var_list_input_tensor;
                for(auto& tensor : *tensors_) {
                  if(tensor->GetTensorClass() == TensorClass::InputTensor) {
                    auto correlated_vars = tensor->GetCorrelatedVariables();

                    for(auto var : *correlated_vars) {
                      var_list_input_tensor.push_back(var);
                    }
                  }
                }
                var_list_input_tensor.sort();
                var_list_input_tensor.unique();

                //
                ConvertToInputCentric(cluster_dataflow);

                /* Complement dataflow */
                // Inherit directives
                if(prev_dataflow != nullptr) {
                  for(auto& upper_directive : *prev_dataflow) {
                    auto upper_directive_var = upper_directive->GetVariable();
                    if(cluster_dataflow->FindDirective(upper_directive_var) == nullptr) {
                      int upper_map_sz = upper_directive->GetSize();
                      int upper_ofs_sz = upper_directive->GetOfs();

                      auto inherited_dataflow = std::make_shared<DFA::directive::TemporalMap>(upper_map_sz, upper_ofs_sz, upper_directive_var);
                      cluster_dataflow->AddDirectiveFront(inherited_dataflow);
                    }
                  }
                }

                // Insert "Unroll" of missing dimensions
                for(auto& var : var_list_input_tensor) {
                  if(cluster_dataflow->FindDirective(var) == nullptr) {

                    if(cluster_dimensions->HasVar(var)) {
                      int dim_sz = cluster_dimensions->GetSize(var);
                      auto dummy_directive = std::make_shared<DFA::directive::TemporalMap>(dim_sz, dim_sz, var);
                      cluster_dataflow->AddDirective(dummy_directive);
                    }
                  }
                }

#ifdef DEBUG_CLUSTER_ANALYSIS
                std::cout << "Cluster level: " << current_cluster_level << std::endl;
                std::cout << "num_cluster_lvs: " << num_cluster_lvs << std::endl;
                std::cout << "Cluster dimensions:" << std::endl;
                std::cout << cluster_dimensions->ToString() << std::endl;
                std::cout << cluster_dataflow->ToString() << std::endl;
                std::cout << "uppermost_cluster_unit_size: " << uppermost_cluster_unit_size << std::endl;
#endif

#ifdef DEBUG_CLUSTER_ANALYSIS
                if(noc == nullptr) {
                  std::cout <<"NoC is null" <<std::endl;
                }
                else {
                  std::cout << "NoC Bandwidth: " << noc->GetBandwidth() << std::endl;
                }

#endif

                ConvertToInputCentric(cluster_dataflow);

                if(current_cluster_level == num_cluster_lvs) {
                  current_cluster_size = num_unit_clusters / uppermost_cluster_unit_size;
                }
                else {
                  current_cluster_size = last_cluster_size;
                }



                auto new_cluster = std::make_shared<DFA::ClusterUnit>(current_cluster_level,
                                                                      current_cluster_size,
                                                                      cluster_dataflow,
                                                                      cluster_dimensions,
                                                                      tensors_,
                                                                      noc);

                last_cluster_size = directive->GetSize();

#ifdef DEBUG_CLUSTER_ANALYSIS
                std::cout << "num_unit_clusters: " << num_unit_clusters << std::endl;
                std::cout << "Current cluster size: " << current_cluster_size << std::endl;
                std::cout << "last_cluster_size: " << last_cluster_size << std::endl;
#endif

                clusters_->PutCluster(new_cluster);
                prev_cluster_dimensions = cluster_dimensions;
                cluster_dimensions = std::make_shared<DFA::DimensionTable>();
                cluster_dimensions->SetOverlapTable(prev_cluster_dimensions->GetOverlapTable());

                // Update Dimension table; new dimension = mapping size at upper cluster
                for(auto var : var_list_input_tensor) {
                  auto directive = cluster_dataflow->FindDirective(var);

                  //Deal with edge cases
                  int final_dim_sz = std::min(directive->GetSize(), prev_cluster_dimensions->GetSize(var));
                  auto dim = std::make_shared<DFA::LayerDimension>(var, final_dim_sz);
                  cluster_dimensions->AddDimension(dim);
                }

                //Restore output-centric info
                /*******************************************/
                //TODO: Update this from CONV-specific to general one
                int size_X, size_Y, size_R, size_S;
                for(auto& dim : *cluster_dimensions) {
                  auto var = dim->GetName();
                  if(var == DFSL::layer_dim_input_width_) {
                    size_X = dim->GetSize();
                  }
                  else if(var == DFSL::layer_dim_input_height_) {
                    size_Y = dim->GetSize();
                  }
                  else if(var == DFSL::layer_dim_weight_width_) {
                    size_S = dim->GetSize();
                  }
                  else if(var == DFSL::layer_dim_weight_height_) {
                    size_R = dim->GetSize();
                  }
                }

                int size_OX = size_X - size_S + 1;
                size_OX = (size_OX > 0)? size_OX : 1;

                int size_OY = size_Y - size_R + 1;
                size_OY = (size_OY > 0)? size_OY : 1;


                auto ox_dim = std::make_shared<DFA::LayerDimension>(DFSL::layer_dim_output_width_, size_OX);
                auto oy_dim = std::make_shared<DFA::LayerDimension>(DFSL::layer_dim_output_height_, size_OY);

                cluster_dimensions->AddDimension(ox_dim);
                cluster_dimensions->AddDimension(oy_dim);
                /*******************************************/

                prev_dataflow = cluster_dataflow;
/*
                //If directive is omitted, inherit dimension from upper cluster
                if(!var_list.empty()) {
                  for(auto var : var_list) {
                    auto dim = std::make_shared<DFA::LayerDimension>(var, prev_cluster_dimensions->GetSize(var));
                    cluster_dimensions->AddDimension(dim);
                  }
                }
*/
                cluster_dataflow = std::make_shared<DFA::DirectiveTable>();

                current_cluster_level--;
              } // End of case Cluster
                break;
              default: {
                error_handler->PrintErrorMsg(TL::ErrorCode::InvalidDirective,"", instance_name_);
              }
            }// End of switch(directive->GetClass)
          } // End of for-each directive in full_dataflow


          std::list<std::string> var_list;
          for(auto& tensor : *tensors_) {
            if(tensor->GetTensorClass() == TensorClass::InputTensor) {
              auto correlated_vars = tensor->GetCorrelatedVariables();

              for(auto var : *correlated_vars) {
                var_list.push_back(var);
              }
            }
          }
          var_list.sort();
          var_list.unique();


          ConvertToInputCentric(cluster_dataflow);

          // Inherit directives
          if(prev_dataflow != nullptr) {
            for(auto& upper_directive : *prev_dataflow) {
              auto upper_directive_var = upper_directive->GetVariable();
              if(prev_dataflow->FindDirective(upper_directive_var) == nullptr) {
                int upper_map_sz = upper_directive->GetSize();
                auto inherited_dataflow = std::make_shared<DFA::directive::TemporalMap>(upper_map_sz, upper_map_sz, upper_directive_var);
                cluster_dataflow->AddDirectiveFront(inherited_dataflow);
              }
            }
          }

          // Insert "Unroll" to still missing dimensions
          for(auto& var : var_list) {
            if(cluster_dataflow->FindDirective(var) == nullptr) {

              if(cluster_dimensions->HasVar(var)) {
                int dim_sz = cluster_dimensions->GetSize(var);
                auto dummy_directive = std::make_shared<DFA::directive::TemporalMap>(dim_sz, dim_sz, var);
                cluster_dataflow->AddDirectiveFront(dummy_directive);
              }
            }
          }


          // Inherit missing dimensions
          if(prev_cluster_dimensions != nullptr) {
            for(auto& dim : *prev_cluster_dimensions) {
              if(!cluster_dimensions->HasVar(dim->GetName())) {
                cluster_dimensions->AddDimension(dim);
              }
            }
          }

          std::cout << "Current cluster level: " << current_cluster_level <<  std::endl;

          auto new_cluster = std::make_shared<DFA::ClusterUnit>(current_cluster_level,
                                                                last_cluster_size,
                                                                cluster_dataflow,
                                                                cluster_dimensions,
                                                                tensors_,
                                                                nocs_->at(current_cluster_level));

          clusters_->PutCluster(new_cluster);


        } // End of function

        void ConvertToInputCentric(std::shared_ptr<DFA::DirectiveTable> dataflow) {

          int IX_size, IY_size, R_size, S_size, OX_size, OY_size;
          bool has_IY = false;
          bool has_IX = false;
          bool has_OY =false;
          bool has_OX =false;

          for(auto directive : *dataflow) {
            if(directive->GetVariable() == "Y") {
              has_IY = true;
              IY_size = directive->GetSize();
            }
            else if(directive->GetVariable() == "X") {
              has_IX = true;
              IX_size = directive->GetSize();
            }
            else if(directive->GetVariable() == "X'") {
              has_OX = true;
              OX_size = directive->GetSize();
            }
            else if(directive->GetVariable() == "Y'") {
              has_OY = true;
              OY_size = directive->GetSize();
            }
            else if(directive->GetVariable() == "R") {
              R_size = directive->GetSize();
            }
            else if(directive->GetVariable() == "S") {
              S_size = directive->GetSize();
            }
          }

          if(!has_IX && has_OX) {
            int directive_idx = dataflow->GetDirectiveIdx("X'");
            dataflow->at(directive_idx)->SetVariable("X");
            dataflow->at(directive_idx)->SetSize(OX_size + S_size - 1);
          }

          if(!has_IY && has_OY) {
            int directive_idx = dataflow->GetDirectiveIdx("Y'");
            dataflow->at(directive_idx)->SetVariable("Y");
            dataflow->at(directive_idx)->SetSize(OY_size + R_size - 1);
          }
        }

		}; // End of class ClusterMap
	}; // End of namespace DFA
}; // End of namespace maestro

#endif
