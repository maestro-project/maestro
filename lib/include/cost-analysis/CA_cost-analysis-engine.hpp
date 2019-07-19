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


#ifndef MAESTRO_CA_COST_ANALYSIS_ENGINE_HPP_
#define MAESTRO_CA_COST_ANALYSIS_ENGINE_HPP_

//#define DEBUG_COST_ANALYSIS

#include <iostream>
#include <memory>
#include <cmath>

#include "BASE_constants.hpp"

#include "BASE_maestro-class.hpp"
#include "TL_error-handler.hpp"

#include "DFA_cluster-unit.hpp"
#include "DFA_cluster-table.hpp"

#include "CA_iterations.hpp"
#include "CA_analysis-types.hpp"
#include "CA_reuse-analysis.hpp"
#include "CA_cost-analysis-results.hpp"

namespace maestro {
  namespace CA {

    class CostAnalysisEngine : public MAESTROClass {
      public:
        CostAnalysisEngine (
          std::shared_ptr<DFA::TensorTable> tensors,
          std::shared_ptr<DFA::ClusterTable> clusters,
          int num_simd_lanes
                             ) :
          tensors_(tensors),
          clusters_(clusters),
          num_simd_lanes_(num_simd_lanes),
          MAESTROClass("PerformanceAnalysis")
        {

        }

        void SetTargetCluster(std::shared_ptr<DFA::ClusterTable> clusters) {
          clusters_ = clusters;
        }

        std::shared_ptr<std::vector<std::shared_ptr<CostAnalyisResults>>> AnalyzeEntireCluster(bool verbose = false) {

          std::shared_ptr<std::vector<std::shared_ptr<CostAnalyisResults>>> ret = std::make_shared<std::vector<std::shared_ptr<CostAnalyisResults>>>();


          std::shared_ptr<CostAnalyisResults> prev_results = nullptr;
          std::shared_ptr<CostAnalyisResults> curr_results;

          int cluster_idx = 0;

          for(int cluster_idx = clusters_->size() -1; cluster_idx >= 0 ; cluster_idx --) {
            if(verbose) {
              std::cout << std::endl;
              std::cout << "Analyzing cluster lv " << clusters_->size()-1 - cluster_idx << std::endl;
              std::cout << std::endl;
            }
            int prev_delay = ((prev_results) == nullptr)? 0 : prev_results->GetRuntime();
            curr_results = AnalyzeSingleCluster(cluster_idx, prev_delay);

            ret->push_back(curr_results);
            prev_results = curr_results;
          }

          return ret;
        }

        std::shared_ptr<CostAnalyisResults> AnalyzeSingleCluster(int cluster_idx, int sub_cluster_delay = 0, bool do_double_buffering = true) {

          /* Base information */
          auto target_cluster = clusters_->GetCluster(cluster_idx);
          int num_sub_clusters = target_cluster->GetNumClusters();
          auto dimensions = target_cluster->GetDimensions();
          auto dataflow = target_cluster->GetDataflow();
          auto noc = target_cluster->GetNoCModel();

          auto output_tensors = tensors_->GetTensorsInClass(DFA::TensorClass::OutputTensor);
          auto input_tensors = tensors_->GetTensorsInClass(DFA::TensorClass::InputTensor);
          auto dim_var_list = target_cluster->GetAllDimensionVars();

          /* Intermediate analysis */
          auto reuse_analysis = std::make_shared<CA::ReuseAnalysis>(target_cluster);

          int buffer_size_mult = do_double_buffering? 2 : 1;

          auto results = std::make_shared<CostAnalyisResults>(clusters_->GetLayerType(), cluster_idx);
          results->UpdateNumSubClusters(target_cluster->GetNumClusters());

          /* Cost stats */
          long peak_noc_bw_req = 0;
          double avg_noc_bw_req = 0;

          /* Init run */
          int num_iter_cases = CA::num_iter_status;
          int num_cases = std::pow(num_iter_cases, dim_var_list->size());

          std::shared_ptr<std::vector<IterationStatus>> iter_status = std::make_shared<std::vector<IterationStatus>>();


          for(auto& tensor : *input_tensors) {
            auto dataclass = tensor->GetDataClass();
            auto coupled_vars = tensor->GetCoupledVariables();
            long size = 1;

            for(auto& dim : *coupled_vars) {
              size *= dimensions->GetSize(dim);
            }

            results->UpdateBufferAccessCount(BufferType::Upstream, BufferAccessType::Write, size, dataclass);


            auto downstream_buffer_req = reuse_analysis->GetMappedVolume(tensor);
            results->UpdateBufferSizeReq(BufferType::Downstream, buffer_size_mult * downstream_buffer_req, dataclass);
          }


          for(auto& tensor : *output_tensors) {
            auto dataclass = tensor->GetDataClass();
            auto coupled_vars = tensor->GetCoupledVariables();
            long size = 1;


            for(auto& dim : *coupled_vars) {

              if(dimensions->IsOverlapped(dim)) {
                auto overlapping_dim = dimensions->GetOverlappingDim(dim);
                int sliding_dim_size = dimensions->GetSize(overlapping_dim);
                int reference_dim_size = dimensions->GetSize(dim);;
                int actual_reference_dim_size = reference_dim_size;
                size *= reference_dim_size - sliding_dim_size + 1;
              }
              else {
                size *= dimensions->GetSize(dim);
              }
            }
            results->UpdateBufferAccessCount(BufferType::Upstream, BufferAccessType::Write, size, dataclass);


            auto downstream_buffer_req = reuse_analysis->GetMappedVolume(tensor);
            results->UpdateBufferSizeReq(BufferType::Downstream, buffer_size_mult * downstream_buffer_req, dataclass);
          }

          /* Main loop */
          for(int iter_status_case_num = 0; iter_status_case_num < num_cases; iter_status_case_num++) {
#ifdef DEBUG_COST_ANALYSIS
            std::cout << "Main Loop " << iter_status_case_num << std::endl;
#endif
            GetIterationCase(target_cluster, dataflow, iter_status_case_num, iter_status);
            long num_case_occurrences = GetNumCaseOccurrence(target_cluster, dataflow, iter_status);

            if(num_case_occurrences == 0) {
              continue;
            }

            long num_partial_outputs = GetNumOutputs(output_tensors, reuse_analysis, iter_status, true, false);
            long num_partial_outputs_min = GetNumOutputs(output_tensors, reuse_analysis, iter_status, true, false, CA::EstimationType::Min);
            long num_partial_outputs_max = GetNumOutputs(output_tensors, reuse_analysis, iter_status, true, false, CA::EstimationType::Max);

            long num_partial_outputs_sp_edge = GetNumOutputs(output_tensors, reuse_analysis, iter_status, true, true);
            long num_target_outputs = GetNumOutputs(output_tensors, reuse_analysis, iter_status, false, false);
            long num_target_outputs_min= GetNumOutputs(output_tensors, reuse_analysis, iter_status, false, false, CA::EstimationType::Min);
            long num_target_outputs_max = GetNumOutputs(output_tensors, reuse_analysis, iter_status, false, false, CA::EstimationType::Max);


            if(num_partial_outputs == 0) {
              continue;
            }

            long ingress_spatial_traffic = 0;
            long ingress_spatial_traffic_min = 0;
            long ingress_spatial_traffic_max = 0;

            for(auto& tensor : *input_tensors) {
              auto dataclass = tensor->GetDataClass();
              int dataclass_id = static_cast<int>(tensor->GetDataClass());
              long upstream_read_this_tensor = GetSpatialTrafficSingleTensor(tensor, reuse_analysis, iter_status);
              long upstream_read_this_tensor_min = GetSpatialTrafficSingleTensor(tensor, reuse_analysis, iter_status, CA::EstimationType::Min);
              long upstream_read_this_tensor_max = GetSpatialTrafficSingleTensor(tensor, reuse_analysis, iter_status, CA::EstimationType::Max);

#ifdef DEBUG_COST_ANALYSIS
              if(dataclass == DataClass::Weight) {
                std::cout << "Weight read: " << upstream_read_this_tensor << std::endl;
              }

              if(dataclass == DataClass::Input) {
                std::cout << "Input read: " << upstream_read_this_tensor << std::endl;
              }

              std::cout << "[CostAnalysisEngine] Upstream read += " << num_case_occurrences << " * " <<  upstream_read_this_tensor << " = " << num_case_occurrences * upstream_read_this_tensor << std::endl;
#endif

              //Buffer access count information update
              long prev_rd_count = results->GetBufferAccessCount(BufferType::Upstream, BufferAccessType::Read, dataclass);

              results->UpdateBufferAccessCount(BufferType::Upstream, BufferAccessType::Read, prev_rd_count + num_case_occurrences * upstream_read_this_tensor, dataclass);

              long num_active_sub_clusters = GetNumActiveSubClusters(iter_status, target_cluster);

              long prev_downstream_wr_count = results->GetBufferAccessCount(BufferType::Downstream, BufferAccessType::Write, dataclass);

              bool is_sp_mapped = false;
              auto coupled_dims = tensor->GetCoupledVariables();

              for(auto& dim : *coupled_dims) {
                if(dataflow->FindDirective(dim) != nullptr && dataflow->FindDirective(dim)->GetClass() == DFA::directive::DirectiveClass::SpatialMap ) {
                  is_sp_mapped = true;
                  break;
                }
              }


              if(is_sp_mapped) {
                auto first_sp_pe_load_volume = reuse_analysis->GetPELoadVolume(tensor,iter_status, true);
                //auto rest_sp_pe_load_volume = reuse_analysis->GetPELoadVolume(tensor,iter_status, false);
                results->UpdateBufferAccessCount(BufferType::Downstream, BufferAccessType::Write, prev_downstream_wr_count + num_case_occurrences * (first_sp_pe_load_volume  * num_active_sub_clusters), dataclass);
              }
              else {
                auto pe_load_volume = reuse_analysis->GetPELoadVolume(tensor,iter_status, true);

                results->UpdateBufferAccessCount(BufferType::Downstream, BufferAccessType::Write, prev_downstream_wr_count + num_case_occurrences * pe_load_volume * num_active_sub_clusters, dataclass);
              }
              ingress_spatial_traffic += upstream_read_this_tensor;
              ingress_spatial_traffic_min += upstream_read_this_tensor_min;
              ingress_spatial_traffic_max += upstream_read_this_tensor_max;

              long prev_downstream_rd_count = results->GetBufferAccessCount(BufferType::Downstream, BufferAccessType::Read, dataclass);
              results->UpdateBufferAccessCount(BufferType::Downstream, BufferAccessType::Read, prev_downstream_rd_count + num_active_sub_clusters * num_case_occurrences * num_partial_outputs, dataclass);

              if(iter_status_case_num == 0) {
                results->UpdateBufferSizeReq(BufferType::Upstream, buffer_size_mult * upstream_read_this_tensor, dataclass);
              }
            }

#ifdef DEBUG_COST_ANALYSIS
            std::cout << "ingress_spatial_traffic" << ingress_spatial_traffic << std::endl;
            std::cout << "Point D" << std::endl;
#endif

            long egress_spatial_traffic = 0;
            long egress_spatial_traffic_min = 0;
            long egress_spatial_traffic_max = 0;

            for(auto& tensor : *output_tensors) {
              auto dataclass = tensor->GetDataClass();
              int dataclass_id = static_cast<int>(tensor->GetDataClass());

              long upstream_write_this_tensor = num_target_outputs;
              long upstream_write_this_tensor_min = num_target_outputs_min;
              long upstream_write_this_tensor_max = num_target_outputs_max;

              long num_active_sub_clusters = GetNumActiveSubClusters(iter_status, target_cluster);

              long downstream_write_this_tensor = num_partial_outputs;
              long prev_downstream_wr_count = results->GetBufferAccessCount(BufferType::Downstream, BufferAccessType::Write, dataclass);
              results->UpdateBufferAccessCount(BufferType::Downstream, BufferAccessType::Write, prev_downstream_wr_count + num_case_occurrences * downstream_write_this_tensor, dataclass);

              long downstream_read_this_tensor = num_partial_outputs-1;
              long prev_downstream_rd_count = results->GetBufferAccessCount(BufferType::Downstream, BufferAccessType::Read, dataclass);
              results->UpdateBufferAccessCount(BufferType::Downstream, BufferAccessType::Read, prev_downstream_rd_count + num_case_occurrences * downstream_read_this_tensor, dataclass);

              egress_spatial_traffic += upstream_write_this_tensor;
              egress_spatial_traffic_min += upstream_write_this_tensor_min;
              egress_spatial_traffic_max += upstream_write_this_tensor_max;

              if(iter_status_case_num == 0) {
                results->UpdateBufferSizeReq(BufferType::Upstream, buffer_size_mult * upstream_write_this_tensor, dataclass);
              }
            }

            /* Delays */
            long comp_delay = (sub_cluster_delay == 0)? static_cast<long>(std::ceil(
                                static_cast<double>(num_partial_outputs) / static_cast<double>(num_simd_lanes_)))
                              : sub_cluster_delay;
            long ingress_comm_delay = noc->GetOutStandingDelay(ingress_spatial_traffic);
            long egress_comm_delay = noc->GetOutStandingDelay(egress_spatial_traffic);

            long actual_delay = (do_double_buffering)? std::max( egress_comm_delay, std::max(comp_delay, ingress_comm_delay)) : ingress_comm_delay + comp_delay + egress_comm_delay;


            long comp_delay_min = (sub_cluster_delay == 0)? static_cast<long>(std::ceil(
                                static_cast<double>(num_partial_outputs_min) / static_cast<double>(num_simd_lanes_)))
                              : sub_cluster_delay;
            long ingress_comm_delay_min = noc->GetOutStandingDelay(ingress_spatial_traffic_min);
            long egress_comm_delay_min = noc->GetOutStandingDelay(egress_spatial_traffic_min);

            long actual_delay_min = (do_double_buffering)? std::max( egress_comm_delay_min, std::max(comp_delay_min, ingress_comm_delay_min)) : ingress_comm_delay_min + comp_delay_min + egress_comm_delay_min;


            long comp_delay_max = (sub_cluster_delay == 0)? static_cast<long>(std::ceil(
                                static_cast<double>(num_partial_outputs_max) / static_cast<double>(num_simd_lanes_)))
                              : sub_cluster_delay;
            long ingress_comm_delay_max = noc->GetOutStandingDelay(ingress_spatial_traffic_max);
            long egress_comm_delay_max = noc->GetOutStandingDelay(egress_spatial_traffic_max);

            long actual_delay_max = (do_double_buffering)? std::max( egress_comm_delay_max, std::max(comp_delay_max, ingress_comm_delay_max)) : ingress_comm_delay_max + comp_delay_max + egress_comm_delay_max;



            if(iter_status_case_num != 0) {
              peak_noc_bw_req = std::max(peak_noc_bw_req, std::max(ingress_spatial_traffic, egress_spatial_traffic));
              avg_noc_bw_req = std::max(avg_noc_bw_req, peak_noc_bw_req/static_cast<double>(actual_delay));
            }


#ifdef DEBUG_COST_ANALYSIS
            if(actual_delay == egress_comm_delay) {
              std::cout << "Egress dominates: " << egress_comm_delay << std::endl;
            }
            else if(actual_delay == ingress_comm_delay) {
              std::cout << "Ingress dominates: " << ingress_comm_delay << std::endl;
            }
            else {
              std::cout << "computation dominates: " << comp_delay << std::endl;
            }

            std::cout << "Num Cases: " << num_case_occurrences << std::endl;

            std::cout << "num_partial_outputs: " << num_partial_outputs << std::endl;

#endif

            // Log the run time
            results->UpdateRuntime(results->GetRuntime(CA::EstimationType::Exact) + num_case_occurrences * actual_delay, CA::EstimationType::Exact);
            results->UpdateRuntime(results->GetRuntime(CA::EstimationType::Min) + num_case_occurrences * actual_delay_min, CA::EstimationType::Min);
            results->UpdateRuntime(results->GetRuntime(CA::EstimationType::Max) + num_case_occurrences * comp_delay_max, CA::EstimationType::Max);

            long num_active_sub_clusters = GetNumActiveSubClusters(iter_status, target_cluster);

//            std::cout << "num_active_sub_clusters: " << num_active_sub_clusters << std::endl;

            results->UpdateNumComputations(results->GetNumComputations() + num_case_occurrences * ((num_active_sub_clusters-1) * num_partial_outputs  + num_partial_outputs_sp_edge));

          } // End of for (iter_status_case_num)
#ifdef DEBUG_COST_ANALYSIS
          std::cout << "Num sub clusters at cluster level " << target_cluster->GetClusterLevel() << ": " << target_cluster->GetNumClusters(false) << std::endl;
#endif

          results->UpdatePeakBWReq(peak_noc_bw_req);
          results->UpdateAvgBWReq(avg_noc_bw_req);

          return results;
        }

      protected:
        std::shared_ptr<DFA::TensorTable> tensors_;
        std::shared_ptr<DFA::ClusterTable> clusters_;
        int num_simd_lanes_;

      private:

        std::shared_ptr<std::map<std::string, bool>> GetCleanTemporalEdgeInfo(
            std::shared_ptr<DFA::ClusterUnit> target_cluster) {
          std::shared_ptr<std::map<std::string, bool>> ret = std::make_shared<std::map<std::string, bool>>();
          auto dataflow = target_cluster->GetDataflow();

          auto vars = tensors_->GetTensorVarsInClass(DFA::TensorClass::InputTensor);

          for(auto var : *vars) {
            auto directive = dataflow->FindDirective(var);
            if(directive->GetClass() == DFA::directive::DirectiveClass::TemporalMap) {
              (*ret)[var] = false;
            }
          }

          return ret;
        }


        bool IsEdgeCaseValid(std::shared_ptr<DFA::ClusterUnit> target_cluster, std::shared_ptr<std::map<std::string, bool>> tp_edge_info) {
          bool ret = true;
          auto dataflow = target_cluster->GetDataflow();

          for(auto directive : *dataflow) {
            auto var = directive->GetVariable();
            if(directive->GetClass() == DFA::directive::DirectiveClass::TemporalMap) {
              if(target_cluster->GetNumTemporalIterations_SingleDim(var, (*tp_edge_info)[var] ) == 0) {
                ret = false;
              }
            }
          }

          return ret;
        }

        long GetNumOutputs(
            std::shared_ptr<std::list<std::shared_ptr<DFA::Tensor>>> output_tensors,
            std::shared_ptr<CA::ReuseAnalysis> reuse_analysis,
            std::shared_ptr<std::vector<IterationStatus>> iter_status,
            bool get_num_partial_sums = false,
            bool is_sp_edge_pe = false,
            CA::EstimationType estimation_type = CA::EstimationType::Exact) {
          long num_outputs = 0;

          for(auto tensor: *output_tensors) {
            num_outputs += reuse_analysis->GetNumOutputs(tensor, iter_status, get_num_partial_sums, is_sp_edge_pe, estimation_type);
          }

          return num_outputs;
        }

        long GetNumActiveSubClusters(
            std::shared_ptr<std::vector<IterationStatus>> iter_status,
            std::shared_ptr<DFA::ClusterUnit> target_cluster) {
          long ret = 0;

          int num_sub_clusters = target_cluster->GetNumClusters();
          auto dimensions = target_cluster->GetDimensions();
          auto dataflow = target_cluster->GetDataflow();


          for(int directive_id = 0; directive_id < iter_status->size(); directive_id++) {
            auto directive = dataflow->at(directive_id);
            auto directive_class = directive->GetClass();
            auto directive_dim = directive->GetVariable();

            if(directive_class == DFA::directive::DirectiveClass::SpatialMap) {
              auto iter_status_dim = iter_status->at(directive_id);
              if(iter_status_dim == CA::IterationStatus::Edge) {
                ret = target_cluster->GetNumClusters(true);
              }
              else if(iter_status_dim == CA::IterationStatus::Init || iter_status_dim == CA::IterationStatus::Unroll) {
                if(target_cluster->IsInitEdge(directive_dim)) {
                  ret = target_cluster->GetNumClusters(true);
                }
                else {
                  ret = target_cluster->GetNumClusters(false);
                }
              }
              else {
                ret = target_cluster->GetNumClusters(false);
              }
              break;
            }
          }

          return ret;

        }


        long GetSpatialTraffic(std::shared_ptr<std::list<std::shared_ptr<DFA::Tensor>>> tensors,
                               std::shared_ptr<CA::ReuseAnalysis> reuse_analysis,
                               std::shared_ptr<std::vector<IterationStatus>> iter_status) {
          long spatial_traffic = 0;

          for(auto tensor : *tensors) {
            spatial_traffic += GetSpatialTrafficSingleTensor(tensor, reuse_analysis, iter_status);
          }

          return spatial_traffic;
        }

        long GetSpatialTrafficSingleTensor(std::shared_ptr<DFA::Tensor> tensor,
                               std::shared_ptr<CA::ReuseAnalysis> reuse_analysis,
                               std::shared_ptr<std::vector<IterationStatus>> iter_status,
                               CA::EstimationType estimation_type = CA::EstimationType::Exact) {
          return reuse_analysis->GetSpatialLoadVolume(tensor, iter_status, estimation_type);
        }





        // Getting iter_status as argument to prevent frequent object creation/deletion
        void GetIterationCase(
            std::shared_ptr<DFA::ClusterUnit> target_cluster,
            std::shared_ptr<DFA::DirectiveTable> dataflow,
            int iter_status_case_num,
            std::shared_ptr<std::vector<IterationStatus>> iter_status)
        {
          iter_status->clear();
          int num_iter_cases = CA::num_iter_status;

          for (int directive_id = 0; directive_id < dataflow->size(); directive_id++) {
            int divider = std::pow(num_iter_cases, dataflow->size() - directive_id - 1);
            auto directive = dataflow->at(directive_id);
            auto dim = directive->GetVariable();

#ifdef DEBUG_COST_ANALYSIS
            std::cout << "dim " << dim << ":";
#endif

            switch ((iter_status_case_num / divider) % num_iter_cases) {
              case 0: {

                if(iter_status_case_num == 0) {
#ifdef DEBUG_COST_ANALYSIS
                  std::cout << "Init" << std::endl;
#endif
                  iter_status->push_back(IterationStatus::Init);
                }
                else {
                  if(target_cluster->IsUnrolled(dim)) {
#ifdef DEBUG_COST_ANALYSIS
                    std::cout << "Unroll" << std::endl;
#endif
                    iter_status->push_back(IterationStatus::Unroll);
                  }
                  else {
#ifdef DEBUG_COST_ANALYSIS
                    std::cout << "Init" << std::endl;
#endif
                    iter_status->push_back(IterationStatus::Init);
                  }
                }
                break;
              }
              case 1: {
#ifdef DEBUG_COST_ANALYSIS
                std::cout << "Steady" << std::endl;
#endif
                iter_status->push_back(IterationStatus::Steady);
                break;
              }
              case 2: {
#ifdef DEBUG_COST_ANALYSIS
                std::cout << "Edge" << std::endl;
#endif
                iter_status->push_back(IterationStatus::Edge);
                break;
              }
              default: {
                error_handler_->PrintErrorMsg(TL::ErrorCode::InvalidAnalysisCase, "");
              }
            }
         }
       }


        std::shared_ptr<CA::IterationStatusInfo> GetIterationInfo(
            std::shared_ptr<DFA::ClusterUnit> target_cluster,
            std::shared_ptr<DFA::DirectiveTable> dataflow,
            int iter_status_case_num,
            std::shared_ptr<std::vector<IterationStatus>> iter_status) {
          auto ret = std::make_shared<CA::IterationStatusInfo>();
          auto dimensions = target_cluster->GetDimensions();

          for(auto& directive : *dataflow) {
            auto status_list = std::make_shared<std::list<CA::IterationStatus>>();

            auto dim = directive->GetVariable();
            int dim_size = dimensions->GetSize(dim);
            int map_size = directive->GetSize();
            int ofs_size = directive->GetOfs();

            //TODO: Need to consider overlapping dimensions
            if(directive->GetClass() == DFA::directive::DirectiveClass::TemporalMap) {
              if(map_size == dim_size) { //TODO: verify this
                status_list->push_back(CA::IterationStatus::InitUnroll);
              }
              else if(map_size > dim_size) {
                status_list->push_back(CA::IterationStatus::InitEdge);
              }
              else {
                status_list->push_back(CA::IterationStatus::Init);
                if(map_size + ofs_size < dim_size) {
                  status_list->push_back(CA::IterationStatus::Steady);
                }

                if(dim_size % ofs_size != 0) {
                  status_list->push_back(CA::IterationStatus::Edge);
                }
                else {
                  if( (dim_size / ofs_size) * (ofs_size-1) + map_size > dim_size) {
                    status_list->push_back(CA::IterationStatus::Edge);
                  }
                }
              } // End of else, which indicates if (map_size < dim_size)
            } // End of if(this directive class == TemporalMap)
            else if(directive->GetClass() == DFA::directive::DirectiveClass::SpatialMap) {
              int num_clusters = target_cluster->GetNumClusters(false);
              int num_edge_clusters = target_cluster->GetNumClusters(true);
              int first_sp_iteration_coverage = (num_clusters-1) * ofs_size + map_size;
              int first_edge_sp_iteration_coverage = (num_edge_clusters-1) * ofs_size + map_size;

              if(first_sp_iteration_coverage == dim_size) {
                status_list->push_back(CA::IterationStatus::InitUnroll);
              }
              else if(first_sp_iteration_coverage > dim_size) {
                status_list->push_back(CA::IterationStatus::InitEdge);
              }
              else {
                int second_sp_itreation_coverage = 2 * (num_clusters-1) * ofs_size + ofs_size + map_size;

                status_list->push_back(CA::IterationStatus::Init);

                if(second_sp_itreation_coverage < dim_size) {
                  status_list->push_back(CA::IterationStatus::Steady);
                }

                if((dim_size - map_size) % ofs_size != 0) {
                  status_list->push_back(CA::IterationStatus::Edge);
                }
              }

              /*
              else if(first_sp_iteration_coverage < dim_size) {
                if(first_edge_sp_iteration_coverage > dim_size) {
                  status_list->push_back(CA::IterationStatus::InitEdgeSpEdge);
                }
                else if (first_edge_sp_iteration_coverage == dim_size ) {
                  status_list->push_back(CA::IterationStatus::InitEdge);
                }
                else {
                  //TODO: In this case, the number of edge clusters computation is wrong. Add error message for this
//                  this->error_handler_->PrintErrorMsg();
                  this->error_handler_->TerminateProgram();
                }
              }
              else {
                int second_sp_itreation_coverage = 2 * (num_clusters-1) * ofs_size + map_size;


                status_list->push_back(CA::IterationStatus::Init);

                if(second_sp_itreation_coverage < dim_size) {
                  status_list->push_back(CA::IterationStatus::Steady);
                }

                  TODO
                //int second_last_sp_iteration_converage = dim_size % (num_clusters * ofs_size) - (map_size - ofs_size);

                //if(dim_size % ((num_clusters) * ofs_size) != 0) {
                //  status_list->push_back(CA::IterationStatus::Edge);
                //}
//
              }
              */
            }
            else {
              this->error_handler_->PrintErrorMsg(TL::ErrorCode::InvalidCluster, std::to_string(target_cluster->GetClusterLevel()), this->GetName());
              this->error_handler_->TerminateProgram();
            }


            ret->AddIterationInfo(std::make_pair(dim, status_list));
          }

          return ret;
        }


       long GetNumCaseOccurrence(
           std::shared_ptr<DFA::ClusterUnit> target_cluster,
           std::shared_ptr<DFA::DirectiveTable> dataflow,
           std::shared_ptr<std::vector<IterationStatus>> iter_status)
       {
         long num_case_occurrences = 1;


         for(int directive_id = 0; directive_id < iter_status->size(); directive_id++) {
           auto directive = dataflow->at(directive_id);
           auto directive_class = directive->GetClass();
           auto dim = directive->GetVariable();
           bool has_tp_edge = target_cluster->HasTemporalEdgeCase(dim);
           bool has_sp_edge = target_cluster->HasSpatialEdgeCase();
           auto dim_iter_status = iter_status->at(directive_id);

           switch(dim_iter_status) {
             case IterationStatus::Steady: {
               if(target_cluster->IsUnrolled(dim)) {
                 num_case_occurrences = 0;
               }
               else {
                 if(directive_class == DFA::directive::DirectiveClass::TemporalMap) {
                   if(has_tp_edge) {
                     num_case_occurrences *= target_cluster->GetNumTemporalIterations(dim) - 2; // Subtract first (init) and last (edge)
                   }
                   else {
                     num_case_occurrences *= target_cluster->GetNumTemporalIterations(dim) - 1; // Subtract first
                   }
                 }
                 else {
                   if(has_sp_edge) {
                     num_case_occurrences *= target_cluster->GetNumSpatialIterations(dim) - 2;
                   }
                   else {
                     num_case_occurrences *= target_cluster->GetNumSpatialIterations(dim) -1;
                   }
                 }
               }

               break;
             }
             case IterationStatus::Edge: {
               if(directive_class == DFA::directive::DirectiveClass::TemporalMap) {
                 if(!has_tp_edge || target_cluster->IsInitEdge(dim) || target_cluster->IsInitUnroll(dim)) {
                   num_case_occurrences = 0;
                 }
               }
               else {
                 if(!has_sp_edge || target_cluster->IsInitEdge(dim) || target_cluster->IsInitUnroll(dim)) {
                   num_case_occurrences = 0;
                 }
               }

               break;
             }
             case IterationStatus::Init: {
               num_case_occurrences *= 1;
               break;
             }
             default: { //Init/Unroll occurs only once
               num_case_occurrences *= 1;
             }
           }
         }

#ifdef DEBUG_COST_ANALYSIS
         std::cout << "Num cases: " << num_case_occurrences << std::endl;
#endif

         return num_case_occurrences;
       }

    }; // End of class PerformanceAnalysis
  }; // End of namespace CA
}; // End of namespace maestro

#endif
