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
#include "DFA_iteration-status.hpp"
#include "DFA_iteration-analysis.hpp"

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

        CostAnalysisEngine (
          std::shared_ptr<ConfigurationV2> configs,
          std::shared_ptr<DFA::TensorTable> tensors,
          std::shared_ptr<DFA::ClusterTable> clusters
                             ) :
          configs_(configs),
          tensors_(tensors),
          clusters_(clusters),
          num_simd_lanes_(configs->target_accelerator_->GetVectorWidth()),
          MAESTROClass("PerformanceAnalysis")
        {

        }


        void SetTargetCluster(std::shared_ptr<DFA::ClusterTable> clusters) {
          clusters_ = clusters;
        }

        std::shared_ptr<std::vector<std::shared_ptr<CostAnalyisResults>>> AnalyzeEntireCluster(bool write_log_file = false) {

          std::shared_ptr<std::vector<std::shared_ptr<CostAnalyisResults>>> ret = std::make_shared<std::vector<std::shared_ptr<CostAnalyisResults>>>();

          AnalyzeClusterLevel_V2(0, clusters_->size(), clusters_->GetCluster(0)->GetDimensions(), ret, 1, true, write_log_file);

          return ret;
        }

        void AnalyzeClusterLevel_V2(
            int cluster_idx,
            int num_cluster_lvs,
            std::shared_ptr<DFA::DimensionTable> dimensions,
            std::shared_ptr<std::vector<std::shared_ptr<CostAnalyisResults>>> ret,
            int print_cluster_lv = 0,
            bool do_double_buffering = true,
            bool write_log_file = false) {

          /* Base information */
          std::shared_ptr<DFA::ClusterUnit> target_cluster = clusters_->GetCluster(cluster_idx);
          int num_sub_clusters = target_cluster->GetNumClusters(false);
          int num_edge_clusters = target_cluster->GetNumClusters(true);
          int num_active_clusters = num_sub_clusters;
          auto dataflow = target_cluster->GetDataflow();
          auto noc = target_cluster->GetNoCModel();

          auto output_tensors = tensors_->GetTensorsInClass(DFA::TensorClass::OutputTensor);
          auto input_tensors = tensors_->GetTensorsInClass(DFA::TensorClass::InputTensor);

          std::ofstream log_file;

          if(write_log_file && cluster_idx <= print_cluster_lv) {
            log_file.open("log.txt",std::fstream::in | std::fstream::out | std::fstream::app);
            log_file << "=======================" << std::endl;
            log_file << "Log for cluster level " << cluster_idx << std::endl;
            log_file << "=======================" << std::endl;

            log_file << "Num sub clusters: " << num_sub_clusters << std::endl;

            log_file << "Cluster tile size" << std::endl;
            log_file << dimensions->ToString() << std::endl;

            log_file << "Cluster Dataflow" << std::endl;
            log_file << dataflow->ToString() << std::endl;
          }

          /* Intermediate analysis */
          auto reuse_analysis = std::make_shared<CA::ReuseAnalysis>(target_cluster, write_log_file);
          auto results = std::make_shared<CostAnalyisResults>(clusters_->GetLayerType(), cluster_idx);
          results->UpdateNumSubClusters(target_cluster->GetNumClusters());

          /* Cost stats */
          long peak_noc_bw_req = 0;
          long double avg_noc_bw_req = 0;

          long double delays[static_cast<int>(DelayType::NumDelayTypes)][static_cast<int>(ValueType::NumValTypes)];
          for(int i = 0; i < static_cast<int>(DelayType::NumDelayTypes); i++) {
            delays[i][static_cast<int>(ValueType::Min)] = std::numeric_limits<long double>::max();
            delays[i][static_cast<int>(ValueType::Max)] = std::numeric_limits<long double>::min();
            delays[i][static_cast<int>(ValueType::Avg)] = 0;
          }

          auto iteration_analysis = std::make_unique<DFA::IterationAnalysis>(dimensions, target_cluster);
          std::shared_ptr<std::vector<std::shared_ptr<DFA::IterationStatus>>> all_iteration_cases = iteration_analysis->GetAllIterationsStatus();

          //Set the buffer size based on the worst case
          // TODO: Apply case-based analysis
//          UpdateBufferSizeReq(results, dimensions, reuse_analysis, do_double_buffering);

          long num_total_cases = 0;
          int case_id = 0;
          for(auto& iteration_case : *all_iteration_cases) {
            if(case_id == 0) {
              UpdateBufferSizeReq(results, dimensions, reuse_analysis, iteration_case, cluster_idx, num_cluster_lvs, do_double_buffering);
            }

            long num_case_occurrences = iteration_case->GetNumOccurrences();
            assert(num_case_occurrences > 0);

            if(write_log_file && cluster_idx <= print_cluster_lv) {
              log_file << "======================= CASE " << case_id << " =======================" << std::endl;
              log_file << "@ cluster level " << cluster_idx << std::endl;
              log_file << iteration_case->ToString() << std::endl;
            }

            long ingress_spatial_traffic = 0;
            long egress_spatial_traffic = 0;

            long num_partial_sums = 0;
            long tensor_spatial_partial_sum_mapping_size = 0;
            for(auto& tensor : *output_tensors) {
              long tensor_egress_traffic = reuse_analysis->GetSpatialEgressTraffic(tensor, iteration_case);
              long tensor_spatial_mapping_size = reuse_analysis->GetOutputTensorSpatialMappingSize(tensor, iteration_case);
              tensor_spatial_partial_sum_mapping_size += reuse_analysis->GetOutputTensorSpatialMappingSize(tensor, iteration_case, true);
              auto data_class = tensor->GetDataClass();

              if(write_log_file && cluster_idx <= print_cluster_lv) {
                log_file << "Output Tensor " << tensor->GetTensorName() << std::endl;
                log_file << "\tegress_traffic " << tensor_egress_traffic << std::endl;
                log_file << "\tspatial_mapping_size" << tensor_spatial_mapping_size << std::endl;
                log_file << "\ttensor_spatial_partial_sum_mapping_size" << tensor_spatial_partial_sum_mapping_size << std::endl;
              }


              egress_spatial_traffic += tensor_egress_traffic;

              long upstream_write_this_tensor = tensor_egress_traffic;

              long prev_upstream_wr_count = results->GetBufferAccessCount(BufferType::Upstream, BufferAccessType::Write, data_class);
              results->UpdateBufferAccessCount(BufferType::Upstream, BufferAccessType::Write, prev_upstream_wr_count + num_case_occurrences * upstream_write_this_tensor, data_class);

              long prev_downstream_wr_count = results->GetBufferAccessCount(BufferType::Downstream, BufferAccessType::Write, data_class);
              results->UpdateBufferAccessCount(BufferType::Downstream, BufferAccessType::Write, prev_downstream_wr_count + num_case_occurrences * tensor_spatial_partial_sum_mapping_size, data_class);

              long prev_downstream_rd_count = results->GetBufferAccessCount(BufferType::Downstream, BufferAccessType::Read, data_class);
              results->UpdateBufferAccessCount(BufferType::Downstream, BufferAccessType::Read, prev_downstream_rd_count + num_case_occurrences * tensor_spatial_partial_sum_mapping_size, data_class);
              num_partial_sums += reuse_analysis->GetNumCriticalPathPartialSums(tensor, iteration_case);
            }


            if(num_partial_sums <= 0) {
              std::cout << "Num partial sums is less than 0!" << std::endl;
              if(write_log_file && cluster_idx <= print_cluster_lv) {
                log_file << "Skipping Invalid case" << std::endl;
              }
              continue;
            }


            for(auto& tensor : *input_tensors) {
              long tensor_ingress_traffic = reuse_analysis->GetSpatialIngressTraffic(tensor, iteration_case);
              long tensor_spatial_mapping_size = reuse_analysis->GetInputTensorSpatialMappingSize(tensor, iteration_case);
              auto data_class = tensor->GetDataClass();

              if(write_log_file && cluster_idx <= print_cluster_lv) {
                log_file << "Input Tensor " << tensor->GetTensorName() << std::endl;
                log_file << "\tingress_traffic " << tensor_ingress_traffic << std::endl;
                log_file << "\tspatial_mapping_size " << tensor_spatial_mapping_size << std::endl;
              }

              ingress_spatial_traffic += tensor_ingress_traffic;

              long upstream_read_this_tensor = tensor_ingress_traffic;

              long prev_rd_count = results->GetBufferAccessCount(BufferType::Upstream, BufferAccessType::Read, data_class);
              results->UpdateBufferAccessCount(BufferType::Upstream, BufferAccessType::Read, prev_rd_count + num_case_occurrences * upstream_read_this_tensor, data_class);

              long prev_downstream_wr_count = results->GetBufferAccessCount(BufferType::Downstream, BufferAccessType::Write, data_class);
              results->UpdateBufferAccessCount(BufferType::Downstream, BufferAccessType::Write, prev_rd_count + num_case_occurrences * upstream_read_this_tensor, data_class);

              long prev_downstream_rd_count = results->GetBufferAccessCount(BufferType::Downstream, BufferAccessType::Read, data_class);
              results->UpdateBufferAccessCount(BufferType::Downstream, BufferAccessType::Read, prev_downstream_rd_count + num_case_occurrences * tensor_spatial_partial_sum_mapping_size, data_class);
            }

            double arithmetic_intensity = static_cast<double>(tensor_spatial_partial_sum_mapping_size)/static_cast<double>(ingress_spatial_traffic);

            results->SetArithmeticIntensity(arithmetic_intensity);


            //TODO: Exactly model cross-PE accumulation

            if(write_log_file && cluster_idx <= print_cluster_lv) {
              log_file << "Overall ingress_spatial_traffic: " << ingress_spatial_traffic << std::endl;
              log_file << "Overall egress_spatial_traffic: " << egress_spatial_traffic << std::endl;
              log_file << "Number of MACs over sub cluster array: " << tensor_spatial_partial_sum_mapping_size << std::endl;
            }

            ////////////////////////////

            long computation_delay = 0;
            std::shared_ptr<std::vector<std::shared_ptr<CA::CostAnalyisResults>>> sub_cluster_results = std::make_shared<std::vector<std::shared_ptr<CA::CostAnalyisResults>>>();

            std::shared_ptr<DFA::directive::Directive> spmap_directive = nullptr;
            for(auto& directive : *dataflow) {
              if(directive->GetClass() == DFA::directive::DirectiveClass::SpatialMap) {
                spmap_directive = directive;
                break;
              }
            }

            if(spmap_directive == nullptr) {
              error_handler_->PrintErrorMsg(TL::ErrorCode::NoSpatialMap, std::to_string(cluster_idx) ,this->GetName());
              error_handler_->TerminateProgram();
            }

            auto spmap_dim_iter_state = iteration_case->GetIterState(spmap_directive->GetVariable());

            if(spmap_dim_iter_state->IsEdge()) {
              num_active_clusters = num_edge_clusters;
            }
            else {
              num_active_clusters = num_sub_clusters;
            }

            // Recursively process subclusters
            if(cluster_idx < num_cluster_lvs-1) {
              if(spmap_dim_iter_state->IsEdge()) {
                if(spmap_dim_iter_state->HasSpEdgeEdge()) {
                  auto subclsuter_dim_under_sp_edge_edge = reuse_analysis->ConstructSubClusterDimension(iteration_case, true);
                  AnalyzeClusterLevel_V2(cluster_idx+1, num_cluster_lvs, subclsuter_dim_under_sp_edge_edge, ret, print_cluster_lv, do_double_buffering, write_log_file);
                  auto sp_edge_edge_subcluster_res = ret->at(ret->size()-1);
                  sub_cluster_results->push_back(sp_edge_edge_subcluster_res);

                  int num_rem_clusters = num_edge_clusters-1;
                  if(num_rem_clusters > 0 ) {
                    auto this_subclsuter_dim = reuse_analysis->ConstructSubClusterDimension(iteration_case, false);
                    AnalyzeClusterLevel_V2(cluster_idx+1, num_cluster_lvs, this_subclsuter_dim, ret, print_cluster_lv,do_double_buffering, write_log_file);
                    auto this_subcluster_res = ret->at(ret->size()-1);
                    this_subcluster_res->SetNumSpatialOccurrences(num_rem_clusters);
                    sub_cluster_results->push_back(this_subcluster_res);
                  }
                } // End of if(spmap_dim_iter_state->HasSpEdgeEdge())
                else {
                  auto this_subclsuter_dim = reuse_analysis->ConstructSubClusterDimension(iteration_case, false);
                  AnalyzeClusterLevel_V2(cluster_idx+1, num_cluster_lvs, this_subclsuter_dim, ret, print_cluster_lv, do_double_buffering, write_log_file);
                  auto this_subcluster_res = ret->at(ret->size()-1);
                  this_subcluster_res->SetNumSpatialOccurrences(num_edge_clusters);
                  sub_cluster_results->push_back(this_subcluster_res);
                } // End of else of if(spmap_dim_iter_state->HasSpEdgeEdge())
              } // End of if(spmap_dim_iter_state->IsEdge())
              else {
                auto this_subclsuter_dim = reuse_analysis->ConstructSubClusterDimension(iteration_case, false);
                AnalyzeClusterLevel_V2(cluster_idx+1, num_cluster_lvs, this_subclsuter_dim, ret, print_cluster_lv, do_double_buffering, write_log_file);
                auto this_subcluster_res = ret->at(ret->size()-1);
                this_subcluster_res->SetNumSpatialOccurrences(num_sub_clusters);
                sub_cluster_results->push_back(this_subcluster_res);
              }

              // Take the worst-case delay as the computation delay
              for(auto& sub_res : *sub_cluster_results) {
                computation_delay = std::max(computation_delay, sub_res->GetRuntime());
              }

            } // End of if(cluster_idx < num_cluster_lvs-1)
            else { // Base cluster
              computation_delay =static_cast<long>(
                  std::ceil(static_cast<double>(num_partial_sums) / static_cast<double>(num_simd_lanes_)));
            }
            ////////////////////////////

            long ingress_comm_delay = noc->GetOutStandingDelay(ingress_spatial_traffic);
            long egress_comm_delay = noc->GetOutStandingDelay(egress_spatial_traffic);

            long outstanding_delay;
            if(iteration_case->isAllInit()) {
              outstanding_delay = (do_double_buffering)? computation_delay + ingress_comm_delay : ingress_comm_delay + computation_delay + egress_comm_delay;
            }
            else {
              outstanding_delay = (do_double_buffering)? std::max( egress_comm_delay, std::max(computation_delay, ingress_comm_delay)) : ingress_comm_delay + computation_delay + egress_comm_delay;
            }

            results->UpdateRuntime(results->GetRuntime(CA::EstimationType::Exact) + num_case_occurrences * outstanding_delay, CA::EstimationType::Exact);
            results->UpdateNumComputations(results->GetNumComputations() + num_case_occurrences * tensor_spatial_partial_sum_mapping_size);


            long double num_active_unit_clusters = 0;
            for(auto& sub_res : * sub_cluster_results) {
              num_active_unit_clusters += sub_res->GetNumAvgActiveClusters() * sub_res->GetNumSpatialOccurrences();
            }

            num_active_unit_clusters = (num_active_unit_clusters== 0)? num_active_clusters : num_active_unit_clusters;
            results->SetNumAvgActiveClusters(results->GetNumAvgActiveClusters() + num_active_unit_clusters * num_case_occurrences);

            //TODO: Doble check
            if(computation_delay == 0)
              computation_delay = 1;

            peak_noc_bw_req = std::max(peak_noc_bw_req, std::max(ingress_spatial_traffic, egress_spatial_traffic)/computation_delay);
            avg_noc_bw_req += (num_case_occurrences * std::max(ingress_spatial_traffic, egress_spatial_traffic))/computation_delay;


            delays[static_cast<int>(DelayType::Ingress)][static_cast<int>(ValueType::Avg)] += num_case_occurrences * ingress_comm_delay;
            delays[static_cast<int>(DelayType::Egress)][static_cast<int>(ValueType::Avg)] += num_case_occurrences * egress_comm_delay;
            delays[static_cast<int>(DelayType::Computation)][static_cast<int>(ValueType::Avg)] += num_case_occurrences * computation_delay;

            delays[static_cast<int>(DelayType::Ingress)][static_cast<int>(ValueType::Min)]
              = std::min(delays[static_cast<int>(DelayType::Ingress)][static_cast<int>(ValueType::Min)], static_cast<long double>(ingress_comm_delay)) ;
            delays[static_cast<int>(DelayType::Egress)][static_cast<int>(ValueType::Min)]
              = std::min(delays[static_cast<int>(DelayType::Egress)][static_cast<int>(ValueType::Min)], static_cast<long double>(egress_comm_delay)) ;
            delays[static_cast<int>(DelayType::Computation)][static_cast<int>(ValueType::Min)]
              = std::min(delays[static_cast<int>(DelayType::Computation)][static_cast<int>(ValueType::Min)], static_cast<long double>(computation_delay));

            delays[static_cast<int>(DelayType::Ingress)][static_cast<int>(ValueType::Max)]
              = std::max(delays[static_cast<int>(DelayType::Ingress)][static_cast<int>(ValueType::Max)], static_cast<long double>(ingress_comm_delay)) ;
            delays[static_cast<int>(DelayType::Egress)][static_cast<int>(ValueType::Max)]
              = std::max(delays[static_cast<int>(DelayType::Egress)][static_cast<int>(ValueType::Max)], static_cast<long double>(egress_comm_delay)) ;
            delays[static_cast<int>(DelayType::Computation)][static_cast<int>(ValueType::Max)]
              = std::max(delays[static_cast<int>(DelayType::Computation)][static_cast<int>(ValueType::Max)], static_cast<long double>(computation_delay));

            num_total_cases += num_case_occurrences;
            case_id++;

            if(write_log_file && cluster_idx <= print_cluster_lv) {
              if(iteration_case->isAllInit()) {
                log_file << "Note: Initialization case; cannot exploit latency hiding in this case" << std::endl;
              }

              log_file << "num computations (per iteration): " << tensor_spatial_partial_sum_mapping_size << std::endl;
              log_file << "ingress_spatial_traffic (per iteration): " << ingress_spatial_traffic << std::endl;
              log_file << "egress_spatial_traffic (per iteration): " << egress_spatial_traffic << std::endl;
              log_file << std::endl;

              log_file << "ingress_comm_delay (per iteration): " << ingress_comm_delay << std::endl;
              log_file << "egress_comm_delay (per iteration): " << egress_comm_delay << std::endl;
              log_file << "computation_delay (per iteration): " << computation_delay << std::endl;
              log_file << std::endl;

              if(do_double_buffering) {
                if(iteration_case->isAllInit()){
                  log_file << "This case is <<Ingress communication + computation>> bound (Initialization case)" << std::endl;
                }
                else if(outstanding_delay == computation_delay) {
                  log_file << "This case is <<Computation>> bound" << std::endl;
                }
                else if(outstanding_delay == ingress_comm_delay) {
                  log_file << "This case is <<Ingress communication>> bound" << std::endl;
                }
                else if(outstanding_delay == egress_comm_delay) {
                  log_file << "This case is <<Egress communication>> bound" << std::endl;
                }
              }

              log_file << "outstanding_delay (per iteration): " << outstanding_delay << std::endl;
              log_file << "outstanding_delay (for all iterations in this case): " << num_case_occurrences * outstanding_delay << std::endl;
              log_file << "======================= END CASE " << case_id << " =======================\n\n" << std::endl;
            }
          } // End of for_each (iteration_case) in (all_iteration_cases)
          avg_noc_bw_req = avg_noc_bw_req / num_total_cases;

          if(num_total_cases != 0) {
            results->SetNumAvgActiveClusters(results->GetNumAvgActiveClusters() / num_total_cases);
          }
          results->UpdatePeakBWReq(peak_noc_bw_req);
          results->UpdateAvgBWReq(avg_noc_bw_req);

          delays[static_cast<int>(DelayType::Ingress)][static_cast<int>(ValueType::Avg)] /= num_total_cases;
          delays[static_cast<int>(DelayType::Egress)][static_cast<int>(ValueType::Avg)] /= num_total_cases;
          delays[static_cast<int>(DelayType::Computation)][static_cast<int>(ValueType::Avg)] /= num_total_cases;

          results->UpdateDelay(DelayType::Ingress, ValueType::Avg, delays[static_cast<int>(DelayType::Ingress)][static_cast<int>(ValueType::Avg)]);
          results->UpdateDelay(DelayType::Egress, ValueType::Avg, delays[static_cast<int>(DelayType::Egress)][static_cast<int>(ValueType::Avg)]);
          results->UpdateDelay(DelayType::Computation, ValueType::Avg, delays[static_cast<int>(DelayType::Computation)][static_cast<int>(ValueType::Avg)]);

          results->UpdateDelay(DelayType::Ingress, ValueType::Min, delays[static_cast<int>(DelayType::Ingress)][static_cast<int>(ValueType::Min)]);
          results->UpdateDelay(DelayType::Egress, ValueType::Min, delays[static_cast<int>(DelayType::Egress)][static_cast<int>(ValueType::Min)]);
          results->UpdateDelay(DelayType::Computation, ValueType::Min, delays[static_cast<int>(DelayType::Computation)][static_cast<int>(ValueType::Min)]);

          results->UpdateDelay(DelayType::Ingress, ValueType::Max, delays[static_cast<int>(DelayType::Ingress)][static_cast<int>(ValueType::Max)]);
          results->UpdateDelay(DelayType::Egress, ValueType::Max, delays[static_cast<int>(DelayType::Egress)][static_cast<int>(ValueType::Max)]);
          results->UpdateDelay(DelayType::Computation, ValueType::Max, delays[static_cast<int>(DelayType::Computation)][static_cast<int>(ValueType::Max)]);

          ret->push_back(results);
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

          auto iteration_analysis = std::make_unique<DFA::IterationAnalysis>(dimensions, target_cluster);


          /* Init run */
          int num_iter_cases = CA::num_iter_status;
          int num_cases = std::pow(num_iter_cases, dim_var_list->size());

          std::shared_ptr<std::vector<IterationStatus>> iter_status = std::make_shared<std::vector<IterationStatus>>();

//          UpdateBufferSizeReq(results, dimensions, reuse_analysis, do_double_buffering);

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

#ifdef DEBUG_COST_ANALYSIS
            std::cout << "egress_spatial_traffic " << egress_spatial_traffic << std::endl;
            std::cout << "Point E " << std::endl;
#endif


            /* Delays */
            long comp_delay = (sub_cluster_delay == 0)? static_cast<long>(std::ceil(
                                static_cast<double>(num_partial_outputs) / static_cast<double>(num_simd_lanes_)))
                              : sub_cluster_delay;
            long ingress_comm_delay = noc->GetOutStandingDelay(ingress_spatial_traffic);
            long egress_comm_delay = noc->GetOutStandingDelay(egress_spatial_traffic);

            long actual_delay = (do_double_buffering)? std::max( egress_comm_delay, std::max(comp_delay, ingress_comm_delay)) : ingress_comm_delay + comp_delay + egress_comm_delay;
//            long actual_delay = (do_double_buffering)? std::max( egress_comm_delay, comp_delay + ingress_comm_delay) : ingress_comm_delay + comp_delay + egress_comm_delay;


            long ingress_max = results->GetDelay(DelayType::Ingress,ValueType::Max);
            if(ingress_comm_delay > ingress_max) results->UpdateDelay(DelayType::Ingress, ValueType::Max, ingress_comm_delay);

            long ingress_min = results->GetDelay(DelayType::Ingress,ValueType::Min);
            if(ingress_comm_delay < ingress_min) results->UpdateDelay(DelayType::Ingress, ValueType::Min, ingress_comm_delay);


            long egress_max = results->GetDelay(DelayType::Egress,ValueType::Max);
            if(egress_comm_delay > egress_max) results->UpdateDelay(DelayType::Egress, ValueType::Max, egress_comm_delay);

            long egress_min = results->GetDelay(DelayType::Egress,ValueType::Min);
            if(egress_comm_delay < egress_min) results->UpdateDelay(DelayType::Egress, ValueType::Min, egress_comm_delay);

            long compute_max = results->GetDelay(DelayType::Computation,ValueType::Max);
            if(comp_delay > compute_max) results->UpdateDelay(DelayType::Computation, ValueType::Max, comp_delay);

            long compute_min = results->GetDelay(DelayType::Computation,ValueType::Min);
            if(comp_delay < compute_min) results->UpdateDelay(DelayType::Computation, ValueType::Min, comp_delay);



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
        std::shared_ptr<ConfigurationV2> configs_;
        std::shared_ptr<DFA::TensorTable> tensors_;
        std::shared_ptr<DFA::ClusterTable> clusters_;
        int num_simd_lanes_;

      private:

        void UpdateBufferSizeReq(
            std::shared_ptr<CostAnalyisResults> results,
            std::shared_ptr<DFA::DimensionTable> dimensions,
            std::shared_ptr<CA::ReuseAnalysis> reuse_analysis,
            std::shared_ptr<DFA::IterationStatus> iter_status,
            int cluster_idx,
            int num_cluster_lvs,
            bool do_double_buffering)
        {
          auto output_tensors = tensors_->GetTensorsInClass(DFA::TensorClass::OutputTensor);
           auto input_tensors = tensors_->GetTensorsInClass(DFA::TensorClass::InputTensor);

           int buffer_size_mult = do_double_buffering? 2 : 1;

           for(auto& tensor : *input_tensors) {
             auto dataclass = tensor->GetDataClass();
             auto coupled_vars = tensor->GetCoupledVariables();
             long size = 1;

             for(auto& dim : *coupled_vars) {
               size *= dimensions->GetSize(dim);
             }

             auto upstream_buffer_req = reuse_analysis->GetSpatialIngressTraffic(tensor, iter_status);
             auto prev_upstream_buffer_req = results->GetBufferSizeReq(BufferType::Upstream,tensor->GetDataClass());

             results->UpdateBufferSizeReq(BufferType::Upstream,prev_upstream_buffer_req + upstream_buffer_req * buffer_size_mult, dataclass);

             results->UpdateBufferAccessCount(BufferType::Upstream, BufferAccessType::Write, size, dataclass);

             auto downstream_buffer_req = reuse_analysis->GetMappedVolume(tensor);
             auto prev_downstream_buffer_req = results->GetBufferSizeReq(BufferType::Downstream, tensor->GetDataClass());
             results->UpdateBufferSizeReq(BufferType::Downstream, prev_downstream_buffer_req + buffer_size_mult * downstream_buffer_req, dataclass);

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

             auto upstream_buffer_req = reuse_analysis->GetSpatialEgressTraffic(tensor, iter_status);
             results->UpdateBufferSizeReq(BufferType::Upstream, buffer_size_mult * upstream_buffer_req, dataclass);
             if(cluster_idx == 0 && buffer_size_mult * upstream_buffer_req > configs_->l2_size_) {
               error_handler_->PrintErrorMsg(TL::ErrorCode::NotEnoughL2Buffer,std::to_string(buffer_size_mult * upstream_buffer_req), this->GetName());
               error_handler_->TerminateProgram();
             }

             results->UpdateBufferAccessCount(BufferType::Upstream, BufferAccessType::Read, size, dataclass);

             auto downstream_buffer_req = reuse_analysis->GetMappedVolume(tensor);
             results->UpdateBufferSizeReq(BufferType::Downstream, buffer_size_mult * downstream_buffer_req, dataclass);
             if(cluster_idx == num_cluster_lvs -1 && buffer_size_mult * downstream_buffer_req > configs_->l1_size_) {
               error_handler_->PrintErrorMsg(TL::ErrorCode::NotEnoughL1Buffer,std::to_string(buffer_size_mult * downstream_buffer_req), this->GetName());
               error_handler_->TerminateProgram();
             }
           }
        }

        void UpdateArithmeticIntensity(
            std::shared_ptr<CostAnalyisResults> results,
            std::shared_ptr<DFA::DimensionTable> dimensions,
            long num_macs_across_subclusters) {
          long size = 1;

          for(auto& dim : *dimensions) {
            size *= dim->GetSize();
          }

          double arithmetic_intensity = static_cast<double>(num_macs_across_subclusters) / static_cast<double>(size);

          results->SetArithmeticIntensity(arithmetic_intensity);
        }


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
