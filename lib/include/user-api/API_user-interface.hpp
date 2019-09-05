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


#ifndef API_USER_INTERFACE_HPP_
#define API_USER_INTERFACE_HPP_

#include <iostream>
#include <memory>
#include <vector>

#include "option.hpp"

#include "BASE_maestro-class.hpp"
#include "BASE_base-objects.hpp"

#include "DFSL_parser.hpp"

#include "DFA_tensor.hpp"

#include "AHW-noc-model.hpp"

#include "CA_cost-analysis-engine.hpp"
#include "CA_cost-analysis-results.hpp"

#include "API_configuration.hpp"

namespace maestro {

  class API : public MAESTROClass {

    public:
      API (std::shared_ptr<Configuration> config) :
        configuration_(config), num_macs_(0) {
        message_printer_->PrintMsg(0, "[Warning] maestro::API is deprecated. Update to maestro::APIV2 is recommended.");
      }

      std::string GetNetworkName() {
        return configuration_->network_->GetName();
      }

      void ParseDFSL(std::string dfsl_file_name, bool verbose = false)
      {
        DFSL::DFSLParser dfsl_parser(dfsl_file_name);
        dfsl_parser.ParseDFSL(configuration_->network_);

        if(verbose) {
          std::cout << "Parsing finished" << std::endl;
          std::cout <<"Network name:" << configuration_->network_->GetName() << std::endl;

          for(auto& layer: *(configuration_->network_)) {
            std::cout << layer->ToString() << std::endl;
          }
        }
      }

      void CreateNetworkWithALayer(std::shared_ptr<DFA::Layer> curr_layer,
                                   bool verbose = true) {

        configuration_->network_->AddLayer(curr_layer);

        if(verbose) {
          std::cout << "Parsing finished" << std::endl;
          std::cout <<"Network name:" << configuration_->network_->GetName() << std::endl;

          for(auto& layer: *(configuration_->network_)) {
            std::cout << layer->ToString() << std::endl;
          }
        }
      }

      void GetCostsFromAnalysisResultsSingleCluster(
              std::shared_ptr<CA::CostAnalyisResults> results,
              std::vector<long double> &costs) {

        long num_computations = results->GetNumComputations();
        costs.push_back(num_computations);

        long num_abs_computations = results->GetTopNumComputations();
        costs.push_back(num_abs_computations);

        costs.push_back(results->GetRuntime(CA::EstimationType::Exact));
        costs.push_back(results->GetRuntime(CA::EstimationType::Max));
        costs.push_back(results->GetRuntime(CA::EstimationType::Min));

        long double throughput = static_cast<long double>(num_computations) / results->GetRuntime(CA::EstimationType::Exact);
        long double throughput_min = static_cast<long double>(num_computations) / results->GetRuntime(CA::EstimationType::Max);
        long double throughput_max = static_cast<long double>(num_computations) / results->GetRuntime(CA::EstimationType::Min);

        long double abs_throughput = static_cast<long double>(num_abs_computations) / results->GetRuntime(CA::EstimationType::Exact);
        costs.push_back(abs_throughput);
        long double abs_throughput_min = static_cast<long double>(num_abs_computations) / results->GetRuntime(CA::EstimationType::Max);
        costs.push_back(abs_throughput_min);
        long double abs_throughput_max = static_cast<long double>(num_abs_computations) / results->GetRuntime(CA::EstimationType::Min);
        costs.push_back(abs_throughput_max);


        int num_data_classes = static_cast<int>(DataClass::NumDataClasses);
        long num_l1_read[num_data_classes];

        long total_l1_write = 0;
        long total_l1_read = 0;

        for(auto tensor : *(configuration_->tensors_)) {
          auto dataclass = tensor->GetDataClass();
          double reuse_factor = static_cast<double>(num_computations) / results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Write, dataclass);
          costs.push_back(reuse_factor);
          total_l1_write += results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Write, dataclass);
          total_l1_read += results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Read, dataclass);
        }

        double overall_reuse_factor = static_cast<double>(total_l1_read) / static_cast<double>(total_l1_write);
        costs.push_back(overall_reuse_factor);


        long double l2_write_energy = 0;
        long double l2_read_energy = 0;
        long double l1_write_energy = 0;
        long double l1_read_energy = 0;
        long double total_energy = 0;

        long double tmp;
        for(auto tensor : *(configuration_->tensors_)) {
          auto dataclass = tensor->GetDataClass();

          tmp = results->GetBufferAccessCount(CA::BufferType::Upstream, CA::BufferAccessType::Write, dataclass) * l2_energy_multiplier;
          l2_write_energy += tmp;

          tmp = results->GetBufferAccessCount(CA::BufferType::Upstream, CA::BufferAccessType::Read, dataclass)  * l2_energy_multiplier;
          l2_read_energy += tmp;

          tmp = results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Write, dataclass) * l1_energy_multiplier;
          l1_write_energy += tmp;

          tmp = results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Read, dataclass) * l1_energy_multiplier;
          l1_read_energy += tmp;
        }

        costs.push_back(l2_write_energy);
        costs.push_back(l2_read_energy);
        costs.push_back(l1_write_energy);
        costs.push_back(l1_read_energy);
        costs.push_back(results->GetPeakBWReq());
        costs.push_back(results->GetAvgBWReq());


        total_energy = l2_write_energy + l2_read_energy + l1_write_energy + l1_read_energy + num_computations;
        costs.push_back(total_energy);
      }

      void AddTensor(std::string tensor_name,
                     DFA::TensorClass tensor_class,
                     DataClass data_class,
                     std::list<std::string>& correlated_vars) {
        std::shared_ptr<std::list<std::string>> corr_vars_to_be_added = std::make_shared<std::list<std::string>>();

        for(auto var : correlated_vars) {
          corr_vars_to_be_added->push_back(var);
        }

        std::shared_ptr<DFA::Tensor> new_tensor = std::make_shared<DFA::Tensor>(tensor_name, tensor_class, data_class, corr_vars_to_be_added);
        configuration_->tensors_->AddTensor(new_tensor);

      }

      //From top cluster (global buffer side) to lower cluster (PE side)
      void AddNoC(int bandwidth, int hops, int hop_latency, bool multicast_support, bool verbose = false) {
        auto noc = std::make_shared<maestro::abstract_hw::NetworkOnChipModel>(bandwidth, hops, hop_latency, multicast_support);
        configuration_->nocs_->push_back(noc);

        if(verbose) {
           std::cout << "Added a NoC at cluster level" << configuration_->nocs_->size() << std::endl;
        }
      }

      void AnalyzeClusters(std::shared_ptr<std::list<std::shared_ptr<std::pair<std::string, std::string>>>> overlap_dim_list, bool verbose = false) {
        for(auto layer: *(configuration_->network_)) {
          auto dataflow = layer->GetDataflow();
          auto dimensions = layer->GetDimensions();
          auto dimension_table = ConstructDimensionTable(dimensions);
          dimension_table->AddOverlapDimensions(overlap_dim_list);

          if(verbose) {
            std::cout << "Layer " << layer->GetName() << std::endl;
            std::cout << "<Dataflow>\n" << dataflow->ToString() <<std::endl;
          }

          auto cluster_analysis = std::make_shared<DFA::ClusterAnalysis>
                                    (LayerType::CONV, configuration_->num_pes_, configuration_->tensors_,
                                     dimension_table, dataflow, configuration_->nocs_);
          configuration_->cluster_analysis_->push_back(cluster_analysis);
        }

        if(verbose) {
          std::cout << "Cluster construction and analysis is done" << std::endl;
        }
      }

      void PrintAnalysisResultsSingleCluster(std::shared_ptr<CA::CostAnalyisResults> results) {
        std::cout << std::endl;
        std::cout << std::endl;

        long num_computations = results->GetNumComputations();
        long num_abs_computations = results->GetTopNumComputations();

        long double throughput = static_cast<double>(num_computations) / results->GetRuntime(CA::EstimationType::Exact);
        long double throughput_min = static_cast<double>(num_computations) / results->GetRuntime(CA::EstimationType::Max);
        long double throughput_max = static_cast<double>(num_computations) / results->GetRuntime(CA::EstimationType::Min);

        long double abs_throughput = static_cast<double>(num_abs_computations) / results->GetRuntime(CA::EstimationType::Exact);
        long double abs_throughput_min = static_cast<double>(num_abs_computations) / results->GetRuntime(CA::EstimationType::Max);
        long double abs_throughput_max = static_cast<double>(num_abs_computations) / results->GetRuntime(CA::EstimationType::Min);


        std::cout << "Num MACs: " << num_computations << std::endl;
        std::cout << "Num MACs (absolute): " << num_abs_computations << std::endl;


        std::cout << std::endl;
        std::cout << "[Performance Analysis]" << std::endl;
        std::cout << "Runtime: " << results->GetRuntime(CA::EstimationType::Exact) << " cycles" << std::endl;
        std::cout << "Runtime(Min): " << results->GetRuntime(CA::EstimationType::Min) << " cycles" << std::endl;
        std::cout << "Runtime(Max): " << results->GetRuntime(CA::EstimationType::Max) << " cycles" << std::endl;

        std::cout << "Throughput: " << throughput << " MACs/cycle" << std::endl;
        std::cout << "Throughput(Min): " << throughput_min << " MACs/cycle" << std::endl;
        std::cout << "Throughput(Max): " << throughput_max << " MACs/cycle" << std::endl;


        std::cout << "AbsThroughput: " << abs_throughput << " MACs/cycle" << std::endl;
        std::cout << "AbsThroughput(Min): " << abs_throughput_min << " MACs/cycle" << std::endl;
        std::cout << "AbsThroughput(Max): " << abs_throughput_max << " MACs/cycle" << std::endl;


        std::cout << "[Buffer Access Analysis]" << std::endl;


        int num_data_classes = static_cast<int>(DataClass::NumDataClasses);
        long num_l1_read[num_data_classes];

        long total_l1_write = 0;
        long total_l1_read = 0;

        for(auto tensor : *(configuration_->tensors_)) {
          auto dataclass = tensor->GetDataClass();

          std::cout << "Tensor " << tensor->GetTensorName() << std::endl;
          std::cout << "L2 size requirement: " << results->GetBufferSizeReq(CA::BufferType::Upstream, dataclass) << std::endl;
          std::cout << "L1 size requirement: " << results->GetBufferSizeReq(CA::BufferType::Downstream, dataclass) << std::endl;

          std::cout << "L2 buffer write: "
              << results->GetBufferAccessCount(CA::BufferType::Upstream, CA::BufferAccessType::Write, dataclass) << std::endl;
          std::cout << "L2 buffer read: "
              << results->GetBufferAccessCount(CA::BufferType::Upstream, CA::BufferAccessType::Read, dataclass) << std::endl;
          std::cout << "L1 buffer write: "
              << results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Write, dataclass) << std::endl;
          std::cout << "L1 buffer read: "
              << results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Read, dataclass) << std::endl;
          std::cout << "Data reuse factor: "
              << static_cast<double>(num_computations) / results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Write, dataclass) << std::endl;
          total_l1_write += results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Write, dataclass);
          total_l1_read += results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Read, dataclass);
        }

        std::cout << "Overall data reuse factor: " << static_cast<double>(total_l1_read) / static_cast<double>(total_l1_write);

        std::cout << std::endl;
        std::cout << "[Energy Analysis]" << std::endl;

        long double l2_write_energy = 0;
        long double l2_read_energy = 0;
        long double l1_write_energy = 0;
        long double l1_read_energy = 0;
        long double total_energy = 0;

        std::cout << "-For each data class" << std::endl;
        long double tmp;
        for(auto tensor : *(configuration_->tensors_)) {
          auto dataclass = tensor->GetDataClass();
          std::cout << "Tensor " << tensor->GetTensorName() << std::endl;

          tmp = results->GetBufferAccessCount(CA::BufferType::Upstream, CA::BufferAccessType::Write, dataclass) * l2_energy_multiplier;
          std::cout << "L2 buffer write energy: " << tmp << " X MAC energy" << std::endl;
          l2_write_energy += tmp;

          tmp = results->GetBufferAccessCount(CA::BufferType::Upstream, CA::BufferAccessType::Read, dataclass)  * l2_energy_multiplier;
          std::cout << "L2 buffer read energy: "  << tmp << " X MAC energy" << std::endl;
          l2_read_energy += tmp;

          tmp = results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Write, dataclass) * l1_energy_multiplier;
          std::cout << "L1 buffer write energy: " << tmp << " X MAC energy" << std::endl;
          l1_write_energy += tmp;

          tmp = results->GetBufferAccessCount(CA::BufferType::Downstream, CA::BufferAccessType::Read, dataclass) * l1_energy_multiplier;
          std::cout << "L1 buffer read energy: " << tmp << " X MAC energy" << std::endl;
          l1_read_energy += tmp;
        }

        std::cout << std::endl;

        std::cout << "[Summary]" << std::endl;
        std::cout << "TotalL2 buffer write energy: " << l2_write_energy << " X MAC energy" << std::endl;
        std::cout << "Total L2 buffer read energy: " << l2_read_energy << " X MAC energy" << std::endl;
        std::cout << "Total L1 buffer write energy: " << l1_write_energy << " X MAC energy" << std::endl;
        std::cout << "Total L1 buffer read energy: " << l1_read_energy << " X MAC energy" << std::endl;
        std::cout << "Total MAC energy: " << num_computations << " X MAC energy" << std::endl;

        std::cout <<"Peak bandwidth requirement: " <<  results->GetPeakBWReq() << std::endl;
        std::cout <<"Avg bandwidth requirement: " <<  results->GetAvgBWReq() << std::endl;

        std::cout << std::endl;
        total_energy = l2_write_energy + l2_read_energy + l1_write_energy + l1_read_energy + num_computations;
        std::cout << "Total energy consumption: " << total_energy << " X MAC energy" << std::endl;

        std::cout << "Runtime: " << results->GetRuntime() << " cycles" << std::endl;
        std::cout << "Throughput: " << throughput << " MACs/cycle" << std::endl;
        long double performance_per_enrgy = throughput / total_energy;
        std::cout << "Performance per MAC energy: " << performance_per_enrgy << " MACs/cycle/(MAC_energy)" << std::endl;
      }

      std::shared_ptr<std::vector<std::shared_ptr<CA::CostAnalyisResults>>> AnalyzeCostAllClusters(int layer_id, bool show_all_cluster_results = false, bool verbose = false) {
        auto target_cluster_analysis = configuration_->cluster_analysis_->at(layer_id);
        auto clusters = target_cluster_analysis->GetClusters();
        auto perf_analysis = std::make_unique<CA::CostAnalysisEngine>
                               (configuration_->tensors_, clusters, configuration_->target_accelerator_->GetVectorWidth());
        auto results = perf_analysis->AnalyzeEntireCluster(verbose);

        if(verbose) {
          for(int cluster_lv = 0; cluster_lv < target_cluster_analysis->GetClusters()->size(); cluster_lv++) {
            std::cout << "Cluster dataflow at level " << cluster_lv << "\n" << target_cluster_analysis->GetClusters()->GetCluster(cluster_lv)->GetDataflow()->ToString() << std::endl;
          }

          for(int cluster_lv = 0; cluster_lv < target_cluster_analysis->GetClusters()->size(); cluster_lv++) {
            std::cout << "Dimensions at level " << cluster_lv << std::endl;
            auto dimensions = target_cluster_analysis->GetClusters()->GetCluster(cluster_lv)->GetDimensions();
            for(auto& dim : *dimensions) {
              std::cout << dim->ToString() << std::endl;
            }
          }

        }


        if(show_all_cluster_results && verbose) {
          int cluster_lv = 0;
          for(auto& cluster_res : *results) {
            std::cout << "Cluster Level " << cluster_lv << std::endl;
            PrintAnalysisResultsSingleCluster(cluster_res);
            cluster_lv++;
          }
        }
        else if (verbose){
          auto top_cluster_res = results->at(results->size()-1);
          PrintAnalysisResultsSingleCluster(top_cluster_res);
        }

        return results;
      }

      std::shared_ptr<std::vector<std::shared_ptr<std::vector<std::shared_ptr<CA::CostAnalyisResults>>>>>
        AnalyzeNeuralNetwork(bool verbose = true) {
        auto ret = std::make_shared<std::vector<std::shared_ptr<std::vector<std::shared_ptr<CA::CostAnalyisResults>>>>>();

        int layer_id = 0;
        for(auto layer : *(configuration_->network_)) {
          auto layer_results = AnalyzeCostAllClusters(layer_id, false, verbose);

          long num_macs = this->GetNumPartialSums(layer_id);
          layer_results->at(layer_results->size()-1)->UpdateTopNumComputations(num_macs);

          ret->push_back(layer_results); // Take the top level cluster results
          layer_id++;
        }

        return ret;
      }

      int GetNumTopClusters(int layer_id) {
        auto cluster_table = configuration_->cluster_analysis_->at(layer_id-1)->GetClusters();
        return cluster_table->GetCluster(0)->GetNumClusters(false);
      }

      long GetTempIterOverInnermostCluster(int layer_id) {
        auto target_cluster_table = configuration_->cluster_analysis_->at(layer_id-1)->GetClusters();
        long ret = 1;
        for(int cluster_idx = 0; cluster_idx < target_cluster_table->size()-1; cluster_idx++) {
          ret *= target_cluster_table->GetCluster(cluster_idx)->GetNumTotalIterations();
        }
        return ret;
      }

      long GetTensorSize(int layer_num, DataClass data_class) {
        for(auto& tensor : *configuration_->tensors_) {
          if(tensor->GetDataClass() == data_class) {
            auto coupled_dims = tensor->GetCoupledVariables();
            auto layer_dim = configuration_->network_->at(layer_num)->GetDimensions();
            long ret = 1;

            for(auto dim : *layer_dim) {
              auto dim_name = dim->GetName();


              for(auto tensor_dim : *coupled_dims) {
                if(tensor_dim == dim_name) {
                  ret *= dim->GetSize();
                  break;
                }
              }
            }

            return ret;
          }
        }

        return 0;
      }


    protected:
      std::shared_ptr<Configuration> configuration_;
      long num_macs_;

    private:
      long GetNumPartialSums(int layer_id) {
        auto full_dimension = configuration_->network_->at(layer_id)->GetDimensions();
        long ret = 1;
        for(auto dim : *full_dimension) {
          ret *= dim->GetSize();
        }

        return ret;
      }

      //Output centric -> input centric
      std::shared_ptr<DFA::DimensionTable> ConstructDimensionTable(std::shared_ptr<std::vector<std::shared_ptr<DFA::LayerDimension>>> dimensions, bool verbose = false) {
        auto dimension_table = std::make_shared<DFA::DimensionTable>();


        int IX_size, IY_size, R_size, S_size, OX_size, OY_size;
        bool has_IY = false;
        bool has_IX = false;
        bool has_OY =false;
        bool has_OX =false;

        num_macs_ = 1;

        for(auto dim : *dimensions) {
          if(dim->GetName() == "Y") {
            has_IY = true;
            IY_size = dim->GetSize();
          }
          else if(dim->GetName() == "X") {
            has_IX = true;
            IX_size = dim->GetSize();
          }
          else if(dim->GetName() == "X'") {
            has_OX = true;
            OX_size = dim->GetSize();
          }
          else if(dim->GetName() == "Y'") {
            has_OY = true;
            OY_size = dim->GetSize();
          }
          else if(dim->GetName() == "R") {
            R_size = dim->GetSize();
          }
          else if(dim->GetName() == "S") {
            S_size = dim->GetSize();
          }
          num_macs_ *= dim->GetSize();
          dimension_table->AddDimension(dim);
        }

        if(has_IX && !has_OX) {
          auto ox_dim = std::make_shared<DFA::LayerDimension>("X'", IX_size - S_size +1);
          dimension_table->AddDimension(ox_dim);
        }
        if(has_IY && !has_OY) {
          auto oy_dim = std::make_shared<DFA::LayerDimension>("Y'", IY_size - R_size +1);
          dimension_table->AddDimension(oy_dim);
        }
        if(!has_IX && has_OX) {
          auto ix_dim = std::make_shared<DFA::LayerDimension>("X", OX_size + S_size -1);
          dimension_table->AddDimension(ix_dim);
        }
        if(!has_IY && has_OY) {
          auto iy_dim = std::make_shared<DFA::LayerDimension>("Y", OY_size + R_size -1);
          dimension_table->AddDimension(iy_dim);
        }

        if(verbose) {
          std::cout << "Dimensions" << std::endl;

          for(auto dim : *dimension_table) {
            std::cout << "Dim " << dim->GetName() << ", size = " << dim->GetSize() << std::endl;
          }

          std::cout << std::endl;
        }

        return dimension_table;
      }
  };
// End of class API

}
;
// End of namespace maestro
#endif
