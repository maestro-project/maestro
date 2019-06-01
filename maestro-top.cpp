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


#include <iostream>
#include <memory>
#include <vector>
#include <list>

#include <boost/program_options.hpp>

#include "BASE_constants.hpp"
#include "BASE_base-objects.hpp"
#include "option.hpp"

#include "DFA_tensor.hpp"

#include "AHW-noc-model.hpp"

#include "CA_cost-analysis-results.hpp"

#include "API_configuration.hpp"
#include "API_user-interface.hpp"

#include "DSE_cost-database.hpp"
#include "DSE_design_point.hpp"
#include "DSE_hardware_modules.hpp"
#include "DSE_csv_writer.hpp"

int main(int argc, char** argv)
{

  maestro::Options option;
  bool success = option.parse(argc, argv);

  if(!success) {
    std::cout << "[MAESTRO] Failed to parse program options" << std::endl;
  }

  maestro::InitializeBaseObjects();

  int num_pes = option.np;

  if(option.do_dse) {
    for(int num_pes = option.np; num_pes <= option.max_num_pes; num_pes *= option.pe_tick) {
      std::cout << "Iterating number of PEs. Current: " << num_pes << std::endl;
      auto maestro_config = std::make_shared<maestro::Configuration>(num_pes, option.num_alus_per_pe, option.bw, option.l1_size, option.l2_size);
      auto maestro_api = std::make_shared<maestro::API>(maestro_config);
      int noc_bw  = num_pes;

      maestro_api->ParseDFSL(option.dfsl_file_name);

      /* Construct the convolution problem */
      std::list<std::string> input_coupled_vars = {"C", "Y", "X"};
      std::list<std::string> weight_coupled_vars = {"K", "C", "R", "S"};
      std::list<std::string> output_coupled_vars = {"K", "Y", "X"};

      maestro_api->AddTensor("input", maestro::DFA::TensorClass::InputTensor, maestro::DataClass::Input, input_coupled_vars);
      maestro_api->AddTensor("weight", maestro::DFA::TensorClass::InputTensor, maestro::DataClass::Weight, weight_coupled_vars);
      maestro_api->AddTensor("output", maestro::DFA::TensorClass::OutputTensor, maestro::DataClass::Output, output_coupled_vars);

      //Applying different NoC for each cluster is also possible (add more nocs;
      //the following function add the noc to the next lower cluster level
      maestro_api->AddNoC(noc_bw, option.hops, option.hop_latency, option.mc);
      maestro_api->AddNoC(noc_bw, option.hops, option.hop_latency, option.mc);
      maestro_api->AddNoC(noc_bw, option.hops, option.hop_latency, option.mc);
      maestro_api->AddNoC(noc_bw, option.hops, option.hop_latency, option.mc);

      std::shared_ptr<std::list<std::shared_ptr<std::pair<std::string, std::string>>>> overlap_dim_list = std::make_shared<std::list<std::shared_ptr<std::pair<std::string, std::string>>>>();
      auto output_column_overlap = std::make_shared<std::pair<std::string, std::string>>("X", "S");
      auto output_row_overlap = std::make_shared<std::pair<std::string, std::string>>("Y", "R");
      overlap_dim_list->push_back(output_column_overlap);
      overlap_dim_list->push_back(output_row_overlap);

      maestro_api->AnalyzeClusters(overlap_dim_list, false);

      auto analysis_result = maestro_api->AnalyzeNeuralNetwork();
      long total_runtime = 0;

      long l2_rd_input_count = 0;
      long l2_rd_weight_count = 0;
      long l2_wr_input_count = 0;
      long l2_wr_weight_count = 0;
      long l2_wr_output_count = 0;

      long l2_to_l1_rd_input_count = 0;
      long l2_to_l1_rd_weight_count = 0;
      long l1_to_l0_rd_input_count = 0;
      long l1_to_l0_rd_weight_count = 0;
      long l1_to_l0_rd_output_count = 0;

      long pure_l2_rd_input_count = 0;
      long pure_l2_rd_weight_count = 0;
      long pure_l2_rd_output_count = 0;
      long pure_l2_wr_input_count = 0;
      long pure_l2_wr_weight_count = 0;
      long pure_l2_wr_output_count = 0;

      long pure_l1_rd_input_count = 0;
      long pure_l1_rd_weight_count = 0;
      long pure_l1_rd_output_count = 0;
      long pure_l1_wr_input_count = 0;
      long pure_l1_wr_weight_count = 0;
      long pure_l1_wr_output_count = 0;


      long l1_rd_input_count = 0;
      long l1_rd_weight_count = 0;
      long l1_rd_output_count = 0;

      long l1_wr_input_count = 0;
      long l1_wr_weight_count = 0;
      long l1_wr_output_count = 0;

      std::string dfsl_file_name = option.dfsl_file_name;
      dfsl_file_name = dfsl_file_name.substr(dfsl_file_name.find("/") +1);
      dfsl_file_name = dfsl_file_name.substr(dfsl_file_name.find("/") +1);

      int pos_dot = dfsl_file_name.find(".");

      dfsl_file_name = dfsl_file_name.substr(0, pos_dot);
      dfsl_file_name = dfsl_file_name  + ".csv";

      auto csv_writer = std::make_shared<maestro::DSE::CSVWriter>(maestro_config, dfsl_file_name);

      int num_top_clusters = 0;
      int num_bottom_clusters = 0;

      int layer_id = 1;
      for(auto& layer_res : *analysis_result) {
        int cluster_lv = 0;
        long layer_runtime = 0;
        double layer_energy = 0;
        long double layer_perf_per_energy;
        double area;
        double power;
        int vector_width;
        int l2_size = 0;
        int l1_size = 0;
        long num_psums = 0;

        long num_iters = maestro_api->GetTempIterOverInnermostCluster(layer_id);

        std::shared_ptr<maestro::CA::CostAnalyisResults> top_res;

        std::cout << "<<<< Layer " << layer_id << " >>>>" << std::endl;
        for(auto& cluster_res : *layer_res) {
          maestro_api->PrintAnalysisResultsSingleCluster(cluster_res);

          if(cluster_lv == layer_res->size()-1) {
            top_res = cluster_res;
            total_runtime += cluster_res->GetRuntime();
            l2_rd_input_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Upstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Input);
            l2_rd_weight_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Upstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Weight);
            l2_wr_input_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Upstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Input);
            l2_wr_weight_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Upstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Weight);
            l2_wr_output_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Upstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Output);

            long num_sub_clusters = cluster_res->GetNumSubClusters();
            num_top_clusters = num_sub_clusters;

            l2_to_l1_rd_input_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Input) * num_sub_clusters - l2_rd_input_count;
            l2_to_l1_rd_weight_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Weight) * num_sub_clusters - l2_rd_weight_count;

            layer_runtime = cluster_res->GetRuntime();
            layer_energy += l2_rd_input_count * maestro::l2_energy_multiplier;
            layer_energy += l2_wr_input_count * maestro::l2_energy_multiplier;
            layer_energy += l2_rd_weight_count * maestro::l2_energy_multiplier;
            layer_energy += l2_wr_weight_count * maestro::l2_energy_multiplier;
            layer_energy += l2_wr_output_count * maestro::l2_energy_multiplier;

            l2_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Upstream, maestro::DataClass::Input);
            l2_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Upstream, maestro::DataClass::Output);
            l2_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Upstream, maestro::DataClass::Weight);
            num_psums = cluster_res->GetNumComputations();
          }
          if(cluster_lv == 0) {

            l1_rd_input_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Input);
            l1_rd_weight_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Weight);
            l1_rd_output_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Output);
            l1_wr_input_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Input);
            l1_wr_weight_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Weight);
            l1_wr_output_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Output);


            long num_sub_clusters = cluster_res->GetNumSubClusters();
            long num_cur_clusters = num_pes / num_sub_clusters;
            num_bottom_clusters = num_cur_clusters;

            l1_to_l0_rd_input_count =  (cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Input) - l1_wr_input_count) * num_pes * num_iters;
            l1_to_l0_rd_weight_count = (cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Weight) - l1_wr_weight_count) * num_pes * num_iters;
            l1_to_l0_rd_output_count = (cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Output) -l1_wr_output_count) * num_pes * num_iters;

            layer_energy += l1_rd_input_count * maestro::l1_energy_multiplier;
            layer_energy += l1_wr_input_count * maestro::l1_energy_multiplier;
            layer_energy += l1_rd_weight_count * maestro::l1_energy_multiplier;
            layer_energy += l1_wr_weight_count * maestro::l1_energy_multiplier;
            layer_energy += l1_rd_output_count * maestro::l1_energy_multiplier;
            layer_energy += l1_wr_output_count * maestro::l1_energy_multiplier;

            l1_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Downstream, maestro::DataClass::Input);
            l1_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Downstream, maestro::DataClass::Output);
            l1_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Downstream, maestro::DataClass::Weight);
          }

          cluster_lv++;
        }

        layer_energy *= maestro::DSE::cost::mac_energy;

        layer_perf_per_energy = static_cast<long double>(num_psums) / static_cast<long double>(layer_runtime) / static_cast<long double>(layer_energy);

        layer_perf_per_energy *= 1000000000; //nW -> W

        vector_width = option.num_alus_per_pe;

        long input_tensor_size = maestro_api->GetTensorSize(layer_id-1, maestro::DataClass::Input);
        long weight_tensor_size = maestro_api->GetTensorSize(layer_id-1, maestro::DataClass::Weight);

        long applied_l2_size = (option.full_buffer)? l2_size + weight_tensor_size : l2_size;

        if(option.full_buffer) {
          std::cout << "L2 is estimated as full. Size: " << l2_size << " + " << weight_tensor_size << " = " << applied_l2_size << std::endl;
        }
        else {
          std::cout << "L2 is normal. Size: " << l2_size << std::endl;
        }


        std::shared_ptr<maestro::DSE::Accelerator> accelerator = std::make_shared<maestro::DSE::Accelerator>(num_pes, vector_width, noc_bw, l1_size, applied_l2_size);
        area = accelerator->GetArea();
        power = accelerator->GetPower();

        long double ops_per_joule = num_psums / layer_energy * 1000000000; //nJ -> J


        auto layer_dp = std::make_shared<maestro::DSE::DesignPoint>(
            maestro::DSE::OptimizationTarget::Runtime, layer_runtime, layer_energy,
            layer_perf_per_energy, area, power, num_pes, noc_bw, vector_width, l2_size, l1_size);

        layer_dp->PutMulticastingFactor("input", static_cast<double>(num_iters*l1_wr_input_count)/l2_rd_input_count);
        layer_dp->PutMulticastingFactor("weight", static_cast<double>(num_iters*l1_wr_weight_count)/l2_rd_weight_count);

        double pe_power = accelerator->GetPEPower();
        double l1_power = accelerator->GetL1Power();
        double l2_power = accelerator->GetL2Power();
        double noc_power = accelerator->GetNoCPower();

        csv_writer->WriteDesignPoint(maestro_config, layer_dp,maestro_api->GetNetworkName(), layer_id, num_psums, input_tensor_size, weight_tensor_size, ops_per_joule, pe_power, l1_power, l2_power, noc_power, top_res);

        layer_id++;

      }
    } // End of for (num_pes)
  } // End of if(do_dse)
  else {
    auto maestro_config = std::make_shared<maestro::Configuration>(option.np, option.num_alus_per_pe, option.bw, option.l1_size, option.l2_size);
    auto maestro_api = std::make_shared<maestro::API>(maestro_config);

    maestro_api->ParseDFSL(option.dfsl_file_name);

    /* Construct the convolution problem */
    std::list<std::string> input_coupled_vars = {"C", "Y", "X"};
    std::list<std::string> weight_coupled_vars = {"K", "C", "R", "S"};
    std::list<std::string> output_coupled_vars = {"K", "Y", "X"};

    maestro_api->AddTensor("input", maestro::DFA::TensorClass::InputTensor, maestro::DataClass::Input, input_coupled_vars);
    maestro_api->AddTensor("weight", maestro::DFA::TensorClass::InputTensor, maestro::DataClass::Weight, weight_coupled_vars);
    maestro_api->AddTensor("output", maestro::DFA::TensorClass::OutputTensor, maestro::DataClass::Output, output_coupled_vars);

    //Applying different NoC for each cluster is also possible (add more nocs;
    //the following function add the noc to the next lower cluster level
    maestro_api->AddNoC(option.bw, option.hops, option.hop_latency, option.mc);
    maestro_api->AddNoC(option.bw, option.hops, option.hop_latency, option.mc);
    maestro_api->AddNoC(option.bw, option.hops, option.hop_latency, option.mc);
    maestro_api->AddNoC(option.bw, option.hops, option.hop_latency, option.mc);

    std::shared_ptr<std::list<std::shared_ptr<std::pair<std::string, std::string>>>> overlap_dim_list = std::make_shared<std::list<std::shared_ptr<std::pair<std::string, std::string>>>>();
    auto output_column_overlap = std::make_shared<std::pair<std::string, std::string>>("X", "S");
    auto output_row_overlap = std::make_shared<std::pair<std::string, std::string>>("Y", "R");
    overlap_dim_list->push_back(output_column_overlap);
    overlap_dim_list->push_back(output_row_overlap);

    maestro_api->AnalyzeClusters(overlap_dim_list, false);

    auto analysis_result = maestro_api->AnalyzeNeuralNetwork();
    long total_runtime = 0;

    long l2_rd_input_count = 0;
    long l2_rd_weight_count = 0;
    long l2_wr_input_count = 0;
    long l2_wr_weight_count = 0;
    long l2_wr_output_count = 0;

    long l2_to_l1_rd_input_count = 0;
    long l2_to_l1_rd_weight_count = 0;
    long l1_to_l0_rd_input_count = 0;
    long l1_to_l0_rd_weight_count = 0;
    long l1_to_l0_rd_output_count = 0;

    long pure_l2_rd_input_count = 0;
    long pure_l2_rd_weight_count = 0;
    long pure_l2_rd_output_count = 0;
    long pure_l2_wr_input_count = 0;
    long pure_l2_wr_weight_count = 0;
    long pure_l2_wr_output_count = 0;

    long pure_l1_rd_input_count = 0;
    long pure_l1_rd_weight_count = 0;
    long pure_l1_rd_output_count = 0;
    long pure_l1_wr_input_count = 0;
    long pure_l1_wr_weight_count = 0;
    long pure_l1_wr_output_count = 0;


    long l1_rd_input_count = 0;
    long l1_rd_weight_count = 0;
    long l1_rd_output_count = 0;

    long l1_wr_input_count = 0;
    long l1_wr_weight_count = 0;
    long l1_wr_output_count = 0;

    std::string dfsl_file_name = option.dfsl_file_name;
    dfsl_file_name = dfsl_file_name.substr(dfsl_file_name.find("/") +1);
    dfsl_file_name = dfsl_file_name.substr(dfsl_file_name.find("/") +1);

    int pos_dot = dfsl_file_name.find(".");

    dfsl_file_name = dfsl_file_name.substr(0, pos_dot);
    dfsl_file_name = dfsl_file_name  + ".csv";

    auto csv_writer = std::make_shared<maestro::DSE::CSVWriter>(maestro_config, dfsl_file_name);

    int num_top_clusters = 0;
    int num_bottom_clusters = 0;

    int layer_id = 1;
    for(auto& layer_res : *analysis_result) {
      int cluster_lv = 0;
      long layer_runtime = 0;
      double layer_energy = 0;
      long double layer_perf_per_energy;
      double area;
      double power;
      int num_pes = option.np;
      int noc_bw;
      int vector_width;
      int l2_size = 0;
      int l1_size = 0;
      long num_psums = 0;

      long num_iters = maestro_api->GetTempIterOverInnermostCluster(layer_id);
      std::shared_ptr<maestro::CA::CostAnalyisResults> top_res;


      std::cout << "<<<< Layer " << layer_id << " >>>>" << std::endl;
      for(auto& cluster_res : *layer_res) {
        top_res = cluster_res;

        maestro_api->PrintAnalysisResultsSingleCluster(cluster_res);

        if(cluster_lv == layer_res->size()-1) {
          total_runtime += cluster_res->GetRuntime();
          l2_rd_input_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Upstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Input);
          l2_rd_weight_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Upstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Weight);
          l2_wr_input_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Upstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Input);
          l2_wr_weight_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Upstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Weight);
          l2_wr_output_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Upstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Output);

          long num_sub_clusters = cluster_res->GetNumSubClusters();
          num_top_clusters = num_sub_clusters;

          l2_to_l1_rd_input_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Input) * num_sub_clusters - l2_rd_input_count;
          l2_to_l1_rd_weight_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Weight) * num_sub_clusters - l2_rd_weight_count;

          layer_runtime = cluster_res->GetRuntime();
          layer_energy += l2_rd_input_count * maestro::l2_energy_multiplier;
          layer_energy += l2_wr_input_count * maestro::l2_energy_multiplier;
          layer_energy += l2_rd_weight_count * maestro::l2_energy_multiplier;
          layer_energy += l2_wr_weight_count * maestro::l2_energy_multiplier;
          layer_energy += l2_wr_output_count * maestro::l2_energy_multiplier;

          l2_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Upstream, maestro::DataClass::Input);
          l2_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Upstream, maestro::DataClass::Output);
          l2_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Upstream, maestro::DataClass::Weight);
          num_psums = cluster_res->GetNumComputations();
        }
        if(cluster_lv == 0) {

          l1_rd_input_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Input);
          l1_rd_weight_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Weight);
          l1_rd_output_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Output);
          l1_wr_input_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Input);
          l1_wr_weight_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Weight);
          l1_wr_output_count = cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Write, maestro::DataClass::Output);


          long num_sub_clusters = cluster_res->GetNumSubClusters();
          long num_cur_clusters = num_pes / num_sub_clusters;
          num_bottom_clusters = num_cur_clusters;

          l1_to_l0_rd_input_count =  (cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Input) - l1_wr_input_count) * num_pes * num_iters;
          l1_to_l0_rd_weight_count = (cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Weight) - l1_wr_weight_count) * num_pes * num_iters;
          l1_to_l0_rd_output_count = (cluster_res->GetBufferAccessCount(maestro::CA::BufferType::Downstream, maestro::CA::BufferAccessType::Read, maestro::DataClass::Output) -l1_wr_output_count) * num_pes * num_iters;

          layer_energy += l1_rd_input_count * maestro::l1_energy_multiplier;
          layer_energy += l1_wr_input_count * maestro::l1_energy_multiplier;
          layer_energy += l1_rd_weight_count * maestro::l1_energy_multiplier;
          layer_energy += l1_wr_weight_count * maestro::l1_energy_multiplier;
          layer_energy += l1_rd_output_count * maestro::l1_energy_multiplier;
          layer_energy += l1_wr_output_count * maestro::l1_energy_multiplier;

          l1_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Downstream, maestro::DataClass::Input);
          l1_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Downstream, maestro::DataClass::Output);
          l1_size += cluster_res->GetBufferSizeReq(maestro::CA::BufferType::Downstream, maestro::DataClass::Weight);
        }

        cluster_lv++;
      }

      layer_energy *= maestro::DSE::cost::mac_energy;

      layer_perf_per_energy = static_cast<long double>(num_psums) / static_cast<long double>(layer_runtime) / static_cast<long double>(layer_energy);

      layer_perf_per_energy *= 1000000000; //nW -> W

      num_pes = option.np;
      noc_bw = option.bw;
      vector_width = option.num_alus_per_pe;

      long input_tensor_size = maestro_api->GetTensorSize(layer_id-1, maestro::DataClass::Input);
      long weight_tensor_size = maestro_api->GetTensorSize(layer_id-1, maestro::DataClass::Weight);

      long applied_l2_size = (option.full_buffer)? l2_size + weight_tensor_size : l2_size;

      std::shared_ptr<maestro::DSE::Accelerator> accelerator = std::make_shared<maestro::DSE::Accelerator>(num_pes, vector_width, noc_bw, l1_size, applied_l2_size);
      area = accelerator->GetArea();
      power = accelerator->GetPower();

      long double ops_per_joule = num_psums / layer_energy * 1000000000; //nJ -> J


      auto layer_dp = std::make_shared<maestro::DSE::DesignPoint>(
          maestro::DSE::OptimizationTarget::Runtime, layer_runtime, layer_energy,
          layer_perf_per_energy, area, power, num_pes, noc_bw, vector_width, l2_size, l1_size);

      layer_dp->PutMulticastingFactor("input", static_cast<double>(num_iters*l1_wr_input_count)/l2_rd_input_count);
      layer_dp->PutMulticastingFactor("weight", static_cast<double>(num_iters*l1_wr_weight_count)/l2_rd_weight_count);



      double pe_power = accelerator->GetPEPower();
      double l1_power = accelerator->GetL1Power();
      double l2_power = accelerator->GetL2Power();
      double noc_power = accelerator->GetNoCPower();

      csv_writer->WriteDesignPoint(maestro_config, layer_dp,maestro_api->GetNetworkName(), layer_id, num_psums, input_tensor_size, weight_tensor_size, ops_per_joule, pe_power, l1_power, l2_power, noc_power, top_res);

      layer_id++;

    }


  } // End of else of if(do_dse)

//  std::cout << "=======Summary========" << std::endl;
//  std::cout << "Total runtime: " << total_runtime << " cycles" << std::endl;

  return 0;
}
