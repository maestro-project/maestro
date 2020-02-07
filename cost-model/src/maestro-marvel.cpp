#include "maestro-marvel.h"


void handler(int sig) {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}



/*std::vector<long double> maestro_get_costs_old(
        int num_pes,
        int l1_buffer_size,
        int l2_buffer_size,
        int noc_bw,
        std::shared_ptr<maestro::DFA::Layer> layer) {

    signal(SIGSEGV, handler);

    maestro::Options option;
    option.np = num_pes;
    option.l1_size = l1_buffer_size;
    option.l2_size = l2_buffer_size;
    option.bw = noc_bw;
    option.hop_latency = 1;
    option.do_implicit_reduction = true;
    option.fg_sync = false;
    option.do_reduction = true;
    option.mc = true;

    maestro::InitializeBaseObjects();

    auto maestro_config = std::make_shared<maestro::Configuration>(
            option.np, 1,
            option.bw, option.l1_size, option.l2_size);

    auto maestro_api = std::make_shared<maestro::API>(
            maestro_config);

    maestro_api->CreateNetworkWithALayer(layer, false);

    *//* Construct the convolution problem *//*
    std::list<std::string> input_correlated_vars = {"C", "Y", "X"};
    std::list<std::string> weight_correlated_vars = {"K", "C", "R", "S"};
    std::list<std::string> output_correlated_vars = {"K", "Y", "X"};

    maestro_api->AddTensor("input", maestro::DFA::TensorClass::InputTensor, maestro::DataClass::Input, input_correlated_vars);
    maestro_api->AddTensor("weight", maestro::DFA::TensorClass::InputTensor, maestro::DataClass::Weight, weight_correlated_vars);
    maestro_api->AddTensor("output", maestro::DFA::TensorClass::OutputTensor, maestro::DataClass::Output, output_correlated_vars);

    //Applying different NoC for each cluster is also possible (add more nocs;
    //the following function add the noc to the next lower cluster level
    maestro_api->AddNoC(option.bw, option.hops, option.hop_latency, option.mc, false);
    maestro_api->AddNoC(option.bw, option.hops, option.hop_latency, option.mc, false);
    maestro_api->AddNoC(option.bw, option.hops, option.hop_latency, option.mc, false);
    maestro_api->AddNoC(option.bw, option.hops, option.hop_latency, option.mc, false);

    std::shared_ptr<std::list<std::shared_ptr<std::pair<std::string, std::string>>>> overlap_dim_list = std::make_shared<std::list<std::shared_ptr<std::pair<std::string, std::string>>>>();
    auto output_column_overlap = std::make_shared<std::pair<std::string, std::string>>("X", "S");
    auto output_row_overlap = std::make_shared<std::pair<std::string, std::string>>("Y", "R");
    overlap_dim_list->push_back(output_column_overlap);
    overlap_dim_list->push_back(output_row_overlap);

    maestro_api->AnalyzeClusters(overlap_dim_list,false);

    auto analysis_result = maestro_api->AnalyzeNeuralNetwork(false);

    std::vector<long double> costs;
    for(auto& layer_res : *analysis_result) {
        int cluster_lv = 0;
        for(auto& cluster_res : *layer_res) {
            if(cluster_lv == layer_res->size()-1) {
                maestro_api->GetCostsFromAnalysisResultsSingleCluster(
                        cluster_res, costs);
            }

            cluster_lv++;
        }
    }

    return costs;
}*/


std::map<maestro::MetricType, long double>
        maestro_get_costs(
        int num_pes,
        int l1_buffer_size,
        int l2_buffer_size,
        int noc_bw_value,
        std::shared_ptr<maestro::DFA::Layer> layer) {

    signal(SIGSEGV, handler);

    maestro::Options option;
    option.np = num_pes;
    option.l1_size = l1_buffer_size;
    option.l2_size = l2_buffer_size;
    option.bw = noc_bw_value;
    option.num_simd_lanes = 1;
    option.hop_latency = 1;
    option.do_implicit_reduction = true;
    option.fg_sync = false;
    option.do_reduction = true;
    option.mc = true;

    maestro::InitializeBaseObjects();

    maestro::SetPrintOutLevel(-1);

    std::shared_ptr<std::vector<bool>> noc_multcast = std::make_shared<std::vector<bool>>();
    std::shared_ptr<std::vector<int>> noc_latency = std::make_shared<std::vector<int>>();
    std::shared_ptr<std::vector<int>> noc_bw = std::make_shared<std::vector<int>>();

    noc_bw->push_back(option.bw);
    noc_bw->push_back(option.bw);
    noc_bw->push_back(option.bw);

    noc_latency->push_back(option.hop_latency * option.hops);
    noc_latency->push_back(option.hop_latency * option.hops);
    noc_latency->push_back(option.hop_latency * option.hops);

    noc_multcast->push_back(true);
    noc_multcast->push_back(true);
    noc_multcast->push_back(true);


    auto config = std::make_shared<maestro::ConfigurationV2>(
            option.dfsl_file_name,
            noc_bw,
            noc_latency,
            noc_multcast,
            option.np,
            option.num_simd_lanes,
            option.bw,
            option.l1_size,
            option.l2_size
    );


    auto api = std::make_shared<maestro::APIV2>(config, layer);

    auto analysis_result = api->AnalyzeNeuralNetwork(false, false);

    std::map<maestro::MetricType, long double> costs;
    for(auto& layer_res : *analysis_result) {
        int cluster_lv = 0;
        for(auto& cluster_res : *layer_res) {
            if(cluster_lv == layer_res->size()-1) {
                api->GetCostsFromAnalysisResultsSingleCluster(
                        cluster_res, costs);
            }

            cluster_lv++;
        }
    }

    return costs;
}
