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


#ifndef MAESTRO_DSE_ENGINE_HPP_
#define MAESTRO_DSE_ENGINE_HPP_

#include <iostream>
#include <string>
#include <memory>
#include <limits>

#include "DSE_config.hpp"
#include "DSE_csv_writer.hpp"
#include "DSE_design_point.hpp"
#include "DSE_hardware_modules.hpp"
#include "analysis-structure.hpp"
#include "mapping-analysis.hpp"
#include "cost-analysis.hpp"


//#define DO_DLA
//#define DO_RS


namespace maestro {

  namespace DSE {

    class Engine {
      public:
        Engine(std::string optimization_target, int max_num_pes, int max_bw, int min_num_pes, int min_bw,
            double area_constraint, double power_constraint, int pe_tick, int bw_tick, bool verbose ) :
              max_num_pes_(max_num_pes), max_bw_(max_bw), min_num_pes_(min_num_pes), min_bw_(min_bw),
							area_constraint_(area_constraint),
              power_constraint_(power_constraint), optimal_point_(nullptr),
							pe_tick_(pe_tick), bw_tick_(bw_tick),
              verbose_(verbose) {

          if(optimization_target == "energy") {
            optimization_target_ = DSE::OptimizationTarget::Energy;
            if(verbose) {
              std::cout << "Optimization target: energy" <<  std::endl;
            }
          }
          else if(optimization_target == "performance/energy") {
            optimization_target_ = DSE::OptimizationTarget::PerformancePerWatt;
            if(verbose) {
              std::cout << "Optimization target: performance/energy" <<  std::endl;
            }
          }
          else {
            optimization_target_ = DSE::OptimizationTarget::Runtime;
            if(verbose) {
              std::cout << "Optimization target: runtime" <<  std::endl;
            }
          }
        }

        void SetMapAnalysis(std::shared_ptr<maestro::MappingAnalysis> map_analysis) {
        	map_analysis_ = map_analysis;
        }

        void SetNocModel(std::shared_ptr<maestro::NetworkOnChipModel> noc_model) {
        	noc_model_ = noc_model;
        }
/*
        void SetPerfAnalysis(std::shared_ptr<maestro::PerformanceAnalysis> perf_analysis) {
        	perf_analysis_ = perf_analysis;
        }
*/
        void SetTensors(std::list<std::string> all_tensors) {
        	all_tensors_ = all_tensors;
        }

        void SetInputTensors(std::list<std::string> input_tensors) {
        	input_tensors_ = input_tensors;
        }

        void SetOutputTensors(std::list<std::string> output_tensors) {
        	output_tensors_ = output_tensors;
        }

        void PerformDSE(bool file_output = false, std::string file_name = "", bool do_dla = false, bool do_prs = false) {
          std::shared_ptr<maestro::DSE::DesignPoint> optimal_dp = std::make_shared<maestro::DSE::DesignPoint>(optimization_target_, std::numeric_limits<long>::max() , std::numeric_limits<long>::max(),
          																																																		0, std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 4, 4, 1, 4, 4);
          std::shared_ptr<maestro::DSE::DesignPoint> test_dp;
          std::shared_ptr<maestro::DSE::Accelerator> accelerator = std::make_shared<maestro::DSE::Accelerator>(16, 1, 4, 4, 16);

          int progress = -1;

          long num_partialsums = map_analysis_->GetTotalIterations();

          std::shared_ptr<maestro::DSE::CSVWriter> csv_writer;

          if(file_output) {
          	csv_writer = std::make_shared<maestro::DSE::CSVWriter>(file_name, num_partialsums);
          	if(!csv_writer->Valid()) {
          		std::cout << "[MAESTRO-DSE] Failed to open the output file" << std::endl;
          		exit(-1);
          	}
          }

          long long num_iters = 0;
          long long area_violation = 0;
          long long power_violation = 0;
          long long both_violation = 0;

          for(int num_pes = min_num_pes_; num_pes < max_num_pes_; num_pes += pe_tick_) {
#if defined(DO_DLA)
  for(int c_mapsz = 1; c_mapsz < map_analysis_->GetLoopBound("C"); c_mapsz++) {
  	for(int k_mapsz = 1; k_mapsz < map_analysis_->GetLoopBound("K"); k_mapsz++) {

  		map_analysis_->SetMapSize("C", c_mapsz, c_mapsz, PragmaClass::TEMPORAL_MAP);
  		map_analysis_->SetMapSize("K", k_mapsz, k_mapsz, PragmaClass::SPATIAL_MAP);

  		map_analysis_->FullReset();
  		map_analysis_->PreProcess(num_pes);
#elif defined(DO_RS)
  	  for(int c_mapsz = 1; c_mapsz < map_analysis_->GetLoopBound("C"); c_mapsz++) {
  	  	for(int k_mapsz = 1; k_mapsz < map_analysis_->GetLoopBound("K"); k_mapsz++) {

  	  		map_analysis_->SetMapSize("C", c_mapsz, c_mapsz, DirectiveClass::TEMPORAL_MAP);
  	  		map_analysis_->SetMapSize("K", k_mapsz, k_mapsz, DirectiveClass::TEMPORAL_MAP);

  	  		map_analysis_->FullReset();
  	  		map_analysis_->PreProcess(num_pes);

#endif
            int current_progress = static_cast<int>(static_cast<double>(num_pes)/static_cast<double>(max_num_pes_) * 100);
            if(current_progress > progress) {
              progress = current_progress;
              std::cout << "Processed " << progress << "% of the target design space." << " Current number of PEs:" << num_pes << std::endl;
            }

      			noc_model_->SetBandwidth(8);

      			buff_analysis_ = std::make_shared<maestro::BufferAnalysis>(map_analysis_, noc_model_, num_pes);
      			perf_analysis_ = std::make_shared<maestro::PerformanceAnalysis> (map_analysis_, buff_analysis_, noc_model_);

      			int l2_sram_sz = buff_analysis_->GetL2BufferRequiredSize(all_tensors_);
      			int l1_sram_sz = buff_analysis_->GetL1BufferRequiredSize(all_tensors_);

            accelerator->ReconstructAccelerator(num_pes, 1, 8, l1_sram_sz, l2_sram_sz);

            double estimated_area = accelerator->GetArea();
            double estimated_power = accelerator->GetPower();

            // First level skip test
//            if(estimated_area < area_constraint_ && estimated_power < power_constraint_) {

            	map_analysis_->Reset();
        			map_analysis_->PreProcess(num_pes);
/*
        			std::cout << "num_pes = " << num_pes  << ", spFolding: " << map_analysis_->GetNumSpatialFoldings()
        			          << ", tpIteration: " << map_analysis_->GetNumTemporalIterations()
        								<< ", edge tile: " << map_analysis_->GetNumEdgeTiles()
												<< std::endl;
 */
        			for(int bw = min_bw_; bw <= max_bw_; bw += bw_tick_) {
        				noc_model_->SetBandwidth(bw);

        				buff_analysis_ = std::make_shared<maestro::BufferAnalysis>(map_analysis_, noc_model_, num_pes);
        				perf_analysis_ = std::make_shared<maestro::PerformanceAnalysis> (map_analysis_, buff_analysis_, noc_model_);

        				long runtime = perf_analysis_->GetRunTime(input_tensors_, output_tensors_, num_pes, 1, true);

        				double energy = GetEnergy();

        				int l2_sram_sz = buff_analysis_->GetL2BufferRequiredSize(all_tensors_);
        				int l1_sram_sz = buff_analysis_->GetL1BufferRequiredSize(all_tensors_);

        				accelerator->ReconstructAccelerator( num_pes, 1, bw, l1_sram_sz, l2_sram_sz);

        				long double performance_per_energy = 1/(static_cast<long double>(runtime) * static_cast<long double> (energy));

        				double estimated_area = accelerator->GetArea();
        				double estimated_power = accelerator->GetPower();

        				test_dp = std::make_shared<maestro::DSE::DesignPoint> (optimization_target_, runtime, energy, performance_per_energy, estimated_area, estimated_power, num_pes, bw, 1, l2_sram_sz, l1_sram_sz);

        				for(auto in_data_class : input_tensors_) {
        				  double mult_casting_factor = this->GetMulticastingFactor(in_data_class);
                  test_dp->PutMulticastingFactor(in_data_class, mult_casting_factor);
        				}

        				// Second level skip test
//        				if(estimated_area < area_constraint_ && estimated_power < power_constraint_) {

        					for(int vector_width = 1 ; vector_width <= 1; vector_width *= 2) {
        						buff_analysis_ = std::make_shared<maestro::BufferAnalysis>(map_analysis_, noc_model_, num_pes);
        						perf_analysis_ = std::make_shared<maestro::PerformanceAnalysis> (map_analysis_, buff_analysis_, noc_model_);

        						long runtime = perf_analysis_->GetRunTime(input_tensors_, output_tensors_, num_pes, vector_width, true);

        						double energy = GetEnergy();

        						int l2_sram_sz = buff_analysis_->GetL2BufferRequiredSize(all_tensors_);
        						int l1_sram_sz = buff_analysis_->GetL1BufferRequiredSize(all_tensors_);

        						accelerator->ReconstructAccelerator(num_pes, vector_width, bw, l1_sram_sz, l2_sram_sz);

        						long double performance_per_energy = 1/(static_cast<long double>(runtime) * static_cast<long double> (energy));

        						double estimated_area = accelerator->GetArea();
        						double estimated_power = accelerator->GetPower();

        						test_dp = std::make_shared<maestro::DSE::DesignPoint> (optimization_target_, runtime, energy, performance_per_energy, estimated_area, estimated_power, num_pes, bw, vector_width, l2_sram_sz, l1_sram_sz);

                    for(auto in_data_class : input_tensors_) {
                      double mult_casting_factor = this->GetMulticastingFactor(in_data_class);
                      test_dp->PutMulticastingFactor(in_data_class, mult_casting_factor);
                    }

        						num_iters++;

        						if(estimated_area < area_constraint_ && estimated_power < power_constraint_) {
        							if(file_output) {
        								csv_writer->WriteDesignPoint(test_dp);
        							}
        							if(test_dp < optimal_dp) {
        								optimal_dp = test_dp;
        							}
        						} // End of if(estimated_area < area_constraint &&...)
        					  else if(estimated_area > area_constraint_ && estimated_power > power_constraint_)  {
        							both_violation ++;
        						}
        						else if(estimated_area > area_constraint_) {
        							area_violation++;
        						}
        						else {
        							power_violation++;
        						}

//        						std::cout << "violations" << both_violation << ", " << area_violation << ", " << power_violation << std::endl;
        					} // End of for(vector_width)
/*        				}
        				else {
        					break;
        				}
        				*/
        			} // End of for(bw)
/*            } // End of if(estimated area< area_constraint ..
            else {
            	break;
            }
            */
#if defined(DO_DLA)
  	  	}
  	  }
#elif defined(DO_RS)
  	}
  }
#endif
          } // End of for(num_pes)



/**
          std::cout << "<Design optimization Results>" << std::endl;

          std::cout << "Number of PEs: " << optimal_num_pes << std::endl;
          std::cout << "Vector width of each PE: " << optimal_vector_width << std::endl;
          std::cout << "NoC Bandwidth: " << optimal_bw << " Bytes per cycle" <<  std::endl;
          std::cout << "L2 size: " << optimal_l2_sram_sz << " Bytes" << std::endl;
          std::cout << "L1 size: " << optimal_l1_sram_sz << " Bytes" << std::endl;

          std::cout << "Estimated runtime: " << optimal_runtime << " Cycles" << std::endl;
          std::cout << "Estimated energy: " << optimal_energy/1.73 << " times of MAC energy" << std::endl;

          std::cout << "Estimated area: " << optimal_area << " um^2" << std::endl;
          std::cout << "Estimated power: " << optimal_power << "mW" << std::endl;
*/

          std::cout << "The number of iterated design points" << num_iters << std::endl;
          std::cout << "Area violation" << area_violation << std::endl;
          std::cout << "Power violation" << power_violation << std::endl;
          std::cout << "Both violation" << both_violation << std::endl;


        }

        std::shared_ptr<DSE::DesignPoint> GetOptimalPoint () {
          return optimal_point_;
        }
      protected:

        std::shared_ptr<maestro::MappingAnalysis> map_analysis_;
        std::shared_ptr<maestro::NetworkOnChipModel> noc_model_;
        std::shared_ptr<maestro::BufferAnalysis> buff_analysis_;
        std::shared_ptr<maestro::PerformanceAnalysis> perf_analysis_;
        std::list<std::string> all_tensors_;
        std::list<std::string> input_tensors_;
        std::list<std::string> output_tensors_;


        std::shared_ptr<DSE::DesignPoint> optimal_point_ = nullptr;
        DSE::OptimizationTarget optimization_target_;
        int max_num_pes_ = 256;
        int max_bw_ = 256;
        int min_num_pes_ = 8;
        int min_bw_ = 8;
        double area_constraint_;
        double power_constraint_;
        int pe_tick_;
        int bw_tick_;
        bool verbose_ = false;

      private:
        double GetEnergy () {
          double energy_consumption_ = 0.0;

          double l1_energy = 0.0;

          l1_energy += buff_analysis_->GetL1BufferRead("weight");
          l1_energy += buff_analysis_->GetL1BufferRead("input");
          l1_energy += buff_analysis_->GetL1BufferRead("output");

          l1_energy += buff_analysis_->GetL1BufferWrite("weight", true, true);
          l1_energy += buff_analysis_->GetL1BufferWrite("input", true, true);
          l1_energy += buff_analysis_->GetL1BufferWrite("output", true, true);

          l1_energy *= 2.91;

          double l2_energy = 0.0;

          l2_energy += buff_analysis_->GetL2BufferRead("weight");
          l2_energy += buff_analysis_->GetL2BufferRead("input");
          l2_energy += buff_analysis_->GetL2BufferRead("output");

          l2_energy += buff_analysis_->GetL2BufferWrite("weight", true, true);
          l2_energy += buff_analysis_->GetL2BufferWrite("input", true, true);
          l2_energy += buff_analysis_->GetL2BufferWrite("output", true, true);

          l2_energy *= 32.2;

          return l1_energy + l2_energy;
        }

      double GetMulticastingFactor (std::string target_dataclass) {
        double num_actual_writes = buff_analysis_->GetL1BufferWrite(target_dataclass);
        double num_actual_sends = buff_analysis_->GetL2BufferRead(target_dataclass);

        double multicasting_factor = num_actual_writes / num_actual_sends;

        return multicasting_factor;
      }
    };

  }; // End of namespace DSE
}; // End of namespace maestro
#endif
