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


#ifndef MAESTRO_DFA_LAYER_HPP_
#define MAESTRO_DFA_LAYER_HPP_

#include <memory>

#include "DFA_directives.hpp"
#include "DFA_directive-table.hpp"

namespace maestro{

    enum class ConvLayerDimensionIdentifier {K, C, R, S ,Y, X};
    enum class LayerType {CONV, FC, POOL, DSCONV, TRCONV, LSTM, INVALID};

    namespace DFA {

      class LayerDimension {
        protected:
          std::string name_;
          int size_;
          int stride_;
        public:
          LayerDimension(std::string name, int size, int stride=1) :
            name_(name), size_(size), stride_(stride) {
          }

          std::string GetName(){
            return name_;
          }

          int GetSize() {
            return size_;
          }

          int GetStride() {
            return stride_;
          }

          std::string ToString() {
          	std::string ret = name_ + ": size = " + std::to_string(size_) + ", stride = " + std::to_string(stride_);
          	return ret;
          }
      }; // End of class LayerDimension

      class Layer {
        public:

          Layer(std::string name) : name_(name), misc_factor_(1), type_(LayerType::INVALID) {}

          Layer(std::string name, LayerType type, std::shared_ptr<std::vector<std::shared_ptr<LayerDimension>>> dimensions, int misc_factor) :
            name_(name), type_(type), dimensions_(dimensions), misc_factor_(misc_factor) {
          }

          virtual ~Layer() {}

          virtual bool IsValid() {
            return false;
          }

          std::string GetName() {
            return name_;
          }

          void SetDimensions(std::shared_ptr<std::vector<std::shared_ptr<LayerDimension>>> dimensions) {
          	dimensions_ = dimensions;
          }

          std::shared_ptr<std::vector<std::shared_ptr<LayerDimension>>> GetDimensions() {
            return dimensions_;
          }

          void SetDataflow(std::shared_ptr<DFA::DirectiveTable> directives)  {
          	dataflow_directives_ = directives;
          }

          std::shared_ptr<DFA::DirectiveTable> GetDataflow() {
          	return dataflow_directives_;
          }


          int GetSize(std::string id) {
            for (auto &it : *dimensions_) {
              if(it->GetName() == id) {
                return it->GetSize();
              }
            }
            return -1;
          }

          int GetStride(std::string id) {
            for (auto &it : *dimensions_) {
              if(it->GetName() == id) {
                return it->GetStride();
              }
            }
            return -1;
          }

          virtual std::string ToString() {
          	std::string ret = "BaseLayer";
          	return ret;
          }

        protected:
          LayerType type_;
          std::string name_;
          std::shared_ptr<std::vector<std::shared_ptr<LayerDimension>>> dimensions_;
          std::shared_ptr<DFA::DirectiveTable> dataflow_directives_;
          int misc_factor_;

      }; // End of class Layer

      class ConvLayer : public Layer {
        public:

      		ConvLayer (std::string name) : Layer(name) {
      		}

          ConvLayer (std::string name, std::shared_ptr<std::vector<std::shared_ptr<LayerDimension>>> dimensions, int expansion_factor) :
           Layer(name, LayerType::CONV, dimensions, expansion_factor) {
          }

          virtual ~ConvLayer() {}

          std::string GetName() {
            return name_;
          }

          std::shared_ptr<std::vector<std::shared_ptr<LayerDimension>>> GetDimensions() {
            return dimensions_;
          }

          virtual bool IsValid() {

            int has_input_batch = 0;
            int has_output_channel = 0;
            int has_input_channel = 0;
            int has_weight_height = 0;
            int has_weight_width = 0;
            int has_input_height = 0;
            int has_input_width = 0;

            for(auto &it : *dimensions_) {
              if(it->GetName() == DFSL::layer_dim_input_batch_) {
                has_input_batch++;
                continue;
              }
              if(it->GetName() == DFSL::layer_dim_output_channel_) {
                has_output_channel++;
                continue;
              }
              if(it->GetName() == DFSL::layer_dim_input_channel_) {
                has_input_channel++;
                continue;
              }
              if(it->GetName() == DFSL::layer_dim_weight_height_) {
                has_weight_height++;
                continue;
              }
              if(it->GetName() == DFSL::layer_dim_weight_width_) {
                has_weight_width++;
                continue;
              }
              if(it->GetName() == DFSL::layer_dim_input_height_) {
                has_input_height++;
                continue;
              }
              if(it->GetName() == DFSL::layer_dim_input_width_) {
                has_input_width++;
              }
            }

            if(has_input_batch == 1 && has_output_channel == 1 && has_input_channel == 1
                && has_weight_height == 1 && has_weight_width == 1 && has_input_height == 1
                && has_input_width == 1) {
              return true;
            }

            return false;
          }

          int GetSize(std::string id) {
            for (auto &it : *dimensions_) {
              if(it->GetName() == id) {
                return it->GetSize();
              }
            }
            return -1;
          }

          int GetStrideWidth() {
            for (auto &it : *dimensions_) {
              if(it->GetName() == DFSL::layer_dim_input_width_) {
                return it->GetStride();
              }
            }
            return -1;
          }

          int GetStrideHeight() {
            for (auto &it : *dimensions_) {
              if(it->GetName() == DFSL::layer_dim_input_height_) {
                return it->GetStride();
              }
            }
            return -1;
          }

          virtual std::string ToString() {
          	std::string ret = "Layer " + name_ + "{\nType: CONV\n Dimension {\n";
            for (auto &it : *dimensions_) {
            	ret += it->ToString();
            	ret += "\n";
            }
            ret += "}\n";

            ret += "Dataflow {\n";
            for (auto &it : *dataflow_directives_) {
            	ret += it->ToString();
            	ret += "\n";
            }
            ret += "}\n";

          	return ret;
          }

      }; // End of class ConvLayer

      class FCLayer : public Layer {
        protected:

        public:
          FCLayer (std::string name, LayerType type, std::shared_ptr<std::vector<std::shared_ptr<LayerDimension>>> dimensions) :
            Layer(name, type, dimensions, -1) {
          }

          virtual std::string ToString() {
          	std::string ret = "Layer " + name_ + ", Type: FC {\n";
            for (auto &it : *dimensions_) {
            	ret += it->ToString();
            	ret += "\n";
            }
            ret += "}";

          	return ret;
          }
      }; // End of class FCLayer


      class LSTMLayer : public Layer {
        protected:

        public:
          LSTMLayer (std::string name, LayerType type, std::shared_ptr<std::vector<std::shared_ptr<LayerDimension>>> dimensions) :
            Layer(name, type, dimensions, -1) {
          }

          virtual std::string ToString() {
            std::string ret = "Layer " + name_ + ", Type: LSTM {\n";
            for (auto &it : *dimensions_) {
              ret += it->ToString();
              ret += "\n";
            }
            ret += "}";

            return ret;
          }
      }; // End of class LSTM Layer


    }; // End of namespace DFA
}; // End of namespace maestro


#endif
