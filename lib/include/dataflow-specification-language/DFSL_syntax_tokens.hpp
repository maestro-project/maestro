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


#ifndef MAESTRO_DSL_SYNTAX_TOKENS_HPP_
#define MAESTRO_DSL_SYNTAX_TOKENS_HPP_

#include <string>

namespace maestro {


	namespace DFSL {

	  /* Common tokens */
	  const std::string brace_open_ = "{";
    const std::string brace_close_ = "}";
    const std::string bracket_open_ = "[";
    const std::string bracket_close_ = "]";

    const std::string separator_ = ";";
    const std::string comments_ = "//";

    /* Constant definition */
    const std::string constant_decl_ = "Constant";

		/* Neural network and layer Description */
		const std::string network_decl_ = "Network";
		const std::string layer_decl_ = "Layer";
		const std::string layer_type_decl_ = "Type";
			const std::string layer_type_conv_ = "CONV";
      const std::string layer_type_residual_ = "RESIDUAL_IDENTITY";
			const std::string layer_type_fc_ = "FC";
			const std::string layer_type_pool_ = "POOL";
			const std::string layer_type_dsconv_ = "DSCONV";
			const std::string layer_type_trconv_ = "TRCONV";
      const std::string layer_type_ngconv_ = "NGCONV"; // Nested grouped convolution (ResNeXt)
			const std::string layer_type_lstm_ = "LSTM";
    const std::string layer_stride_decl_ = "Stride";

		const std::string layer_dim_decl_ = "Dimensions";
			const std::string layer_dim_input_batch_    = "N";
      const std::string layer_dim_group_          = "G";
			const std::string layer_dim_output_channel_ = "K";
			const std::string layer_dim_input_channel_  = "C";
			const std::string layer_dim_weight_height_  = "R";
			const std::string layer_dim_weight_width_   = "S";
			const std::string layer_dim_input_height_   = "Y";
			const std::string layer_dim_input_width_    = "X";
      const std::string layer_dim_output_height_  = "Y'";
      const std::string layer_dim_output_width_   = "X'";


		const std::string layer_expansion_factor_decl_ = "ExpansionSize";
		const std::string layer_pooling_size_decl_ = "PoolSize";
		const std::string layer_stride_size_decl_ = "Stride";

		const std::string layer_dataflow_decl_ = "Dataflow";
		const std::string dataflow_temporal_map_ = "TemporalMap";
		const std::string dataflow_spatial_map_ = "SpatialMap";
		const std::string dataflow_cluster_ = "Cluster";

		const std::string dataflow_dim_size_indicatior_ = "Sz";

		const std::string dataflow_cluster_type_logical_ = "L";
		const std::string dataflow_cluster_type_physical_ = "P";


		/* Hardware Resource Description */
		const std::string accelerator_decl_ = "Accelerator";

		// NoC Description
		const std::string noc_decl_ = "NoC";
		const std::string noc_bandwidth_decl_ = "Bandwidth";
		const std::string noc_avg_latency_decl_ = "AvgLatency";

		// PE Description
		const std::string pe_decl_ = "PE";
		const std::string num_pe_decl_ = "NumPEs";
		const std::string vector_width_decl_ = "VectorWidth";

		const std::string mult_precision_decl_ = "MultPrecision";
    const std::string add_precision_decl_ = "AddPrecision";
      const std::string bfloat_precision_type = "BFLOAT";
      const std::string float_precision_type = "FLOAT";
      const std::string fixed_precision_type = "INT";

			const std::string fp4_precision = "FP4";
			const std::string fp8_precision = "FP8";
			const std::string fp16_precision = "FP16";
			const std::string fp32_precision = "FP32";

      const std::string int4_precision = "INT4";
      const std::string int8_precision = "INT8";
      const std::string int16_precision = "INT16";
      const std::string int32_precision = "INT32";


		// Buffer Description
		const std::string buffer_decl_ = "Buffer";
	  const std::string buffer_type_global_ = "GlobalL2";
    const std::string buffer_type_local_ = "LocalL1";
//		const std::string buffer_level_decl_ = "Level:";
//		const std::string buffer_size_decl_ = "Size:";

	};

}
;
// End of name space maestro

#endif
