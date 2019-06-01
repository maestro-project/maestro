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


#ifndef MAESTRO_DFA_ANALYSIS_OUTPUT_HPP_
#define MAESTRO_DFA_ANALYSIS_OUTPUT_HPP_

namespace maestro {
	namespace DFA {

		class BufferAnalysis {
			public:
				BufferAnalysis() {
				}

				void AddSizeReq(int size_req) {
					size_req_.push_back(size_req);
				}

				void AddAccessCounts(int counts) {
					access_counts_.push_back(counts);
				}

			protected:
				std::vector<int> size_req_;
				std::vector<int> access_counts_;
		};


	}; // End of namespace DFA
}; // End of namespace maestro

#endif /* MAESTRO_V2_LIB_INCLUDE_DATAFLOW_ANALYSIS_DFA_ANALYSIS_OUTPUT_HPP_ */
