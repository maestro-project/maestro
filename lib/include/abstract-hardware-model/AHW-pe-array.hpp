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


#ifndef MAESTRO_AHW_PE_ARRAY_HPP_
#define MAESTRO_AHW_PE_ARRAY_HPP_

#include <memory>

#include "../abstract-hardware-model/AHW-pe-model.hpp"

namespace maestro {

	namespace abstract_hw {
		class ProcessingElementArray {
			public:
				ProcessingElementArray(int num_pes, std::shared_ptr<ProcessingElement> pe_model) :
					num_pes_(num_pes), pe_model_(pe_model) {

				}

				int GetNumPEs () {
					return num_pes_;
				}

				std::shared_ptr<ProcessingElement> GetPEModel() {
					return pe_model_;
				}

			protected:
				int num_pes_;
				std::shared_ptr<ProcessingElement> pe_model_;

		};

	}; // End of namespace abstract_hw
}; // End of namespace maestro
#endif
