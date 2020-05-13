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


#ifndef MAESTRO_AHW_PE_MODEL_HPP_
#define MAESTRO_AHW_PE_MODEL_HPP_

namespace maestro {
	namespace AHW {

		enum class Precision {FP8, FP16, FP32, INT8, INT16, INT32};

		class ProcessingElement {
			public:
				ProcessingElement(int l1_size, int vector_width, Precision mult_precision, Precision add_precision) :
					l1_size_(l1_size), vector_width_(vector_width), mult_precision_(mult_precision), add_precision_(add_precision) {
				}

				int GetL1Size() {
					return l1_size_;
				}

				int GetVectorWidth() {
					return vector_width_;
				}

				Precision GetMultPrecision() {
					return mult_precision_;
				}

				Precision GetAddPrecision() {
					return add_precision_;
				}

			protected:
				int l1_size_;
				int vector_width_;
				Precision mult_precision_;
				Precision add_precision_;
		};

	}; // End of namespace abstract_hw
}; // End of namespace maestro
#endif
