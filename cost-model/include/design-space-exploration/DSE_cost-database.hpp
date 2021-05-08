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


#ifndef MAESTRO_DSE_COST_DATABASE_HPP_
#define MAESTRO_DSE_COST_DATABASE_HPP_

namespace maestro {
  namespace DSE {
    namespace cost {
      /* Embedded scaling model */

      const double mac_area = 4470.9014;
      const double mac_power = 1.2223;
      const double mac_energy = 0.904; // nJ; Caution, it's from external source!

      const double sram_unit_size_64 = 64;
      const double sram_area_64 = 4505.1889;
      const double sram_power_64 = 0.2208;

      const double sram_unit_size_32768 = 32768;
      const double sram_area_32768 = 4161.536;
      const double sram_power_32768 = 4;

      const double bus_unit_area = 101.7938;
      const double bus_unit_power = 0.8;

      const double arbiter_unit_area = 395.473;
      const double arbiter_unit_power = 0.34;

    }; // End of namespace scaling_model
  }; // End of namespace DSE
}; // End of namespace maestro


//Felix
// ref:"https://www.tomshardware.com/news/bfloat16-deep-dive-arm-bf16-support-armv8-a,40305.html"
// ref:"https://cacm.acm.org/magazines/2020/7/245701-domain-specific-hardware-accelerators/fulltext"
//namespace maestro {
//    namespace DSE {
//        namespace cost {
//            /* Embedded scaling model */
//
//            const double mac_area = 283;
//            const double mac_power = 1.2223;
//            const double mac_energy = 0.00023; // nJ; Caution, it's from external source!
//
//            const double sram_unit_size_64 = 1;
//            const double sram_area_64 = 0.104;
//            const double sram_power_64 = 0.00345;
//
//            const double sram_unit_size_32768 = 1;
//            const double sram_area_32768 = 0.104;
//            const double sram_power_32768 = 0.00345;
//
//            const double bus_unit_area = 101.7938;
//            const double bus_unit_power = 0.8;
//
//            const double arbiter_unit_area = 395.473;
//            const double arbiter_unit_power = 0.34;
//
//        }; // End of namespace scaling_model
//    }; // End of namespace DSE
//}; // End of namespace maestro
////



#endif
