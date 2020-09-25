#ifndef MAESTRO_MARVEL_HPP_
#define MAESTRO_MARVEL_HPP_

#include <iostream>
#include <memory>
#include <vector>

using namespace std;

#include<boost/program_options.hpp>

#include "BASE_base-objects.hpp"
#include "option.hpp"

#include "DFA_tensor.hpp"

#include "AHW_noc-model.hpp"

#include "CA_cost-analysis-results.hpp"

#include "API_configuration.hpp"
#include "API_user-interface.hpp"
#include "API_user-interface-v2.hpp"

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

std::map<maestro::MetricType, long double>
        maestro_get_costs(
        int num_pes,
        int l1_buffer_size,
        int l2_buffer_size,
        int noc_bw,
        std::shared_ptr<maestro::DFA::Layer> layer);

#endif
