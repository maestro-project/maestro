cmake_minimum_required(VERSION 3.0)
project(maestro)

set(CMAKE_CXX_STANDARD 17)

add_executable(maestro maestro-top.cpp cost-model/src/BASE_base-objects.cpp)

find_package(Boost REQUIRED
    COMPONENTS filesystem system program_options)

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/cost-model/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/cost-model/include/base"
    "${CMAKE_CURRENT_SOURCE_DIR}/cost-model/include/tools"
    "${CMAKE_CURRENT_SOURCE_DIR}/cost-model/include/user-api"
    "${CMAKE_CURRENT_SOURCE_DIR}/cost-model/include/dataflow-analysis"
    "${CMAKE_CURRENT_SOURCE_DIR}/cost-model/include/dataflow-specification-language"
    "${CMAKE_CURRENT_SOURCE_DIR}/cost-model/include/design-space-exploration"
    "${CMAKE_CURRENT_SOURCE_DIR}/cost-model/include/cost-analysis"
    "${CMAKE_CURRENT_SOURCE_DIR}/cost-model/include/abstract-hardware-model"
    "${CMAKE_CURRENT_SOURCE_DIR}/cost-model/src"
    "${Boost_INCLUDE_DIRS}"
)

target_link_libraries(maestro ${Boost_LIBRARIES})
