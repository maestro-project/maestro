
env = Environment()

includes = '''
              .
              ./lib/include
              ./lib/include/base
              ./lib/include/tools
              ./lib/include/user-api
              ./lib/include/dataflow-analysis
              ./lib/include/dataflow-specification-language
              ./lib/include/design-space-exploration
              ./lib/include/cost-analysis
              ./lib/include/abstract-hardware-model
              ./lib/src
'''
env.Append(LINKFLAGS=['-lboost_program_options', '-lboost_filesystem', '-lboost_system'])
env.Append(CXXFLAGS=['-std=c++17', '-lboost_program_options',  '-lboost_filesystem', '-lboost_system'])
env.Append(LIBS=['-lboost_program_options',  '-lboost_filesystem', '-lboost_system' ])

env.Append(CPPPATH = Split(includes))
#env.Program("maestro-top.cpp")
#env.Program('maestro', ['maestro-top.cpp', 'lib/src/maestro_v3.cpp', 'lib/src/BASE_base-objects.cpp' ])
env.Program('maestro', ['maestro-top.cpp', 'lib/src/BASE_base-objects.cpp' ])
#env.Library('maestro', ['maestro-top.cpp', 'lib/src/maestro_v3.cpp', 'lib/src/BASE_base-objects.cpp' ])

