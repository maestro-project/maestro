# maestro-dev
A dev version of MAESTRO. We do not prorvide technical support for this version.

# Package Dependences
C++ compiler (g++)

SCONS build system (scons)

Boost libarary (libboost-all-dev)

# How to compile the code
> scons

# How to run the program
> ./run.sh

# How to change the parameters
Change the contents of "run.sh" For parameters other than listed below, please ignore it; active development is going on them so correct functionailty is not guaranteed.

--verbose=true/false : print out detailed information or not

--DFSL_file='data/DFSL_description/Resnet34_IS.m' : Specify the target dataflow and layer description file

--noc_bw=64 : NoC bandwidth

--noc_hop_latency=1 : NoC latency per hops

--noc_mc_support=true : NoC multicast support (In current dev version it's always on)

--num_pes=256 : Number of PEs

--num_pe_alus=1 : PE ALU vector width

--l1_size=32 : l1 buffer size (inactive in the current dev version)

--l2_size=512 : l2 buffer size (inactive in the current dev version)

# How to change the DNN model and dataflow
Create a DFSL file under "data/DFSL_description" and point the file using --DFSL_file parameter in "run.sh"

For syntax of the DFSL file, please refer to other DFSL files in data/DFSL_description.
