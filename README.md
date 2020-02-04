# maestro-dev
A dev version of MAESTRO. We do not prorvide technical support for this version.

# Package Dependences
C++ compiler (g++)

SCONS build system (scons)

Boost libarary (libboost-all-dev)

Python 2.7 or later

# How to compile the code
> scons

# How to run the program
> ./run.sh

# How to change the parameters
Change the contents of "run.sh" For parameters other than listed below, please ignore it; active development is going on them so correct functionailty is not guaranteed.

--print_res=true/false : If set true, MAESTRO prints out detailed cost information to the screen

--print_res_csv_file=true/false : If set true, MAESTRO prints out a csv file that contains various statistics

--print_log_file=true/false : If set true, MAESTRO prints out a log file that contains various information of detailed computation patterns to "log.txt"

--DFSL_file='data/DFSL_description/Resnet50_dla.m' : Specify the target dataflow and layer description file

--noc_bw=64 : NoC bandwidth

--noc_hop_latency=1 : NoC latency per hops

--noc_mc_support=true : NoC multicast support (In current dev version it's always on)

--num_pes=256 : Number of PEs

--num_pe_alus=1 : PE ALU vector width

--l1_size=32 : l1 buffer size

--l2_size=512 : l2 buffer size

# How to use generate input files for Maestro?
Read the [tutorial](https://github.com/ghjeong12/maestro-dev/blob/dev/docs/frontend_tutorial.md).

# Contributors
Hyoukjun Kwon (hyoukjun@gatech.edu): Main developer, developed core framework and functionalities

Prasanth Chatarasi (cprasanth@gatech.edu): Improved APIs

Felix (Sheng-Chun) Kao (felix@gatech.edu): Implemented Pytorch frontend

Geonhwa Jeong (geonhwa.jeong@gatech.edu): Implemented Keras frontend
