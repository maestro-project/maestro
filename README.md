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

# How to change the DNN model and dataflow
Create a DFSL file under "data/DFSL_description" and point the file using --DFSL_file parameter in "run.sh"

For syntax of the DFSL file, please refer to other DFSL files in data/DFSL_description.

# How to convert PyTorch/Keras model to MAESTRO dataflow
> cd data/frontend_example

Check the messages from the help.
> python framework_to_maestro.py --help

Supported models:
PyTorch: [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)
Keras: [tensorflow.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
    tensorflow 2.0 should be installed.

## Run with default setting
> python framework_to_maestro.py

## Run MAESTRO with the converted dataflow file
Change the contents of "run.sh"

--DFSL_file='data/frontend_example/out/out.m'

Run MAESTRO

> ./run.sh

### Change the input arguement
> python framework_to_maestro.py --api_name pytorch --input_size 3,224,224 --model mobilenet_v2 --dataflow os --outfile out.m 

--api_name: the API name, choose from "pytorch, keras"

--input_size: the input image size of the first layer

--model: the model name from torchvision.models (or tensorflow.keras.applications)
         TO use a custom model, enter custom for this argument.

--custom: Enter the custom network python file name here. 
          The file should have a function whose name is same as the file name and returns the model.
          (This option is working only for keras now)

--dataflow: the dataflow for each layer, choose from "os, ws, rs, dla"

--outfile: the MAESTRO dataflow output file name

# Contributors
Hyoukjun Kwon (hyoukjun@gatech.edu): Main developer, developed core framework and functionalities

Prasanth Chatarasi (cprasanth@gatech.edu): Improved APIs

Felix (Sheng-Chun) Kao (felix@gatech.edu): Implemented Pytorch frontend

Geonhwa Jeong (geonhwa.jeong@gatech.edu): Implemented Keras frontend
