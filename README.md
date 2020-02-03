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

# How to convert PyTorch/Keras model to MAESTRO supported modelfile
> cd data/frontend_example

Check the messages from the help.
> python frameworks_to_modelfile_maestro.py --help

Supported models:

<ul>
  <li> PyTorch: [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html) </li>
  <li> Keras: [tensorflow.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) 
  
  *tensorflow 2.0 should be installed.* </li>
</ul>

## Run with default setting
> python frameworks_to_modelfile_maestro.py

### Change the input argument
> python frameworks_to_modelfile_maestro.py --api_name pytorch --input_size 3,224,224 --model mobilenet_v2 --outfile dnn_model.m 

--api_name: the API name, choose from "pytorch, keras"

--input_size: the input image size of the first layer

--model: the model name from torchvision.models (or tensorflow.keras.applications)
         TO use a custom model, enter custom for this argument.

--custom: Enter the custom network python file name here. 
          The file should have a function whose name is same as the file name and returns the model.
          (This option is working only for keras now)
          
--outfile: the MAESTRO model output file name

### A sample output model file for Pytorch Example:
Api name: Pytorch 

Input_Size: 3,224,224 

Model: Mobilenet_v2

### Command used for setting the above arguments:
>python frameworks_to_modelfile_maestro.py --api_name pytorch --input_size 3,224,224 --model mobilenet_v2 --outfile dnn_model.m 
```
Network torchvision.models.mobilenet {
Layer Conv2d-1 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 32, C: 3, R: 3, S: 3, Y: 224, X: 224 }
}
Layer Conv2d-2 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 32, R: 3, S: 3, Y: 112, X: 112 }
}
Layer Conv2d-3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 16, C: 32, R: 1, S: 1, Y: 112, X: 112 }
}
Layer Conv2d-4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 16, R: 1, S: 1, Y: 112, X: 112 }
}
Layer Conv2d-5 {
Type: DSCONV
Stride { X: 2, Y: 2 }
Dimensions { K: 1, C: 96, R: 3, S: 3, Y: 112, X: 112 }
}
Layer Conv2d-6 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 24, C: 96, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-7 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 144, C: 24, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-8 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 144, R: 3, S: 3, Y: 56, X: 56 }
}
Layer Conv2d-9 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 24, C: 144, R: 1, S: 1, Y: 56, X: 56 }
}
}
```
### A sample output model file for Keras Example:
Api name: Keras 

Input_Size: 3,224,224 

Model: Mobilenet_v2

### Command used for setting the above arguments:
>python frameworks_to_modelfile_maestro.py --api_name keras --input_size 3,224,224 --model mobilenet_v2 --outfile dnn_model.m 

```
Network mobilenetv2_1.00_224 {
Layer block_16_expand {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 960, C: 160, R: 1, S: 1, Y: 7, X: 7 }
}
Layer block_10_expand {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 384, C: 64, R: 1, S: 1, Y: 14, X: 14 }
}
Layer block_3_project {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 144, R: 1, S: 1, Y: 28, X: 28 }
}
Layer block_9_expand {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 384, C: 64, R: 1, S: 1, Y: 14, X: 14 }
}
Layer block_15_project {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 160, C: 960, R: 1, S: 1, Y: 7, X: 7 }
}
Layer block_3_expand {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 144, C: 24, R: 1, S: 1, Y: 56, X: 56 }
}
Layer block_14_depthwise {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 960, R: 3, S: 3, Y: 7, X: 7 }
}
Layer block_9_depthwise {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 384, R: 3, S: 3, Y: 14, X: 14 }
}
Layer block_7_project {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 384, R: 1, S: 1, Y: 14, X: 14 }
}
Layer block_8_depthwise {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 384, R: 3, S: 3, Y: 14, X: 14 }
}
}
```

# How to convert the above generated modelfile to MAESTRO dataflow:

> cd data/frontend_example

Check the messages from the help.
> python modelfile_to_DFSL.py --help

## Run with default setting
> python modelfile_to_DFSL.py (Make sure that user has a model file generated from the previous script named dnn_model.m to use this above setting)

### Change the input argument
> python modelfile_to_DFSL.py --model_file dnn_model --dataflow os --outfile out.m 

--model_file: The model file supported by maestro as specified by the user or generated by the given above script.

--dataflow: the dataflow for each layer, choose from "os, ws, rs, dla"
          
--outfile: the MAESTRO DFSL output file

### A sample output DSFL file Example:
Model File used : dnn_model.m

```
Network mobilenetv2_1.00_224 {
Layer block_16_expand {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 960, C: 160, R: 1, S: 1, Y: 7, X: 7 }
}
Layer block_10_expand {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 384, C: 64, R: 1, S: 1, Y: 14, X: 14 }
}
Layer block_3_project {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 144, R: 1, S: 1, Y: 28, X: 28 }
}
Layer block_9_expand {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 384, C: 64, R: 1, S: 1, Y: 14, X: 14 }
}
Layer block_15_project {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 160, C: 960, R: 1, S: 1, Y: 7, X: 7 }
}
Layer block_3_expand {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 144, C: 24, R: 1, S: 1, Y: 56, X: 56 }
}
Layer block_14_depthwise {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 960, R: 3, S: 3, Y: 7, X: 7 }
}
```

Dataflow: Output Stationary

### Command used for setting the above arguments:
> python modelfile_to_DFSL.py --model_file dnn_model --dataflow os --outfile out.m 

```
Network mobilenetv2_1.00_224 {
Layer block_16_expand {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 960, C: 160, R: 1, S: 1, Y: 7, X: 7 }
Dataflow {
    TemporalMap(16,16) K;
    SpatialMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(1,1) C;
    Cluster(16, P);
    SpatialMap(1,1) K;
    TemporalMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(Sz(R),7) R;
    TemporalMap(Sz(S),7) S;
}
}
Layer block_10_expand {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 384, C: 64, R: 1, S: 1, Y: 14, X: 14 }
Dataflow {
    TemporalMap(16,16) K;
    SpatialMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(1,1) C;
    Cluster(16, P);
    SpatialMap(1,1) K;
    TemporalMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(Sz(R),7) R;
    TemporalMap(Sz(S),7) S;
}
}
Layer block_3_project {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 32, C: 144, R: 1, S: 1, Y: 28, X: 28 }
Dataflow {
    TemporalMap(16,16) K;
    SpatialMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(1,1) C;
    Cluster(16, P);
    SpatialMap(1,1) K;
    TemporalMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(Sz(R),7) R;
    TemporalMap(Sz(S),7) S;
}
```

## Run MAESTRO with the converted dataflow file
Change the contents of "run.sh"

--DFSL_file='data/frontend_example/out/out.m'

Run MAESTRO

> ./run.sh

# Contributors
Hyoukjun Kwon (hyoukjun@gatech.edu): Main developer, developed core framework and functionalities

Prasanth Chatarasi (cprasanth@gatech.edu): Improved APIs

Felix (Sheng-Chun) Kao (felix@gatech.edu): Implemented Pytorch frontend

Geonhwa Jeong (geonhwa.jeong@gatech.edu): Implemented Keras frontend
