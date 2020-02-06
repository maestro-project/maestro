# How to generate the mapping (Maestro input file)
A mapping contains a DNN model and the dataflow for each layer.

This tutorial is written to provide an easy way to generate a mapping from a PyTorch/Keras model.

1. Generate a Maestro DNN Model file from a Pytorch/Keras model.
2. Generate a Maestro Mapping file with the Maestro DNN Model file and specific dataflow.
3. Run with the generated mapping.

For the syntax of the mapping file, please refer to example mapping files in data/mapping.

> cd tools/frontend

Check the messages from the help for the future reference.
> python frameworks_to_modelfile_maestro.py --help

Supported pre-trained models are provided in the following links:

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

The file will be generated in data/model:
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

The file will be generated in data/model
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

# How to convert the above generated modelfile to MAESTRO mapping with dataflow:

> cd tools/frontend

Check the messages from the help.
> python modelfile_to_mapping.py --help

## Run with default setting
> python modelfile_to_mapping.py (Make sure that user has a model file generated from the previous script named dnn_model.m to use this default setting)

### Change the input argument
> python modelfile_to_mapping.py --model_file dnn_model.m --dataflow os --outfile out.m 

--model_file: The model file supported by maestro as specified by the user or generated by the above given script.

--dataflow: the dataflow for each layer, choose from "os, ws, rs, dla"
          
--outfile: the MAESTRO DFSL output file

### Command used for setting the above arguments:
> python modelfile_to_mapping.py --model_file dnn_model.m --dataflow os --outfile out.m 

The mapping file, out.m, will be generated in data/mapping:

## Run MAESTRO with the converted mapping file
Go back to the maestro-dev directory.

> cd ../../

Change the contents of "run.sh" to use the mapping file generated.

--DFSL_file='data/mapping/out.m'

Run MAESTRO

> ./run.sh
