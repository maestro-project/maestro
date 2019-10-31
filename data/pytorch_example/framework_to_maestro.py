#import tensorflow as tf

import torch
import torchvision.models as models

from keras_helper import get_model
from maestro_summary import summary
import re
import argparse
from argparse import RawTextHelpFormatter

# inception_v3 input should be 3, 299, 299


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--api_name', type=str, default="pytorch", help="api choices: pytorch, keras")
    parser.add_argument('--input_size', type=str, default="3,224,224", help='input size')
    parser.add_argument('--model', type=str, default="mobilenet_v2",
    help='model from torchvision choices: \n'
         'resnet18, alexnet, vgg16, squeezenet, densenet, \n'
         'inception_v3, googlenet, shufflenet, mobilenet,\n'
         'mobilenet_v2, wide_resnet50_2, mnasnet,\n'
         '-----\n'
         'model from tensorflow.keras.applications choices: \n'
         'xception, vgg16, vgg19, resnet50, resnet101, \n'
         'resnet152, resnet50_v2, resnet101_v2, resnet152_v2, \n'
         'inception_v3, inception_resnet_v2, mobilenet, mobilenet_v2,\n'
         'densenet121, densenet169, densenet201, nasnet_large, \n'
         'nasnet_mobile\n'
         '-----\n'
         'To use a custom model, enter custom for this arguement')
    parser.add_argument('--custom', type=str, default="none",
    help='Enter the custom network python file name here.\n'
         'The file should have a function with same file name\n '
         'which returns the model\n'
         '(This option is working only for keras)\n')


    parser.add_argument('--dataflow', type=str, default="os", help='dataflow choices: dla, os, ws, rs')
    parser.add_argument('--outfile', type=str, default="out.m", help='output file name')
    opt = parser.parse_args()
    INPUT_SIZE = tuple((int(d) for d in str.split(opt.input_size, ",")))

    if(opt.api_name =='keras'):
        print('Using keras with input size: ' + str(INPUT_SIZE))
        model = None
        if opt.model == 'custom':
            new_module = __import__(opt.custom)
            model = getattr(new_module, opt.custom)()
        else:
            model = get_model(opt.model, INPUT_SIZE[::-1])
        #print(model.summary())
        mae_summary = {}
        for i in range(len(model.layers)):
            cur_config = model.layers[i].get_config()
            cur_config['type'] = model.layers[i].__class__.__name__

            is_conv2d = cur_config['type'] == 'Conv2D'
            is_dwconv2d = cur_config['type'] == 'DepthwiseConv2D'
            is_dense = cur_config['type'] == 'Dense'

            if is_conv2d:
                cur_config['type'] = 'CONV'
            if is_dwconv2d:
                cur_config['type'] = 'DSCONV'

            if(is_conv2d or is_dwconv2d or is_dense):
                cur_ic = []
                cur_ic.append(None)
                if(is_conv2d):
                    cur_ic.append(model.layers[i].output_shape[3])
                elif(is_dwconv2d):
                    cur_ic.append(1)
                elif(is_dense):
                    cur_ic.append(model.layers[i].output_shape[1])

                if(is_dense):
                    cur_ic.append(model.layers[i].input_shape[1])
                    for _ in range(4):
                        cur_ic.append(1)
                else:
                    cur_ic.append(model.layers[i].input_shape[3])
                    cur_ic.append(model.layers[i].kernel_size[0])
                    cur_ic.append(model.layers[i].kernel_size[1])
                    cur_ic.append(model.layers[i].input_shape[1])
                    cur_ic.append(model.layers[i].input_shape[2])

                cur_config['dimension_ic'] = cur_ic
            mae_summary[model.layers[i].get_config()['name']] = cur_config

        with open(opt.dataflow + ".m", "r") as fd:
            with open("util/dpt.m", "r") as fdpt:
                with open("out/"+opt.outfile, "w") as fo:
                    fo.write("Network {} {{\n".format(model.name))
                    for key, val in mae_summary.items():
                        pc = re.compile("^CONV")
                        pd = re.compile("^DSCONV")
                        pf = re.compile("^Dense")

                        #print(val['type'])
                        match_pc = pc.match(val['type'])
                        match_pd = pd.match(val['type'])
                        match_pf = pf.match(val['type'])

                        if match_pc or match_pd or match_pf:
                            fo.write("Layer {} {{\n".format(key))
                            type = val["type"]
                            if match_pf:
                                fo.write("Type: CONV\n")
                            if not match_pf:
                                fo.write("Type: {}\n".format(type))
                                fo.write("Stride {{ X: {}, Y: {} }}\n".format(*val["strides"]))

                            fo.write("Dimensions {{ K: {}, C: {}, R: {}, S: {}, Y: {}, X: {} }}\n".format(
                                *val["dimension_ic"][1:]))

                            if match_pc or match_pf:
                                fd.seek(0)
                                fo.write(fd.read())
                            else:
                                fdpt.seek(0)
                                fo.write(fdpt.read())
                            fo.write("}\n")
                    fo.write("}")

    if(opt.api_name == 'pytorch'):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = getattr(models, opt.model)()
        model = model.to(device)
        mae_summary = summary(model, INPUT_SIZE)
        with open(opt.dataflow + ".m", "r") as fd:
            with open("util/dpt.m", "r") as fdpt:
                with open("out/"+opt.outfile, "w") as fo:
                    fo.write("Network {} {{\n".format(model.__module__))
                    for key, val in mae_summary.items():
                        pc = re.compile("^Conv")
                        pl = re.compile("^Linear")
                        match_pc = pc.match(key)
                        match_pl = pl.match(key)
                        if match_pc or match_pl:
                            fo.write("Layer {} {{\n".format(key))
                            type = val["type"]
                            fo.write("Type: {}\n".format(type))
                            if not match_pl:
                                fo.write("Stride {{ X: {}, Y: {} }}\n".format(*val["stride"]))
                            fo.write("Dimensions {{ K: {}, C: {}, R: {}, S: {}, Y: {}, X: {} }}\n".format(
                                *val["dimension_ic"][1:]))
                            if type == "CONV":
                                fd.seek(0)
                                fo.write(fd.read())
                            else:
                                fdpt.seek(0)
                                fo.write(fdpt.read())
                            fo.write("}\n")
                    fo.write("}")

    print("Done converting to the Maestro DFSL file")
