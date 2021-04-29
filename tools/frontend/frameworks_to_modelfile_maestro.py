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
         'inception_v3, googlenet, shufflenet, \n'
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
         'which returns the model. Also put this file in keras_example\n'
         '(This option is working only for keras)\n')


    parser.add_argument('--outfile', type=str, default="dnn_model.m", help='output file name')
    opt = parser.parse_args()
    INPUT_SIZE = tuple((int(d) for d in str.split(opt.input_size, ",")))

    print('Begin processing')
    print('API name: ' + str(opt.api_name))
    if(opt.model == 'custom'):
        print('Model name: ' + str(opt.custom))
    else:
        print('Model name: ' + str(opt.model))
    print('Input size: ' + str(INPUT_SIZE))
    if(opt.api_name =='keras'):
        from helpers.keras_helper import get_model
        from helpers.keras_maestro_summary import summary

        model = None
        if opt.model == 'custom':
            print(opt.custom)
            module_name = 'keras-example.' + opt.custom
            new_module = __import__(module_name, fromlist=[opt.custom])
            model = getattr(new_module, opt.custom)()
        else:
            model = get_model(opt.model, INPUT_SIZE[::-1])

        mae_summary = summary(model)

        with open("../../data/model/"+opt.outfile, "w") as fo:
            fo.write("Network {} {{\n".format(model.name))
            for key, val in mae_summary.items():
                pc = re.compile("^CONV")
                pd = re.compile("^DSCONV")
                pf = re.compile("^Dense")

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
                    fo.write("}\n")
            fo.write("}")

    if(opt.api_name == 'pytorch'):
        import torch
        import torchvision.models as models
        from helpers.torch_maestro_summary import summary
        import sys, os
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = None
        if opt.model == 'custom':
            sys.path.insert(0,os.path.dirname(opt.custom))
            fn = os.path.basename(opt.custom).split('.')[0]
            new_module = __import__(fn, fromlist=[fn])
            model = getattr(new_module, fn)()
        else:
            model = getattr(models, opt.model)()
        model = model.to(device)
        mae_summary = summary(model, INPUT_SIZE)
        with open("../../data/model/"+opt.outfile, "w") as fo:
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
                    fo.write("}\n")
            fo.write("}")

    print("Done converting to the Maestro DNN MODEL file")
