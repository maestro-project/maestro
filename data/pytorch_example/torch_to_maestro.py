import torch
import torchvision.models as models
from maestro_summary import summary
import re
import argparse



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=tuple, default=(3,224,224), help='input size')
    parser.add_argument('--model', type=str, default="mobilenet_v2", help='model from torchvision choces: \
                                                                          resnet18, alexnet, vgg16, squeezenet, densenet, \
                                                                            inception, googlenet, shufflenet, mobilenet,\
                                                                             wide_resnet50_2, mnasnet')

    parser.add_argument('--dataflow', type=str, default="os", help='dataflow choices: dla, os, ws, rs')
    parser.add_argument('--outfile', type=str, default="out.m", help='output file name')
    opt = parser.parse_args()
    INPUT_SIZE = opt.input_size
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