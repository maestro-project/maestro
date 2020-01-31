import re
import argparse
from argparse import RawTextHelpFormatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model_file', type=str, default="dnn_model", help="<name of your model file with layer specs>")
    parser.add_argument('--dataflow', type=str, default="os", help='dataflow choices: dla, os, ws, rs')
    parser.add_argument('--outfile', type=str, default="out.m", help='output file name')
    opt = parser.parse_args()
    print('Begin processing')
    dsconv = 0
    with open(opt.dataflow + ".m" ,"r") as fd:
        with open("util/dpt.m" , "r") as fdpt:
            with open("out/" + opt.outfile, "w") as fo:
                with open("out/" + opt.model_file + ".m", "r") as fm:
                    for line in fm:
                        if(re.search("DSCONV",line)):
                            dsconv = 1
                        write_pos = re.match("Dimensions" ,line)
                        if(write_pos):
                            fo.write(line)
                            if(dsconv):
                                fdpt.seek(0)
                                fo.write(fdpt.read())
                            else:
                                fd.seek(0)
                                fo.write(fd.read())
                            dsconv=0
                        else:
                            fo.write(line)

print("DFSL file created")

                        
                    
