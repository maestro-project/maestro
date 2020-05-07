import re
import argparse
import os.path
from argparse import RawTextHelpFormatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--mapping_file', type=str, default="Resnet50_kcp_ws.m", help="<name of your mapping file with layer specs>")
    parser.add_argument('--outfile', type=str, default="out.m", help='output file name')
    opt = parser.parse_args()
    print('Begin processing')
    base_path = '../../data/'
    check = 0
    if os.path.exists(base_path + 'mapping/' + opt.mapping_file):
        with open(base_path + 'model/' + opt.outfile, "w") as fo:
            with open(base_path + 'mapping/' + opt.mapping_file, "r") as fm:
                for line in fm:
                    if(re.search("Dataflow",line) or re.search("SpatialMap",line) or re.search("TemporalMap",line) or re.search("Cluster",line)):
                        check = 1
                        continue
                    elif check == 1:
                        check = 0
                        continue
                    else:
                        fo.write(line)
        print("Model file created")
    else:
        print("Mapping file not found, please provide one")
