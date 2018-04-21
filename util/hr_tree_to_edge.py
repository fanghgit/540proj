from __future__ import print_function
import os
import sys
import argparse
import math
import numpy as np



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('train_data', help="Train Data")
    #parser.add_argument('test_data', help="Test Data")
    #parser.add_argument('output_train', help="Output for train")
    #parser.add_argument('output_test', help="Output for test")
    parser.add_argument('hr_file', help="hr file")
    parser.add_argument('hr_output', help="output")

    args = parser.parse_args()
    hr_file = args.hr_file
    hr_output = args.hr_output

    f = open(hr_file, 'r')
    g = open(hr_output, 'w')
    for line in f:
        line = line.strip('\n')
        lbls = line.split(' ')
        for i in range(0,len(lbls)-1):
            print( lbls[i] + " " + lbls[i+1], file = g )
    f.close()
    g.close()
