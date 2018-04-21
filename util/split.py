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
    parser.add_argument('data', help="Data")
    parser.add_argument('train', help="Data")
    parser.add_argument('test', help="Data")
    parser.add_argument('-r', '--ratio', action='store', dest='r', type=float,
                       default=0.5, help="ratio for train/test")

    args = parser.parse_args()
    data = args.data
    train = args.train
    test = args.test
    r = args.r

    n = 0
    f = open(data, 'r')
    for line in f:
        n += 1
    f.close()

    n_train = int(n*r)
    n_test = n - n_train

    line_number = 0
    f = open(data, 'r')
    f_train = open(train, 'w')
    f_test = open(test, 'w')
    perm = np.random.permutation(n)
    train_dict = {}
    for i in range(n_train):
        train_dict[ perm[i] ] = 1
    for line in f:
	line = line.strip('\n')
        if line_number in train_dict:
            print(line, file = f_train)
        else:
            print(line, file = f_test)
        line_number += 1

    f.close()
    f_train.close()
    f_test.close()
