from __future__ import print_function
import os
import sys
import argparse
import math



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('train_data', help="Train Data")
    #parser.add_argument('test_data', help="Test Data")
    #parser.add_argument('output_train', help="Output for train")
    #parser.add_argument('output_test', help="Output for test")
    parser.add_argument('data', help="Data")
    parser.add_argument('data_output', help="Data output")
    parser.add_argument('label_hierarchy_input', help="label input")
    parser.add_argument('label_hierarchy_output', help="label output")
    parser.add_argument('-l', '--label_start_index', action='store', dest='l', type=int,
                       default=0, help="label start index")
    parser.add_argument('-f', '--feature_start_index', action='store', dest='f', type=int,
                       default=0, help="feature start index")
    parser.add_argument('-r', '--remap_flag', action='store', dest='r', type=int,
                       default=0, help="flag for remapping")

    args = parser.parse_args()

    data = args.data
    data_output = args.data_output
    #train_data = args.train_data
    #test_data = args.test_data
    #output_train = args.output_train
    #output_test = args.output_test
    label_hierarchy_input = args.label_hierarchy_input
    label_hierarchy_output = args.label_hierarchy_output

    label_dict = {}
    lbl_idx = 0

    f = open(data, 'r')
    g = open(data_output, 'w')
    line_number = 1
    for line in f:
        line = line.strip('\n')
        tmp = line.split(' ')
        #if ',' in line:
        #    sys.exit("input format wrong (exist ,) at line " + str(line_number))
        all_labels = tmp[0].split(',')
        new_labels = []
        for lbl in all_labels:
        #lbl = tmp[0]
            if lbl in label_dict:
                new_labels.append(label_dict[lbl] )
            else:
                lbl_idx += 1
                label_dict[lbl] = str(lbl_idx)
                new_labels.append( str(lbl_idx) )
        tmp[0] = ','.join(new_labels)
        newline = ' '.join(tmp)
        print(newline, file = g)
        line_number += 1
    f.close()
    g.close()

    f = open(label_hierarchy_input, 'r')
    g = open(label_hierarchy_output, 'w')
    line_number = 1
    for line in f:
        line = line.strip('\n')
        tmp = line.split(' ')
        if len(tmp) > 2:
            sys.exit("more than 2 " + str(line_number))
        lbl1 = tmp[0]
        lbl2 = tmp[1]
        if (lbl1 not in label_dict) or (lbl2 not in label_dict):
            continue
        else:
            tmp[0] = label_dict[lbl1]
            tmp[1] = label_dict[lbl2]
        newline = ' '.join(tmp)
        print(newline, file = g)

    f.close()
    g.close()
