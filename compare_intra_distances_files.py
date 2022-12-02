#!/usr/bin/env python
"""
Created on 5/05/22
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file includes the code needed plotting scores.
"""

from collections import defaultdict
import os
from os import listdir
from os import path
from os.path import isfile, join
import argparse

import os
import json
import argparse
import json
import random
from collections import defaultdict
from itertools import chain, combinations
import distutils
import copy
import json
import math
from secrets import choice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import base64



def create_mapping(labels_file):
    '''
    This function creates the mapping function from the old classes to the new ones.
    :param labels_file: new classes.
    :return: mapping function, index to labels name for new classes, index to labels name for old classes
    '''
    # loading cleaned classes
    print("Loading cleaned Visual Genome classes: {} .".format(labels_file))
    with open(labels_file, 'r') as file:
        cleaned_labels = file.readlines()
    # remove new line symbol and leading/trailing spaces.
    cleaned_labels = [i.strip('\n').strip() for i in cleaned_labels]
    # make dictionary
    cleaned_labels = {id+1: label for id, label in enumerate(cleaned_labels)}     # [1, 1600]
    # get previously labels from the same file and make the mapping function
    map_fn = dict()
    old_labels = dict()
    for new_label_id, new_label_str in cleaned_labels.items():
        new_label_id = int(new_label_id)
        for piece in new_label_str.split(','):
            tmp = piece.split(':')
            assert len(tmp) == 2
            old_label_id = int(tmp[0])
            old_label_str = tmp[1]
            # we need to avoid overriding of same ids like: 17:stop sign,17:stopsign
            if old_label_id not in old_labels.keys():
                old_labels[old_label_id] = old_label_str
                map_fn[old_label_id] = new_label_id
            else:
                print('Warning: label already present for {}:{}. Class {} ignored. '.format(old_label_id,
                                                                                            old_labels[old_label_id],
                                                                                            old_label_str))
    assert len(old_labels) == 1600
    assert len(old_labels) == len(map_fn)
    # print(old_labels[1590], map_fn[1590], cleaned_labels[map_fn[1590]])
    return map_fn, cleaned_labels, old_labels     # all in [1, 1600]


def print_average_results(data_clean, data_noisy, untouched_cls_idx, output_file):
    # remove null values as couple
    # to_remove = []
    # for k in data_clean.keys():
    #     if data_clean[k] is None or data_noisy[k] is None:
    #         to_remove.append(k)
    # for k in to_remove:
    #     data_clean.pop(k)
    #     data_noisy.pop(k)
    # remove null values
    data_clean = {int(k): v for k, v in data_clean.items() if v is not None}
    data_noisy = {int(k): v for k, v in data_noisy.items() if v is not None}

    # weighted average
    for name, data in zip(['noisy', 'cleaned'], [data_noisy, data_clean]):
        for class_type in ['all', 'untouched', 'merged']:
            means_to_consider = []
            for k, v in data.items():
                if class_type == 'all':
                    means_to_consider.append(np.mean(v))
                elif class_type == 'untouched':
                    if k in untouched_cls_idx:
                        means_to_consider.append(np.mean(v))
                else:   # merged
                    if k not in untouched_cls_idx:
                        means_to_consider.append(np.mean(v))
            tmp_mean = np.average(means_to_consider)
            tmp_std = math.sqrt(np.average((means_to_consider-tmp_mean)**2))
            tmp_mean = round(tmp_mean, 2)
            tmp_std = round(tmp_std, 2)
            print("Average intra distances {} classes. Aggregation:{} || Mean: {} || STD: {} . ".format(name, class_type, tmp_mean, tmp_std ))
        print("--")

def parse_args():
    """
    Parse input arguments
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parsing
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--results_clean', type=str, default='./analysis/knn/features_cleaned_intra_class_distance.json', help='File with results about clean features.')
    parser.add_argument('--results_noisy', type=str, default='./analysis/knn/features_noisy_intra_class_distance.json', help='File with results about noisy features.')
    parser.add_argument('--output_folder', type=str, default='./', help='Folder where to save the output file.')
    parser.add_argument('--labels', dest='labels',
                    help='File containing the new cleaned labels. It is needed for extracting the old and new classes indexes.',
                    default="./evaluation/objects_vocab.txt",
                    type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # get labels
    map_fn, cleaned_labels, old_labels = create_mapping(args.labels)
    map_fn_reverse = defaultdict(list)
    for k, v in map_fn.items():
        map_fn_reverse[v].append(k)
    untouched_cls_idx = [k-1 for k, v in map_fn_reverse.items() if len(v) == 1]

    # get features comparison results 
    print('Loading all data.')
    with open(args.results_clean, 'r') as f:
        results_clean = json.load(f)
    with open(args.results_noisy, 'r') as f:
        results_noisy = json.load(f)

    # check if the folder exists
    if os.path.exists(args.output_folder):
        print("Start averaging")
        print_average_results(results_clean, results_noisy, untouched_cls_idx, args.output_folder)
    else:
        print("Folder not valid: ", args.output_folder)
        exit(1)
    
