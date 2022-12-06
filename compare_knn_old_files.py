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


def filter_classes(all_data, clean_cls_idx, classes_type):
    """
    Filter the data according to their classes.  
    :param all_data: all features for each class. A dict {CLASS: list of features}. CLASS values are in [0, 878]
    :param clean_cls_idx: indexes of merged classes. List of values in [1, 878] or [1, 1600]
    """
    if classes_type != 'all':
        if classes_type == 'new':
            all_data_filtered = {int(k): v for k, v in all_data.items() if int(k)+1 in clean_cls_idx}
        else:
            all_data_filtered = {int(k): v for k, v in all_data.items() if int(k)+1 not in clean_cls_idx}
    else:
        all_data_filtered = {int(k): v for k, v in all_data.items()} # just change string to int representation
    return all_data_filtered


def print_results(data_clean, data_noisy, classes_type, freq_noisy_file=None, freq_clean_file=None):
    # load frequencies file
    if freq_noisy_file is not None and freq_clean_file is not None:
        print("Load frequencies ofr weighted average")
        with open(freq_noisy_file, 'r') as f:
            freq_noisy = json.load(f)
            freq_noisy = {int(k): v for k, v in freq_noisy.items()}
        with open(freq_clean_file, 'r') as f:
            freq_clean = json.load(f)
            freq_clean = {int(k): v for k, v in freq_clean.items()}
    else:
        freq_noisy = {int(k): 1 for k, v in data_noisy.items()}
        freq_clean = {int(k): 1 for k, v in data_clean.items()}

    for name, data in zip(['noisy', 'cleaned'], [data_noisy, data_clean]):
        tmp_values = []
        tmp_count = []
        for k, v in data.items():
            # print(data.keys())
            # print(freq_noisy.keys())
            # exit(1)
            tmp_values.append(v)
            if name == 'noisy':
                right_frequencies = freq_noisy
            else:
                right_frequencies = freq_clean
            assert k in right_frequencies
            tmp_count.append(right_frequencies[k])

        # averaging
        tmp_mean = np.average(tmp_values, weights=tmp_count)
        tmp_std = math.sqrt(np.average((tmp_values-tmp_mean)**2, weights=tmp_count))

        # round 
        tmp_mean = round(tmp_mean, 2)
        tmp_std = round(tmp_std, 2)
        print("Proportion of NNs that share the right class with {} classes. Aggregation:{} || Mean: {} || STD: {} . ".format(name, classes_type, tmp_mean, tmp_std ))


def parse_args():
    """
    Parse input arguments
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parsing
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--results_clean', type=str, default='./analysis/knn/knn_euclidean_distance_nn10_feat_cleaned.json', help='File with results about clean features.')
    parser.add_argument('--results_noisy', type=str, default='./analysis/knn/knn_euclidean_distance_nn10_feat_noisy.json', help='File with results about noisy features.')
    parser.add_argument('--noisy_freq_file', type=str, default='./analysis/proposals_class_frequency-extracted_features_develop_VG.json', help='File where all the frequencies are reposted for clean extracted features.')
    parser.add_argument('--clean_freq_file', type=str, default='./analysis/proposals_class_frequency-extracted_features_new_classes_v3_VG.json', help='File where all the frequencies are reposted for clean extracted features.')
    parser.add_argument('--output_folder', type=str, default='./', help='Folder where to save the output file.')
    parser.add_argument('--labels', dest='labels',
                    help='File containing the new cleaned labels. It is needed for extracting the old and new classes indexes.',
                    default="./evaluation/objects_vocab_cleaned.txt",
                    type=str)
    parser.add_argument('--classes', dest='classes',
                help='Classes to consider.',
                default='all',
                choices=['all', 'untouched', 'new'],
                type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # get labels and mapping function
    map_fn, cleaned_labels, old_labels = create_mapping(args.labels)
    map_fn_reverse = defaultdict(list)
    for k, v in map_fn.items():
        map_fn_reverse[v].append(k)
    merged_noisy_cls_idx = [v for k, list_idx in map_fn_reverse.items() for v in list_idx if len(list_idx) > 1]
    merged_clean_cls_idx = [k for k, v in map_fn_reverse.items() if len(v) > 1]

    # get features comparison results 
    print('Loading all data.')
    with open(args.results_clean, 'r') as f:
        results_clean = json.load(f)
    with open(args.results_noisy, 'r') as f:
        results_noisy = json.load(f)

    # filter results according to selected classes
    # NOTE: here we need different indexes in filtering. Remember that noisy examples are in [0,1599] and clean are in [0, 877]
    data_clean = filter_classes(results_clean, merged_clean_cls_idx, args.classes)
    data_noisy = filter_classes(results_noisy, merged_noisy_cls_idx, args.classes)

    # check if the folder exists
    if os.path.exists(args.output_folder):
        print("Start plotting")
        print_results(data_clean, data_noisy, args.classes, args.noisy_freq_file, args.clean_freq_file)
    else:
        print("Folder not valid: ", args.output_folder)
        exit(1)
    
