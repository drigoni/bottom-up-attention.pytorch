#!/usr/bin/env python
"""
Created on 5/05/22
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file includes the code needed plotting scores.
"""
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
import matplotlib.pyplot as plt
import numpy as np



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

def apply_data_transformation(data, subset_cls):
    # filter data
    data = {k: v for k, v in data.items() if v[1] <= 3000}
    data = {k: v for k, v in data.items() if k in subset_cls}
    # order data
    data = dict(sorted(data.items(), key=lambda i: float(i[1][0]), reverse=False))
    # calc cumulative results
    #nposs, aps = list(data.keys()), list(data.values())
    #cum_data = dict()
    #for i in range(len(nposs)):
    #    cum_data[nposs[i]] = np.mean(aps[:i+1]) # at maximum
    #    # cum_data[nposs[i]] = np.mean(aps[i:]) # at minimum
    #    # print('{}:{} .'.format(nposs[i], np.mean(aps[:i+1])))
    #data = cum_data
    #data = dict(sorted(data.items(), key=lambda i: float(i[0]), reverse=False))
    # calc cumulative results by steps
    #nposs, aps = list(data.keys()), list(data.values())
    #cum_data = dict()
    #for i in [10, 30, 60, 100, 200, 300, 400, 600, 1000, 2000, 3000]:
    #    tmp = [p for n, p in zip(nposs, aps) if n <= i] # at maximum
    #    # tmp = [p for n, p in zip(nposs, aps) if n >= i] # at minimum
    #    cum_data[i] = np.mean(tmp) # at maximum
    #data = cum_data
    #data = dict(sorted(data.items(), key=lambda i: float(i[0]), reverse=False))
    return data

def draw_plots_together(counting1, counting2, output):
    # plot first dictionary
    plt.plot(counting1.keys(), counting1.values())
    # plot second dictionary
    plt.plot(counting2.keys(), counting2.values())
    plt.title("AP scores by number of GT boxes")
    ax = plt.gca()
    ax.set_xlabel('Number of GT boxes')        
    ax.set_ylabel('AP scores')
    ax.legend(['post-processing', 'Cleaned classes'])
    plt.savefig(output)  
    print('Saved plot: {}'.format(output))

def draw_loglog_plots_together(counting1, counting2, output):
    # plot first dictionary
    plt.loglog(counting1.keys(), counting1.values(), base=10)
    # plot second dictionary
    plt.loglog(counting2.keys(), counting2.values(), base=10)
    plt.title("AP scores by number of GT boxes")
    ax = plt.gca()
    ax.set_xlabel('log(Number of GT boxes)')        
    ax.set_ylabel('log(AP scores)')
    ax.legend(['post-processing', 'Cleaned classes'])
    plt.savefig(output)  
    print('Saved plot: {}'.format(output))

def parse_args():
    """
    Parse input arguments
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parsing
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--root', dest='root',
                        help='Root folder.',
                        default='{}/'.format(current_dir),
                        type=str)
    parser.add_argument('--file1', dest='file1',
                    help='Frequency file.',
                    default=None,
                    type=str)
    parser.add_argument('--file2', dest='file2',
                        help='Frequency file.',
                        default=None,
                        type=str)
    parser.add_argument('--labels', dest='labels',
                    help='File containing the new cleaned labels.',
                    default="./evaluation/objects_vocab.txt",
                    type=str)
    parser.add_argument('--output', dest='output',
                        help='Dataset file.',
                        default="./aps_scores.pdf",
                        type=str)
    parser.set_defaults(loglog=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # loading data points
    with open(args.file1, 'r') as f:
        counting1 = json.load(f)
    with open(args.file2, 'r') as f:
        counting2 = json.load(f)
    assert len(counting1) == len(counting2)

    # get labels
    map_fn, cleaned_labels, old_labels = create_mapping(args.labels)
    map_fn_reverse = defaultdict(list)
    for k, v in map_fn.items():
        map_fn_reverse[v].append(k)
    subset_indexes = [k for k, v in map_fn_reverse.items() if len(v) == 1]
    subset_cls = [cleaned_labels[idx] for idx in subset_indexes]

    # transform data
    counting1 = apply_data_transformation(counting1, subset_cls)
    counting2 = apply_data_transformation(counting2, subset_cls)

    # plots together the frequencies reported in two files
    if args.loglog is False:
        draw_plots_together(counting1, counting2, args.output)
    else:
        draw_loglog_plots_together(counting1, counting2, args.output)