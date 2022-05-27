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
    data = {k: v[0] for k, v in data.items()}

    # CHECK
    assert len(data) == 878
    assert len(data) >= len(subset_cls)
    #for i in subset_cls:
    #    if i not in data.keys():
    #        print("error: ", i)

    # FILTERING
    print("Data points before filtering: {} .".format(len(data)))
    # data = {k: v for k, v in data.items() if k in subset_cls}
    # data = {k: v for k, v in data.items() if k not in subset_cls}
    # data = {k: v for k, v in data.items() if v[0] <= 400}
    print("Data points after filtering: {} .".format(len(data)))

    # CODE BY CLASS
    tmp = [(v[0], v[1]) for k, v in data.items()]
    tmp = list(sorted(tmp, key=lambda i: float(i[0]), reverse=False))
    # tmp_npos = [v[0] for v in tmp]
    tmp_aps = [v[1] for v in tmp]
    print("Final scores: ", np.mean(tmp_aps))
    # data = {i: ap for i, ap in enumerate(tmp_aps)}
    # cum_data = dict()
    # for i in range(len(tmp_aps)):
        # cum_data[i] = np.mean(tmp_aps[:i+1]) # at maximum
        # cum_data[i] = np.mean(tmp_aps[i:]) # at minimum
    # data = cum_data
    # CUMULATIVE RESULTS
    cum_data = dict()
    for i in range(len(tmp_aps)):
        cum_data[i] = np.sum(tmp_aps[:i+1]) # at maximum
        # cum_data[i] = np.sum(tmp_aps[i:]) # at minimum
    data = cum_data


    # # GROUPING (maybe not the best way to visualize)
    # tmp_data = defaultdict(list)
    # for k, v in data.items():
    #     n = v[0]
    #     ap = v[1]
    #     tmp_data[n].append(ap)
    # data = {k: np.mean(v) for k, v in tmp_data.items()}
    # # SIM CUMULATIVE RESULTS
    # data = dict(sorted(data.items(), key=lambda i: float(i[0]), reverse=False))
    # nposs, aps = list(data.keys()), list(data.values())
    # cum_data = dict()
    # for i in range(len(nposs)):
    #     cum_data[nposs[i]] = np.mean(aps[:i+1]) # at maximum
    #     # cum_data[nposs[i]] = np.mean(aps[i:]) # at minimum
    #     # print('{}:{} .'.format(nposs[i], np.mean(aps[:i+1])))
    # data = cum_data
    # SIM CUMULATIVE BY STEPS
    # data = dict(sorted(data.items(), key=lambda i: float(i[0]), reverse=False))
    # nposs, aps = list(data.keys()), list(data.values())
    # cum_data = dict()
    # for i in [10, 30, 60, 100, 200, 300, 400, 600, 1000, 2000, 3000]:
    #     tmp = [p for n, p in zip(nposs, aps) if n <= i] # at maximum
    #     # tmp = [p for n, p in zip(nposs, aps) if n >= i] # at minimum
    #     cum_data[i] = np.mean(tmp) # at maximum
    # data = cum_data

    # ORDERING
    data = dict(sorted(data.items(), key=lambda i: float(i[0]), reverse=False))
    return data

def draw_plots_together(counting1, counting2, output):
    # plot first dictionary
    plt.plot(counting1.keys(), counting1.values(), linewidth=1, linestyle='-', label='BU post-processing')
    # plot second dictionary
    plt.plot(counting2.keys(), counting2.values(), linewidth=1, linestyle='-', label='BU cleaned classes')
    plt.title("AP scores")
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.set_xlabel('Classes ordered by frequency')        
    ax.set_ylabel('AP scores')
    # ax.legend(['post-processing', 'cleaned classes'])
    plt.savefig(output)  
    print('Saved plot: {}'.format(output))

def draw_loglog_plots_together(counting1, counting2, output):
    # plot first dictionary
    plt.loglog(counting1.keys(), counting1.values(), linewidth=1, linestyle='-', label='post-processing')
    # plot second dictionary
    plt.loglog(counting2.keys(), counting2.values(), linewidth=1, linestyle='-', label='cleaned classes')
    plt.title("AP scores")
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.set_xlabel('log(Classes ordered by frequency)')        
    ax.set_ylabel('log(AP scores)')
    # ax.legend(['post-processing', 'cleaned classes'])
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
    # cleaning according to evaluation of BU model
    subset_cls = [(cls.split(',')[0]).lower().strip() for cls in subset_cls]
    print("Number of untouched classes: {} .".format(len(subset_cls)))

    # transform data
    print("Processing data points in: {} .".format(args.file1))
    counting1 = apply_data_transformation(counting1, subset_cls)
    print("Processing data points in: {} .".format(args.file2))
    counting2 = apply_data_transformation(counting2, subset_cls)

    # plots together the frequencies reported in two files
    if args.loglog is False:
        draw_plots_together(counting1, counting2, args.output)
    else:
        draw_loglog_plots_together(counting1, counting2, args.output)