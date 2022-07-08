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
from posixpath import split
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



def create_mapping(labels_file):
    '''
    This function creates the mapping function from the untouched classes to the new ones.
    :param labels_file: new classes.
    :return: mapping function, index to labels name for new classes, index to labels name for untouched classes
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
    untouched_labels = dict()
    for new_label_id, new_label_str in cleaned_labels.items():
        new_label_id = int(new_label_id)
        for piece in new_label_str.split(','):
            tmp = piece.split(':')
            assert len(tmp) == 2
            untouched_label_id = int(tmp[0])
            untouched_label_str = tmp[1]
            # we need to avoid overriding of same ids like: 17:stop sign,17:stopsign
            if untouched_label_id not in untouched_labels.keys():
                untouched_labels[untouched_label_id] = untouched_label_str
                map_fn[untouched_label_id] = new_label_id
            else:
                print('Warning: label already present for {}:{}. Class {} ignored. '.format(untouched_label_id,
                                                                                            untouched_labels[untouched_label_id],
                                                                                            untouched_label_str))
    assert len(untouched_labels) == 1600
    assert len(untouched_labels) == len(map_fn)
    # print(untouched_labels[1590], map_fn[1590], cleaned_labels[map_fn[1590]])
    return map_fn, cleaned_labels, untouched_labels     # all in [1, 1600]


def make_buckets_and_filtering(data, untouched_cls, classes, n_splits=5):
    # FILTERING
    print("Data points before filtering: {} .".format(len(data)))
    if str.lower(classes) == 'untouched':
        data = {k: v for k, v in data.items() if k in untouched_cls}
    elif str.lower(classes) == 'new':
        data = {k: v for k, v in data.items() if k not in untouched_cls}
    # data = {k: v for k, v in data.items() if v[0] <= 400}
    print("Data points after filtering: {} .".format(len(data)))

    # make buckets
    data = dict(sorted(data.items(), key=lambda i: float(i[1][2]), reverse=True))
    n_data = len(data)
    threshold = n_data//n_splits + 1 # floor(n_Data/n_splits)
    splits = [dict() for i in range(n_splits)]
    for i, (k, v) in zip(range(0, len(data)), data.items()):
        split_idx = math.floor((i+1)/threshold)
        splits[split_idx][k] = v
    return splits

def apply_data_transformation(data):
    # CODE BY CLASS
    tmp = [(v[0], v[1]) for k, v in data.items()]
    tmp = list(sorted(tmp, key=lambda i: float(i[0]), reverse=False))
    tmp_aps = [v[1] for v in tmp]
    print("Final scores: ", np.mean(tmp_aps))

    # CUMULATIVE RESULTS
    cum_data = dict()
    for i in range(len(tmp_aps)):
        cum_data[i] = np.sum(tmp_aps[:i+1]) # at maximum
        # cum_data[i] = np.sum(tmp_aps[i:]) # at minimum
    data = cum_data

    # ORDERING
    data = dict(sorted(data.items(), key=lambda i: float(i[0]), reverse=False))
    return data

def draw_plots_together(counting1, counting2, output, classes):
    plt.clf()
    # plot first dictionary
    plt.plot(counting1.keys(), counting1.values(), linewidth=1, linestyle='-', label='BU post-processing')
    # plot second dictionary
    plt.plot(counting2.keys(), counting2.values(), linewidth=1, linestyle='-', label='BU cleaned classes')
    if str.lower(classes) == 'untouched':
        plt.title('Old classes')
    elif str.lower(classes) == 'new':
        plt.title('New classes')
    else:
        plt.title('All classes')
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.set_xlabel('Classes ordered by frequency')        
    ax.set_ylabel('Cumulative AP')
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
                        help='Root funtoucheder.',
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
    parser.add_argument('--freq1', dest='freq1',
                    help='Training Frequency file.',
                    default=None,
                    type=str)
    parser.add_argument('--freq2', dest='freq2',
                        help='Training Frequency file.',
                        default=None,
                        type=str)
    parser.add_argument('--labels', dest='labels',
                    help='File containing the new cleaned labels.',
                    default="./evaluation/objects_vocab.txt",
                    type=str)
    parser.add_argument('--output', dest='output',
                        help='Dataset file.',
                        default="./aps_scores_by_bucket.pdf",
                        type=str)
    parser.add_argument('--classes', dest='classes',
                help='Classes to consider.',
                choices=['all', 'untouched', 'new'],
                default='all',
                type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # loading data points
    with open(args.file1, 'r') as f:
        counting1 = json.load(f)
    with open(args.file2, 'r') as f:
        counting2 = json.load(f)

    # loading training frequencies.
    # note that the files contains labels(keys) in long format as: "73:sky,1217:weather"
    with open(args.freq1, 'r') as f:
        freq1 = json.load(f)
    with open(args.freq2, 'r') as f:
        freq2 = json.load(f)

    # update frequencies
    for k, v in freq1.items():
        compact_key = k.split(',')[0]
        count = counting1[compact_key][0][0]
        ap =  counting1[compact_key][0][1]
        training_count = v
        counting1[compact_key] = [count, ap, training_count]
    for k, v in freq2.items():
        compact_key = k.split(',')[0]
        count = counting2[compact_key][0][0]
        ap =  counting2[compact_key][0][1]
        training_count = v
        counting2[compact_key] = [count, ap, training_count]
    
    # get labels
    map_fn, cleaned_labels, untouched_labels = create_mapping(args.labels)
    map_fn_reverse = defaultdict(list)
    for k, v in map_fn.items():
        map_fn_reverse[v].append(k)
    subset_indexes = [k for k, v in map_fn_reverse.items() if len(v) == 1]
    untouched_cls = [cleaned_labels[idx] for idx in subset_indexes]
    # cleaning according to evaluation of BU model
    untouched_cls = [(cls.split(',')[0]).lower().strip() for cls in untouched_cls]
    print("Number of untouched classes: {} .".format(len(untouched_cls)))

    n_splits=5
    splits1 = make_buckets_and_filtering(counting1, untouched_cls, args.classes, n_splits)
    splits2 = make_buckets_and_filtering(counting2, untouched_cls, args.classes, n_splits)
    for i, counting1, counting2 in zip(range(1, n_splits+1), splits1, splits2):
        print("---------- Split: ", i)
        output_file = "{}_spl{}.pdf".format(args.output[:-4], str(i))
        # transform data
        print("Processing data points in: {} .".format(args.file1))
        counting1 = apply_data_transformation(counting1)
        print("Processing data points in: {} .".format(args.file2))
        counting2 = apply_data_transformation(counting2)

        # plots together the frequencies reported in two files
        draw_plots_together(counting1, counting2, output_file, args.classes)
