#!/usr/bin/env python
"""
Created on 5/05/22
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file includes the code needed plotting scores.
"""
from email.policy import default
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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns


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


def select_information(data, inf_type):
    if inf_type == 'score':
        result =  {k: np.mean(v) for k ,v in data.items()}
    elif inf_type == 'freq':
        result =  {k: len(v) for k ,v in data.items()}
    else:
        print("Type of information {} not correct.".format(inf_type))
        exit(1)
    result = dict(sorted(result.items(), key=lambda i: i[0], reverse=False))
    return result


def extract_data(output, old_cls_ids, classes_type):
    pred_by_classes = {cls_name: [] for cls_name in range(0, 878)}
    for img_pred in output:
        image_id = img_pred['image_id']
        boxes = img_pred['boxes']
        label_ids = img_pred['labels']  # from [0, 877]
        scores = img_pred['scores']
        for box, label_id, score in zip(boxes, label_ids, scores):
            # assert label_id >=0 and label_id < 878
            if str.lower(classes_type) == 'all':
                pred_by_classes[label_id].append(score)
            elif str.lower(classes_type) == 'old':
                if label_id in old_cls_ids:
                    pred_by_classes[label_id].append(score)
            elif str.lower(classes_type) == 'new':
                if label_id not in old_cls_ids:
                    pred_by_classes[label_id].append(score)
            else:
                print('Error. Type of class error {}.'.format(classes_type))
    # sort
    pred_by_classes = dict(sorted(pred_by_classes.items(), key=lambda i: i[0], reverse=False))
    return pred_by_classes


def draw_plots_together(data1, data2, output, classes_type):
    # default colors
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # matplotlib.rcParams.update({'font.size': 22})
    assert len(data1) == len(data2)
    # plotting
    sns.kdeplot(data=data1,
                color='#1f77b4',
                cut=0,
                common_norm=False,
                label='BUA Original Mapped to Clean')
    sns.kdeplot(data=data2,
                color='#ff7f0e',
                cut=0,
                common_norm=False,
                label='BUA Cleaned')
    if str.lower(classes_type) == 'all':
        plt.title('All Categories', fontsize=22)   
    elif str.lower(classes_type) == 'old':
        plt.title('Untouched Categories', fontsize=22)   
    elif str.lower(classes_type) == 'new':
        plt.title('Merged Categories', fontsize=22)   
    else:
        print('Error. Type of class error {}.'.format(classes_type))
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.set_xlabel('Probability Scores', fontsize=16)    
    ax.set_ylabel('Density', fontsize=16)    
    # ax.set_ylabel('Average Confidence')
    plt.tight_layout(pad=0.05)
    plt.xlim([0, 0.57])
    plt.ylim([0, 7])
    plt.savefig(output,  dpi=300)  
    print('Saved plot: {}'.format(output))


def draw_plot(data, output, label, color, classes_type):
    # default colors
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # matplotlib.rcParams.update({'font.size': 14})
    # plotting
    sns.kdeplot(data=data,
                color=color,
                cut=0,
                common_norm=False,
                label=label)
    if str.lower(classes_type) == 'all':
        plt.title('All Categories', fontsize=22)
    elif str.lower(classes_type) == 'old':
        plt.title('Untouched Categories', fontsize=22)
    elif str.lower(classes_type) == 'new':
        plt.title('Merged Categories', fontsize=22)
    else:
        print('Error. Type of class error {}.'.format(classes_type))
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.set_xlabel('Probability Scores', fontsize=16)      
    # ax.set_ylabel('Average Confidence')
    plt.tight_layout(pad=0.1)
    plt.xlim([0, 0.57])
    plt.ylim([0, 7])
    plt.savefig(output,  dpi=300)  
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
                        help='Result file1.',
                        default=None,
                        type=str)
    parser.add_argument('--file2', dest='file2',
                        help='Result file2.',
                        default=None,
                        type=str)
    parser.add_argument('--labels', dest='labels',
                        help='File containing the new cleaned labels.',
                        default="./evaluation/objects_vocab.txt",
                        type=str)
    parser.add_argument('--output', dest='output',
                        help='Dataset file.',
                        default="./output_scores.pdf",
                        type=str)
    parser.add_argument('--inf_type', dest='inf_type',
                    help='Information to consider. score, freq.',
                    default='score',
                    type=str)
    parser.add_argument('--classes', dest='classes',
                        help='Classes to consider. all, old, new.',
                        default='all',
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
    subset_indexes = [k for k, v in map_fn_reverse.items() if len(v) == 1]
    old_cls_idx = [k-1 for k in subset_indexes] # index starting from 0 e not 1
    # cleaning according to evaluation of BUA model

    # output_data -> list[dict] where each dict has keys: ["image_id", "boxes", "labels", "scores"]
    if args.file1 is not None and args.file2 is not None:
        # loading data points
        output_data1 = torch.load(args.file1)
        output_data2 = torch.load(args.file2)
        # data1 = extract_data(output_data1, map_fn) # only used with file that have 1600 classes
        data1 = extract_data(output_data1, old_cls_idx, args.classes)
        data2 = extract_data(output_data2, old_cls_idx, args.classes)
        data1 = select_information(data1, args.inf_type)
        data2 = select_information(data2, args.inf_type)
        draw_plots_together(data1, data2, args.output, args.classes)
    elif args.file1 is not None and args.file2 is None:
        # loading data points
        output_data = torch.load(args.file1)
        data = extract_data(output_data, old_cls_idx, args.classes)
        data = select_information(data, args.inf_type)
        draw_plot(data, args.output, 'BUA Original Mapped to Clean', '#1f77b4', args.classes)
    elif args.file1 is None and args.file2 is not None:
        # loading data points
        output_data = torch.load(args.file2)
        data = extract_data(output_data, old_cls_idx, args.classes)
        data = select_information(data, args.inf_type)
        draw_plot(data, args.output, 'BUA Cleaned', '#ff7f0e', args.classes)
    else:
        print("Command line parameter error.")
