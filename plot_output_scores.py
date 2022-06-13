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
import matplotlib.pyplot as plt
import numpy as np
import torch


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
        return {k: np.mean(v) for k ,v in data.items()}
    elif inf_type == 'freq':
        return {k: len(v) for k ,v in data.items()}
    else:
        print("Type of information {} not correct.".format(inf_type))
        exit(1)


def extract_data(output, map_fn=None, old_cls=None, classes=None):
    pred_by_classes = {cls_name: [] for cls_name in range(0, 879)}
    for img_pred in output:
        image_id = img_pred['image_id']
        boxes = img_pred['boxes']
        label_ids = img_pred['labels']
        scores = img_pred['scores']
        for box, label_id, score in zip(boxes, label_ids, scores):
            if map_fn is None:
                new_label_id = label_id
            else:
                new_label_id = map_fn[label_id + 1] - 1
            pred_by_classes[new_label_id].append(score)
    # sort
    pred_by_classes = dict(sorted(pred_by_classes.items(), key=lambda i: i[0], reverse=False))
    return pred_by_classes


# def draw_plots_together(data1, data2, output):
#     assert len(data1) == len(data2)
#     # get mean
#     means = [np.nanmean(list(data1.values())), -np.nanmean(list(data2.values()))]
#     print("BU post-processing mean {} over {} predicted classes".format(means[0], sum(~np.isnan(list(data1.values())))))
#     print("BU cleaned classes mean {} over {} predicted classes".format(means[1], sum(~np.isnan(list(data2.values())))))
#     # plotting
#     plt.bar(data1.keys(), data1.values(),  label='BU post-processing', color='#1f77b4')
#     plt.bar(data2.keys(), [-a for a in data2.values()], label='BU cleaned classes', color='#ff7f0e')
#     plt.hlines(means, 0, 878, linewidth=0.5, linestyle='dashed', colors='black', label='Mean')
#     plt.title('Classes Predictions')
#     plt.legend(loc="upper right")
#     ax = plt.gca()
#     ax.set_xlabel('Classes')        
#     ax.set_ylabel('Confidence')
#     plt.savefig(output)  
#     print('Saved plot: {}'.format(output))

def draw_plots_together(data1, data2, output):
    # default colors
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    assert len(data1) == len(data2)
    # get mean
    diff = [abs(a-b) for a, b in zip(data1.values(), data2.values())]
    # plotting
    print("Average absolute differences {} over {} predicted classes".format(np.nanmean(diff), sum(~np.isnan(diff))))
    plt.bar(data1.keys(), diff,  label='ABS score differences', color='#2ca02c')
    plt.hlines([np.nanmean(diff)], 0, 878, linewidth=1, linestyle='dashed', colors='black', label='Mean')
    plt.title('Classes Predictions')
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.set_xlabel('Classes')        
    ax.set_ylabel('Average Absolute Difference')
    plt.savefig(output)  
    print('Saved plot: {}'.format(output))


def draw_plot(data, output, label, color):
    # plotting
    means = [np.nanmean(list(data.values()))]
    print("Average {} over {} predicted classes".format(means[0], sum(~np.isnan(data.values()))))
    plt.bar(data.keys(), data.values(),  label=label, color=color)
    plt.hlines(means, 0, max(data.keys()), linewidth=1, linestyle='dashed', colors='black', label='Mean')
    # plot second dictionary
    plt.title('Classes Predictions')
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.set_xlabel('Classes')        
    ax.set_ylabel('Average Confidence')
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
    # map_fn, cleaned_labels, old_labels = create_mapping(args.labels)
    # map_fn_reverse = defaultdict(list)
    # for k, v in map_fn.items():
    #     map_fn_reverse[v].append(k)
    # subset_indexes = [k for k, v in map_fn_reverse.items() if len(v) == 1]
    # old_cls = [cleaned_labels[idx] for idx in subset_indexes]
    # # cleaning according to evaluation of BU model
    # old_cls = [(cls.split(',')[0]).lower().strip() for cls in old_cls]

    # output_data -> list[dict] where each dict has keys: ["image_id", "boxes", "labels", "scores"]
    if args.file1 is not None and args.file2 is not None:
        # loading data points
        output_data1 = torch.load(args.file1)
        output_data2 = torch.load(args.file2)
        # data1 = extract_data(output_data1, map_fn) # only used with file that have 1600 classes
        data1 = extract_data(output_data1)
        data2 = extract_data(output_data2)
        data1 = select_information(data1, args.inf_type)
        data2 = select_information(data2, args.inf_type)
        draw_plots_together(data1, data2, args.output)
    elif args.file1 is not None and args.file2 is None:
        # loading data points
        output_data = torch.load(args.file1)
        data = extract_data(output_data)
        data = select_information(data, args.inf_type)
        draw_plot(data, args.output, 'BU post-processing', color='#1f77b4')
    elif args.file1 is None and args.file2 is not None:
        # loading data points
        output_data = torch.load(args.file2)
        data = extract_data(output_data)
        data = select_information(data, args.inf_type)
        draw_plot(data, args.output, 'BU cleaned classes', color='#ff7f0e')
    else:
        print("Command line parameter error.")
