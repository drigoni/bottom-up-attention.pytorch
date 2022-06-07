#!/usr/bin/env python
"""
Created on 5/05/22
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file includes the code for counting the classes instances.
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


def load_dataset(annotations_file):
    '''
    This function loads the Visual Genome annotations.
    :param annotations_file: annotations file.
    :return: annotations data. 
    '''
    print("Loading Visual Genome annotations in {} .".format(annotations_file))
    with open(annotations_file, 'r') as file:
        annotations = json.load(file)
    return annotations


def get_all_classes(annotations):
    '''
    Makes a dictionary for all the categories.
    :param annotations: annotations file.
    return dictionary of the classes.
    '''
    categories = {}
    for el in annotations['categories']:
        categories[el['id']] = el['name']
    return categories

def get_all_boxes(annotations):
    '''
    Makes a dictionary with all the GT bounding boxes.
    :param annotations: annotations file.
    return  dictionary of the GT bounding boxes.
    '''
    boxes = {}
    for el in annotations['annotations']:
        boxes[el['id']] = el['category_id']
    return boxes

def get_boxes_freq_by_class(boxes, categories):
    '''
    Makes a dictionary {class: frequency_of_the_class} counting all the occurrence in the GT bounding boxes.
    :param boxes: dictionary of the boxes.
    :param categories: dictionary of the categories.
    return  dictionary with all the frequencies by class.
    '''
    counting = {val: 0 for val in categories.values()}
    for key, val in boxes.items():
        label = categories[val]
        counting[label] += 1
    return counting

def draw_plot(counting, output):
    '''
    Plot the classes frequencies.
    :param counting: the frequency for each class.
    :param output: the path of the output file.
    '''
    counting_sorted = dict(sorted(counting.items(), key=lambda i: i[1], reverse=True))
    x_axis = list(range(len(counting_sorted)))
    plt.plot(x_axis, counting_sorted.values())
    plt.title("Boxes frequencies by category")
    ax = plt.gca()
    ax.set_xlabel('Category')        
    ax.set_ylabel('Frequency')
    plt.savefig(output)  
    print('Saved plot: {}'.format(output))
    text_output = output[:-4] + '.txt'
    with open(text_output, 'w') as f:
        json.dump(counting_sorted, f, indent=2)
    print('Saved file: {}'.format(text_output))


def draw_plots_together(counting1, counting2, output):
    # plot first dictionary
    counting1_sorted = dict(sorted(counting1.items(), key=lambda i: i[1], reverse=True))
    x_axis1 = list(range(len(counting1_sorted)))
    plt.plot(x_axis1, [math.log(i, 10) for i in counting1_sorted.values()])
    # plot second dictionary
    counting2_sorted = dict(sorted(counting2.items(), key=lambda i: i[1], reverse=True))
    x_axis2 = list(range(len(counting2_sorted)))
    plt.plot(x_axis2, [math.log(i, 10) for i in counting2_sorted.values()])
    plt.title("Boxes frequencies by category")
    ax = plt.gca()
    ax.set_xlabel('Category')        
    ax.set_ylabel('log(Frequency)')
    ax.legend(['Noisy categories', 'Cleaned categories'])
    plt.savefig(output)  
    print('Saved plot: {}'.format(output))

def draw_loglog_plots_together(counting1, counting2, output):
    # plot first dictionary
    counting1_sorted = dict(sorted(counting1.items(), key=lambda i: i[1], reverse=True))
    x_axis1 = list(range(len(counting1_sorted)))
    plt.loglog(x_axis1, counting1_sorted.values(), base=10)
    # plot second dictionary
    counting2_sorted = dict(sorted(counting2.items(), key=lambda i: i[1], reverse=True))
    x_axis2 = list(range(len(counting2_sorted)))
    plt.loglog(x_axis2, counting2_sorted.values(), base=10)
    plt.title("Boxes frequencies by category")
    ax = plt.gca()
    ax.set_xlabel('log(Category)')        
    ax.set_ylabel('log(Frequency)')
    ax.legend(['Noisy categories', 'Cleaned categories'])
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
    parser.add_argument('--file', dest='file',
                        help='Dataset file or frequency file.',
                        default="./datasets/visual_genome/annotations/visual_genome_train.json",
                        type=str)
    parser.add_argument('--file2', dest='file2',
                        help='None or frequency file.',
                        default=None,
                        type=str)
    parser.add_argument('--output', dest='output',
                        help='Dataset file.',
                        default="./classes_frequency.pdf",
                        type=str)
    parser.add_argument('--loglog', dest='loglog',
                        help='True to plot log-log plot.',
                        action='store_true')
    parser.set_defaults(loglog=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.file2 is None:
        # extract frequency just for one split of dataset
        annotations = load_dataset(args.file)
        categories = get_all_classes(annotations)
        boxes = get_all_boxes(annotations)
        counting = get_boxes_freq_by_class(boxes, categories)
        draw_plot(counting, args.output)
    else:
        with open(args.file, 'r') as f:
            counting1 = json.load(f)
        with open(args.file2, 'r') as f:
            counting2 = json.load(f)
        # plots together the frequencies reported in two files
        if args.loglog is False:
            draw_plots_together(counting1, counting2, args.output)
        else:
            draw_loglog_plots_together(counting1, counting2, args.output)