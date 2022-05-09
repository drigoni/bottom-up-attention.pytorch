#!/usr/bin/env python
"""
Created on 5/05/22
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file includes the code needed for creating the new clean dataset.
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
    import matplotlib.pyplot as plt
    counting_sorted = dict(sorted(counting.items(), key=lambda i: i[1], reverse=True))
    x_axis = list(range(len(counting_sorted)))
    plt.plot(x_axis, counting_sorted.values())
    plt.title("Boxes frequencies by classes")
    ax = plt.gca()
    ax.set_xlabel('Classes')        
    ax.set_ylabel('Frequency')
    plt.savefig(output)  
    print('Saved plot: {}'.format(output))
    text_output = output[:-4] + '.txt'
    with open(text_output, 'w') as f:
        json.dump(counting_sorted, f, indent=2)
    print('Saved file: {}'.format(text_output))


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
                        help='Dataset file.',
                        default="./datasets/visual_genome/annotations/visual_genome_train.json",
                        type=str)
    parser.add_argument('--output', dest='output',
                    help='Dataset file.',
                    default="./classes_frequency.pdf",
                    type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    annotations = load_dataset(args.file)
    categories = get_all_classes(annotations)
    boxes = get_all_boxes(annotations)
    counting = get_boxes_freq_by_class(boxes, categories)
    draw_plot(counting, args.output)