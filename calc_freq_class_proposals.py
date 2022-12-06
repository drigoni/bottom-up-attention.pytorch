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


def load_data(img_folder, map_fn_reverse, model_type, images_name):
    print("Model type: ", model_type)
    max_number_of_classes = 877 if model_type =='cleaned' else 1599

    # get all extracted file in the folder
    onlyfiles = [join(img_folder, f) for f in listdir(img_folder) if isfile(join(img_folder, f))]
    print('Number of files: ', len(onlyfiles))
    onlyfiles = [f for f in onlyfiles if f[-4:] == '.npz']
    print('Number of .npz files: ', len(onlyfiles))

    # load all data ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    all_data = defaultdict(list)
    count_zeros = 0
    for img_file in onlyfiles:
        img_id = img_file.split('/')[-1][:-4]
        if img_id not in images_name:
            continue
        with np.load(img_file, allow_pickle=True) as f:
            data_info = f['info'].item() # check https://stackoverflow.com/questions/40219946/python-save-dictionaries-through-numpy-save
            # info = {
            #     "objects": classes.cpu().numpy(),
            #     "cls_prob": cls_probs.cpu().numpy(),
            #     'attrs_id': attr_probs,
            #     'attrs_scores': attr_scores,
            # }
            data_num_bbox = f['num_bbox']
            data_boxes = f['bbox']
            data_features = f['x']
            assert data_num_bbox == len(data_info['objects']) == len(data_info['cls_prob']) == len(data_boxes) == len(data_features)
            assert img_id not in all_data['image_id']

            # bounding boxes filtering according to its label 
            filtered_boxes = []
            filtered_features = []
            for box_idx in range(data_num_bbox):
                box_label_idx = data_info['objects'][box_idx]  # in [0, 877 or 1599]
                assert len(data_boxes[box_idx]) == 4
                assert 0 <= box_label_idx <= max_number_of_classes
                # NOTE: BE SURE EVERYTHING IS np.float32 WHEN DEALING WITH base64.b64encode() function
                filtered_features.append(data_features[box_idx])
                filtered_boxes.append(data_boxes[box_idx])
            
            if len(filtered_boxes) == 0:
                filtered_boxes.append(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
                filtered_features.append(data_features[0])
                count_zeros += 1

            all_data['image_id'].append(img_id)
            # all_data['image_w'].append(f['image_w'])
            # all_data['image_h'].append(f['image_h'])
            all_data['num_boxes'].append(len(filtered_boxes))
            # need to be encoded. See adaptive_detection_features_converter.py
            # all_data['boxes'].append(base64.b64encode(np.array(filtered_boxes)))  # need to be encoded. See adaptive_detection_features_converter.py
            all_data['features'].append(np.array(filtered_features))
            # all_data['image_h_inner'].append(f['image_h_inner'])
            # all_data['image_w_inner'].append(f['image_w_inner'])
            # all_data['info'].append(f['info'])
            all_data['cls'].append(f['info'].item()['objects'])
            # info = {
            #     "objects": classes.cpu().numpy(),
            #     "cls_prob": cls_probs.cpu().numpy(),
            #     'attrs_id': attr_probs,
            #     'attrs_scores': attr_scores,
            # }
        
    print("Number of images with zero boxes: ", count_zeros)
    return all_data


def get_class_frequency(extracted_features):
    counter_extracted_features = dict()
    flat_extracted_features = [v for v_list in extracted_features['cls'] for v in v_list]
    for c in flat_extracted_features:
        if c in counter_extracted_features:
            counter_extracted_features[c] += 1
        else:
            counter_extracted_features[c] = 1
    return  counter_extracted_features


def plot_class_frequency(c_features, categories, output_folder):
    results = dict()
    for k in categories.keys():
        if k in c_features:
            results[k] = c_features[k]
        else:
            results[k] = None

    output_file = "{}proposals_class_frequency.json".format(output_folder)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        print('Saved file: {}'.format(output_file))


def parse_args():
    """
    Parse input arguments
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parsing
    parser = argparse.ArgumentParser(description='Inputs')
    # parser.add_argument('--extracted_features', type=str, default='./extracted_features/extracted_features_develop_VG/', help='Folder of extracted features')
    parser.add_argument('--extracted_features', type=str, default='./extracted_features/extracted_features_new_classes_v3_VG/', help='Folder of extracted features')
    parser.add_argument('--output_folder', type=str, default='./', help='Folder where to save the output file.')
    # parser.add_argument('--split_file', type=str, default='./datasets/visual_genome/annotations/visual_genome_val.json', help='Dataset.')
    parser.add_argument('--split_file', type=str, default='./datasets/cleaned_visual_genome/annotations/cleaned_visual_genome_val.json', help='Dataset.')
    parser.add_argument('--labels', dest='labels',
                    help='File containing the new cleaned labels. It is needed for extracting the old and new classes indexes.',
                    default="./evaluation/objects_vocab_cleaned.txt",
                    type=str)
    parser.add_argument('--model', dest='model',
            help='Model trained on new classes (878 labels) or model post-processed (1600 to 878 labels).',
            default='noisy',
            choices=['noisy', 'cleaned'],
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

    # get images names 
    with open(args.split_file, 'r') as f:
        dataset = json.load(f)

    images_name = [i['file_name'][:-4] for i in dataset['images']]
    categories = {int(i['id']): i['name']  for i in dataset['categories']}

    # check if the folder exists
    if os.path.exists(args.extracted_features):
        print('Loading all data.')
        all_data = load_data(args.extracted_features, map_fn_reverse, args.model, images_name)
        counter_extracted_features = get_class_frequency(all_data)
        print("Start plotting")
        plot_class_frequency(counter_extracted_features, categories, args.output_folder)
    else:
        print("Folder not valid: ", args.extracted_features)
        exit(1)
    
