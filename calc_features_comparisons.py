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
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity



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


def load_data(img_folder, map_fn_reverse, classes_type, model_type, images_name):
    print("Considering just classes type: ", classes_type)
    print("Model type: ", model_type)
    max_number_of_classes = 877 if model_type =='cleaned' else 1599
    # get new classes labels
    if classes_type != 'all':
        if model_type == 'noisy':
            untouched_cls_idx = {v[0]: k for k, v in map_fn_reverse.items() if len(v) == 1}
        elif model_type == 'cleaned':
            untouched_cls_idx = {k: v[0]  for k, v in map_fn_reverse.items() if len(v) == 1}
        else:
            print('Error in model type: ', model_type)
            exit(1)
        untouched_cls_idx = [k-1 for k, v in untouched_cls_idx.items()]

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
                if classes_type == 'untouched':
                    if box_label_idx in untouched_cls_idx:
                        filtered_boxes.append(data_boxes[box_idx])
                        filtered_features.append(data_features[box_idx])
                elif classes_type == 'new':
                    if box_label_idx not in untouched_cls_idx:
                        filtered_boxes.append(data_boxes[box_idx])
                        filtered_features.append(data_features[box_idx])
                elif classes_type == 'all':
                    filtered_features.append(data_features[box_idx])
                    filtered_boxes.append(data_boxes[box_idx])
                else:
                    print('Error.')
                    exit(1)
            
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


def exec_features_comparison(all_data_noisy, all_data_clean, map_fn_reverse, categories_noisy, categories_clean, classes_type, output_file):

    accepted_classes = {
        'man': [0, 0, 0],
        'person': [128, 128, 128],
        'window': [255, 0, 0],
        'shirt': [128, 0, 0],
        'tree': [255, 255, 0],
        'building': [128, 128, 0],
        'wall': [0, 255, 0],
        'sky': [0, 128, 0],
        'sign': [0, 255, 255],
        'woman': [0, 0, 255],
        'ground': [255, 0, 255],
    }

    # group features by class, REMEMBER empty list when there are no bounding boxes extracted for some classes
    features_per_class_noisy = {k: [] for k in range(len(categories_noisy))}    # index starting from 0
    features_per_class_clean = {k: [] for k in range(len(categories_clean))}    # index starting from 0
    for cls, features in zip(all_data_noisy['cls'], all_data_noisy['features']):
        for c_index, f in zip(cls, features):
            # cls_label = categories_noisy[c_index]
            # features_per_class_noisy[cls_label].append(f)
            features_per_class_noisy[c_index].append(f)
    for cls, features in zip(all_data_clean['cls'], all_data_clean['features']):
        for c_index, f in zip(cls, features):
            # cls_label = categories_clean[c_index]
            # features_per_class_clean[cls_label].append(f)
            features_per_class_clean[c_index].append(f)  

    # calculate clusters coordinates
    clusters_noisy = {k: np.mean(np.array(v), axis=0) for k, v in features_per_class_noisy.items() if len(v) > 0}    # index starting from 0
    clusters_clean = {k: np.mean(np.array(v), axis=0) for k, v in features_per_class_clean.items() if len(v) > 0}    # index starting from 0

    # calculate cosine_similarity
    clusters_distance_noisy = cosine_distances(list(clusters_noisy.values()))
    clusters_distance_clean = cosine_distances(list(clusters_clean.values()))
    
    comparison = {}
    for k, list_noisy_idx in map_fn_reverse.items():      # starting from 1
        # check that the clean class is in the extracted features
        if k-1 in clusters_clean.keys():
            # mean and variance for clean classes
            tmp_clean_key = list(clusters_clean.keys()).index(k-1)
            mean_clean = np.mean(clusters_distance_clean[tmp_clean_key])
            std_clean = np.std(clusters_distance_clean[tmp_clean_key])
            # mean and variance for noisy classes
            tmp_noisy_keys = [list(clusters_noisy.keys()).index(idx-1) for idx in list_noisy_idx if idx-1 in clusters_noisy.keys()]
            if len(tmp_noisy_keys) > 0:
                tmp_noisy_features = np.stack([clusters_distance_noisy[key] for key in tmp_noisy_keys], axis=0)
                mean_noisy = np.mean(tmp_noisy_features)
                std_noisy = np.std(tmp_noisy_features)
                comparison[k] = [float(mean_clean), float(std_clean), float(mean_noisy), float(std_noisy), categories_clean[k-1],  len(list_noisy_idx)]
            else:
                comparison[k] = [float(mean_clean), float(std_clean), None, None, categories_clean[k-1], len(list_noisy_idx)]
                print("No noisy classes found for clean class: {}|{}".format(k-1, categories_clean[k-1]))

    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
        print('Saved file: {}'.format(output_file))

    # plot hitmap
    # plt.imshow(clusters_distance_noisy, cmap='hot')
    # plt.colorbar()
    # # plt.imshow(clusters_distance_clean, cmap='hot')
    # plt.savefig(output_file, dpi=1000)
    
    exit(1)


def parse_args():
    """
    Parse input arguments
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parsing
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--extracted_features_noisy', type=str, default='./extracted_features_develop_VG/', help='Folder of extracted features')
    parser.add_argument('--extracted_features_clean', type=str, default='./extracted_features_new_classes_v3_VG/', help='Folder of extracted features')
    parser.add_argument('--output_folder', type=str, default='./proposals_features_t-sne.pdf', help='Folder where to save the output file.')
    parser.add_argument('--split_file_noisy', type=str, default='./datasets/visual_genome/annotations/visual_genome_val.json', help='Dataset.')
    parser.add_argument('--split_file_clean', type=str, default='./datasets/cleaned_visual_genome/annotations/cleaned_visual_genome_val.json', help='Dataset.')
    parser.add_argument('--n_limit', type=int, default=1000)
    parser.add_argument('--labels', dest='labels',
                    help='File containing the new cleaned labels. It is needed for extracting the old and new classes indexes.',
                    default="./evaluation/objects_vocab.txt",
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
    # get labels
    map_fn, cleaned_labels, old_labels = create_mapping(args.labels)
    map_fn_reverse = defaultdict(list)
    for k, v in map_fn.items():
        map_fn_reverse[v].append(k)

    # get images names 
    with open(args.split_file_noisy, 'r') as f:
        dataset = json.load(f)
    images_name = [i['file_name'][:-4] for i in dataset['images']]
    categories_noisy = {int(i['id']): i['name']  for i in dataset['categories']}
    with open(args.split_file_clean, 'r') as f:
        dataset = json.load(f)
    images_name = [i['file_name'][:-4] for i in dataset['images']]
    categories_clean = {int(i['id']): i['name']  for i in dataset['categories']}

    # check if the folder exists
    if os.path.exists(args.extracted_features_noisy) and os.path.exists(args.extracted_features_clean):
        print('Loading all data.')
        all_data_noisy = load_data(args.extracted_features_noisy, map_fn_reverse, args.classes, "noisy", images_name)
        all_data_clean = load_data(args.extracted_features_clean, map_fn_reverse, args.classes, "cleaned", images_name)
        print("Start calculation")
        exec_features_comparison(all_data_noisy, all_data_clean, map_fn_reverse, categories_noisy, categories_clean, args.classes, args.output_folder)
    else:
        print("Folder not valid. ")
        exit(1)
    
