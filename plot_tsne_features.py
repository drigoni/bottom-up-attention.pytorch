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


def visualize_tsne(datasets, n_limit, output_file, categories, model_type, apply_correction=False, use_background=False):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    plt.rcParams['legend.fontsize'] = 4
    if model_type == 'cleaned':
        accepted_classes = {
            '51:man,1511:young man,683:men,774:guy,1441:male': [0, 0, 0],
            '454:window,979:windows,1422:side window,537:front window,282:skylight,1586:panes': [128, 128, 128],
            '382:trees,292:tree,1185:pine trees,688:pine tree,1436:tree line': [255, 0, 0],
            '365:person,635:adult,949:worker,943:pedestrian': [128, 0, 0],
            '52:shirt,1404:tshirt,1404:t shirt,1404:t-shirt,1226:dress shirt,1099:tee shirt,1157:sweatshirt,653:undershirt,233:tank top,133:jersey,1288:blouse': [255, 255, 0],
            '178:building,670:buildings,581:skyscraper,1193:second floor': [128, 128, 0],
            '17:stop sign,17:stopsign,1437:sign post,941:traffic sign,589:street sign,817:signs,129:sign,245:stop': [0, 255, 0],
            '91:woman,749:women,858:lady,996:she,1486:ladies,1245:mother,1539:bride': [0, 128, 0],
            '1101:walls,249:wall,62:rock wall,1220:stone wall,1279:brick wall': [0, 255, 255],
            '612:dirt,466:ground,1272:soil,1476:pebbles,1477:mud': [0, 0, 255],
            '73:sky,1217:weather': [255, 0, 255],
        }
    else:
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
    # colors_per_class = {i: list(np.random.choice(range(256), size=3)) for i in range(1601)}

    c_examples = []
    c_correction = 0
    vector_features = []
    vector_classes = []
    for img_id, img_nbox, boxes_features, img_cls_ids in zip(datasets['image_id'], datasets['num_boxes'], datasets['features'], datasets['cls']):
        # continue just if the number of examples are less than "n_limit" and it is not a duplicate
        if len(c_examples) > n_limit:
            break
        else:
            if img_id in c_examples:
                continue
            else:
                c_examples.append(img_id)
        for i_proposal in range(img_nbox):
            proposal_features = boxes_features[i_proposal]
            proposal_class_id = img_cls_ids[i_proposal]
            proposal_class_label = categories[proposal_class_id]
            if proposal_class_label in accepted_classes.keys():  # filter to accept just the accepted_classed
                vector_features.append(proposal_features)
                vector_classes.append(proposal_class_id)
    vector_features = np.array(vector_features)
    vector_classes = np.array(vector_classes)
    print("Executing T-SNE.")
    model = TSNE(n_components=2, learning_rate='auto', init='random')
    embeddings = model.fit_transform(vector_features)

    print("Saving the plot.")
    # def scale_to_01_range(x):
    #     value_range = (np.max(x) - np.min(x))
    #     starts_from_zero = x - np.min(x)
    #     return starts_from_zero / value_range
    # tx = scale_to_01_range(embeddings[:, 0])
    # ty = scale_to_01_range(embeddings[:, 1])
    tx = embeddings[:, 0]
    ty = embeddings[:, 1]
    # for every class, we'll add a scatter plot separately
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for label_name in accepted_classes.keys():
        label_idx = list(categories.values()).index(label_name)
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(vector_classes) if l == label_idx]
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        # convert the class color to matplotlib format
        color = np.array(accepted_classes[label_name], dtype=float) / 255
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=[color], label=label_name.replace('_', ''), alpha=0.5, s=0.5)
    # build a legend using the labels we set previously
    ax.legend(loc='best')
    # finally, show the plot
    file_name = output_file
    plt.savefig(file_name, dpi=1000)


def parse_args():
    """
    Parse input arguments
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parsing
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--extracted_features', type=str, default='./extracted_features_new_classes_v3_VG/', help='Folder of extracted features')
    parser.add_argument('--output_folder', type=str, default='./proposals_features_t-sne.pdf', help='Folder where to save the output file.')
    parser.add_argument('--split_file', type=str, default='./datasets/cleaned_visual_genome/annotations/cleaned_visual_genome_val.json', help='Dataset.')
    parser.add_argument('--n_limit', type=int, default=1000)
    parser.add_argument('--labels', dest='labels',
                    help='File containing the new cleaned labels. It is needed for extracting the old and new classes indexes.',
                    default="./evaluation/objects_vocab_cleaned.txt",
                    type=str)
    parser.add_argument('--classes', dest='classes',
                help='Classes to consider.',
                default='all',
                choices=['all', 'untouched', 'new'],
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
        all_data = load_data(args.extracted_features, map_fn_reverse, args.classes, args.model, images_name)
        print("Start plotting")
        visualize_tsne(all_data, args.n_limit, args.output_folder, categories, args.model)
    else:
        print("Folder not valid: ", args.extracted_features)
        exit(1)
    
