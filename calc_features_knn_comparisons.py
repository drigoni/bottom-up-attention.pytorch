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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm



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


def load_data(img_folder, model_type, images_name):
    """
    Load extracted features fro the given folder.
    :param img_folder: folder where there are the extracted features
    :param model_type: indicates if the features are from a model trained on the old classes or on the new classes
    :param images_name: list of all the images to consider according to the dataset. Validation set of VG for example.
    """
    print("Model type: ", model_type)
    # get all extracted file in the folder
    onlyfiles = [join(img_folder, f) for f in listdir(img_folder) if isfile(join(img_folder, f))]
    print('Number of files: ', len(onlyfiles))
    onlyfiles = [f for f in onlyfiles if f[-4:] == '.npz']
    print('Number of .npz files: ', len(onlyfiles))
    # drigoni TODO: filter
    # onlyfiles = onlyfiles[:10000]


    # load all data ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    all_data = defaultdict(list)
    count_zeros = 0
    for img_file in tqdm(onlyfiles):
        img_id = img_file.split('/')[-1][:-4]

        # filter if image should not be in the folder according to the dataset .json file
        if img_id not in images_name:
            continue

        # load extracted features for each image
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
            # print(len(data_features[0]))    # 2048
            assert data_num_bbox == len(data_info['objects']) == len(data_info['cls_prob']) == len(data_boxes) == len(data_features)
            assert img_id not in all_data['image_id']

            # bounding boxes filtering according to its label 
            filtered_boxes = []
            filtered_features = []
            for box_idx in range(data_num_bbox):
                box_label_idx = data_info['objects'][box_idx]  # in [0, 877 or 1599]
                assert len(data_boxes[box_idx]) == 4
                # NOTE: BE SURE EVERYTHING IS np.float32 WHEN DEALING WITH base64.b64encode() function
                filtered_features.append(data_features[box_idx])
                filtered_boxes.append(data_boxes[box_idx])
            
            if len(filtered_boxes) == 0:
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
    print("Number of bounding boxes: ", sum(all_data['num_boxes']))
    return all_data


def prepare_features(all_data, categories):
    """
    This function prepare the features for the analysis.
    :param all_data: all noisy or clean data. A dict of list. categories_clean.keys() -> 'image_id', 'num_boxes', 'features', 'cls'
    :param categories: list of noisy or clean categories
    """
    # group features by class, REMEMBER empty list when there are no bounding boxes extracted for some classes
    features_per_class = {k: [] for k in range(len(categories))}    # index starting from 0
    features2img_per_class = {k: [] for k in range(len(categories))}    # index starting from 0
    for image_id, cls, features in zip(range(len(all_data['cls'])), all_data['cls'], all_data['features']):
        for c_index, f in zip(cls, features):
            assert len(f) == 2048
            features_per_class[c_index].append(f)
            features2img_per_class[c_index].append(image_id)
    return features_per_class, features2img_per_class


def knn_analysis_old(features, output_folder, type, n_neighbors=8):
    print("Start KNN_old fit with k={}.".format(n_neighbors))
    
    # group features by class, REMEMBER empty list when there are no bounding boxes extracted for some classes
    features_dict = {k: v for k, v in features.items()}    # index starting from 0
    X_data = []
    X_labels = []
    for k, v in features_dict.items():
        X_data.extend(v)
        X_labels.extend([k]*len(v))
    # X_labels, X_data =  zip(*features_dict.items())     # NOTE: X_data is a list of list of bounding boxes
    X_labels =  np.stack(X_labels, axis=0)
    X_data =  np.stack(X_data, axis=0)

    print("Start KNN predictions")
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', p=2)
    neigh.fit(X_data, X_labels)
    # OLD VERSION, WHICH IS BIASED
    # KNN_classes = list(neigh.classes_)
    # results = neigh.predict_proba(X_data)   # NOTE: IT IS BIASED. self inclusive!!!!
    # # each key corresponds to a class index and its value is a list of predictions.
    # # so: key -> [[....], [....]]. Inner lists refer to predictions per class
    # all_results_dict = defaultdict(list)
    # for prob, y_label in zip(results, X_labels):
    #     all_results_dict[y_label].append(prob)
    # # each key corresponds to a class index and its value represent the average value of proportions of NN belonging to the class
    # # NOTE: the predictor sees only a subset of all classes, just those where there is a predicted bounding boxes. So it is needed a map from the 1600 to the app. 1269 
    # results_dict = dict()
    # for k, predictions in all_results_dict.items():
    #     current_predictor_index = KNN_classes.index(k)
    #     tmp = [el[current_predictor_index] for el in predictions]
    #     score = sum(tmp)/(len(tmp)+1e-9)
    #     results_dict[int(k)] = round(score * 100, 2)

    # NEW VERSION, NOT BIASED
    neigh_dist, neigh_ind = neigh.kneighbors(X=None, n_neighbors=None, return_distance=True)  # [N, n_neighbors], [N, n_neighbors]. It is not self inclusive
    results_dict = defaultdict(list)
    for n, y_label in zip(neigh_ind, X_labels):
        neigh_labels = X_labels[n]
        neigh_labels_eq = (neigh_labels == y_label).astype(int)
        prob = sum(neigh_labels_eq)/(len(neigh_labels_eq)+1e-9)
        results_dict[y_label].append(prob)
    results_dict = {int(k): round(np.mean(v)*100, 2) for k, v in results_dict.items()}

    # DEBUGGING
    # for i, prob, label, dist, index in zip(range(len(results)), results, X_labels, neigh_dist, neigh_ind):
    #     current_predictor_index = KNN_classes.index(label)
    #     print('----')
    #     print('label GT:', label)
    #     print('pred cls index:', current_predictor_index)
    #     print('prob:', prob)
    #     print('prob>0:', [j for j, p in enumerate(prob) if p > 0])
    #     print('prob[label GT]:', prob[current_predictor_index])
    #     print('')
    #     print('dist:', dist)
    #     print('index:', index)
    #     print('index_ cls:', X_labels[index])
    #     print('----')
    #     if i > 2:
    #         exit(1)
    
    # dump distances
    output_file = "{}knn_euclidean_distance_nn{}_feat_{}.json".format(output_folder, n_neighbors, type)
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
        print('Saved file: {}'.format(output_file))


def knn_analysis(features, features2images, output_folder, model_type, ignore_points_from_same_image=False, micro_average=True):
    print("Start KNN analysis")
    print("Note: are features from the same image ignored?", ignore_points_from_same_image)
    # group features by class, REMEMBER empty list when there are no bounding boxes extracted for some classes
    features_dict = {int(k): v for k, v in features.items()}    # index starting from 0
    count_features = {int(k): len(v) for k, v in features2images.items()}    # index starting from 0
    assert len(features_dict.keys()) == 878 if model_type == 'cleaned' else len(features_dict.keys()) == 1600

    X_data = []
    X_labels = []
    X_images = []
    for k, v in features_dict.items():
        v2images = features2images[k]
        X_data.extend(v)
        X_labels.extend([k]*len(v))
        X_images.extend(v2images)
    # X_labels, X_data =  zip(*features_dict.items())     # NOTE: X_data is a list of list of bounding boxes
    X_data =  np.stack(X_data, axis=0)         # [N, features]
    X_labels =  np.stack(X_labels, axis=0)     # [N]  
    X_images =  np.stack(X_images, axis=0)     # [N]
    assert X_data.shape[0] == X_labels.shape[0] == X_images.shape[0]

    print("Start KNN predictions with n_neighbors=200.")
    neigh = KNeighborsClassifier(n_neighbors=200, weights='uniform', p=2)
    neigh.fit(X_data, X_labels)
    # results = neigh.predict_proba(X_data)
    neigh_dist, neigh_ind = neigh.kneighbors(X=None, n_neighbors=None, return_distance=True)  # [N, n_neighbors], [N, n_neighbors]. It is not self inclusive
    assert neigh_dist.shape[0] == X_data.shape[0]

    results_1 = {k: [] for k, v in features.items()}
    results_5 = {k: [] for k, v in features.items()}
    results_10 = {k: [] for k, v in features.items()}
    results_100 = {k: [] for k, v in features.items()}
    for el in range(neigh_dist.shape[0]):
        # current point data
        el_data = X_data[el, :]     # [2048]    
        el_label = X_labels[el]     # ()
        el_image = X_images[el]     # ()

        # all neighbors indexes and distances
        neig_indexes = neigh_ind[el, :]         # [n_neighbors]
        neig_distances = neigh_dist[el, :]      # [n_neighbors]
        # sort neigh according to their distances. Less the distance, better it is
        neigh_ordered_distances = np.argsort(neig_distances)  # descending order. # NOTE: it should be already ordered
        neig_distances = neig_distances[neigh_ordered_distances]
        neig_indexes = neig_indexes[neigh_ordered_distances]
        assert len(neig_indexes) > 0
        assert len(neig_indexes) == len(neig_distances)

        # retrieve indexes images and filter them
        if ignore_points_from_same_image:
            neig_images = X_images[neig_indexes]
            neig_images_boolean = (neig_images != el_image)
            neig_available = neig_indexes[neig_images_boolean]    
        else:
            neig_available = neig_indexes
        assert len(neig_available) >= 100

        # calculate proportion
        for value, res_storage in zip([1, 5, 10, 100], [results_1, results_5, results_10, results_100]):
            neig_to_consider = neig_available[:value]
            neig_classes = X_labels[neig_to_consider]
            tmp_hits = (neig_classes == el_label).astype(int)
            tmp_mean = np.mean(tmp_hits)
            # store proportion 
            res_storage[el_label].append(tmp_mean)
    assert len(results_1.keys()) == len(features_dict.keys())

    # get untouched classes
    if model_type == 'noisy':
        untouched_cls_idx = [v[0]-1 for k, v in map_fn_reverse.items() if len(v) == 1]
    elif model_type == 'cleaned':
        untouched_cls_idx = [k-1  for k, v in map_fn_reverse.items() if len(v) == 1]
    # assert len(untouched_cls_idx) == 515 # number of untouched classes

    
    # weighted average
    for value, res_storage in zip([1, 5, 10, 100], [results_1, results_5, results_10, results_100]):
        for class_type in ['all', 'untouched', 'merged']:
            means_to_consider = []
            count_to_consider = []
            for k, v in res_storage.items():
                if len(v) > 0:
                    if class_type == 'all':
                        means_to_consider.append(np.mean(v))
                        count_to_consider.append(len(v))
                    elif class_type == 'untouched':
                        if k in untouched_cls_idx:
                            means_to_consider.append(np.mean(v))
                            count_to_consider.append(len(v))  
                    else:
                        if k not in untouched_cls_idx:
                            means_to_consider.append(np.mean(v))
                            count_to_consider.append(len(v)) 
            # tmp_mean = round(np.mean([i*j for i, j in zip(means_to_consider, count_to_consider)]), 3)
            if micro_average:
                tmp_mean = np.average(means_to_consider, weights=count_to_consider)
                tmp_std = math.sqrt(np.average((means_to_consider-tmp_mean)**2, weights=count_to_consider))
            else:
                tmp_mean = np.average(means_to_consider)
                tmp_std = math.sqrt(np.average((means_to_consider-tmp_mean)**2))
            tmp_mean = round(tmp_mean * 100, 2)
            tmp_std = round(tmp_std * 100, 2)
            print("Proportion of k={} NNs that share the right class with {} classes. Aggregation:{} || Mean: {} || STD: {} . ".format(value, model_type, class_type, tmp_mean, tmp_std ))
        print("--")


def parse_args():
    """
    Parse input arguments
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parsing
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--features_noisy', type=str, default='./extracted_features/extracted_features_develop_VG/', help='Folder of extracted features')
    parser.add_argument('--features_clean', type=str, default='./extracted_features/extracted_features_new_classes_v3_VG/', help='Folder of extracted features')
    parser.add_argument('--output_folder', type=str, default='./analysis/knn/', help='Folder where to save the output file.')
    parser.add_argument('--split_file_noisy', type=str, default='./datasets/visual_genome/annotations/visual_genome_val.json', help='Dataset.')
    parser.add_argument('--split_file_clean', type=str, default='./datasets/cleaned_visual_genome/annotations/cleaned_visual_genome_val.json', help='Dataset.')
    parser.add_argument('--labels', dest='labels',
                    help='File containing the new cleaned labels. It is needed for extracting the old and new classes indexes.',
                    default="./evaluation/objects_vocab_cleaned.txt",
                    type=str)
    parser.add_argument('--classes', dest='classes',
                help='Classes to consider.',
                default='all',
                choices=['all', 'untouched', 'new'],
                type=str)
    parser.add_argument('--analysis', dest='analysis',
            help='Analysis to perform.',
            default='knn',
            choices=['knn', 'knn_old'],
            type=str)
    parser.add_argument('--k', dest='k',
            help='KNN parameter k',
            default=10,
            type=int)
    parser.add_argument('--ignore', dest='ignore',
                        help='True to ignore features from same image.',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # get labels and mapping function
    map_fn, cleaned_labels, old_labels = create_mapping(args.labels)
    map_fn_reverse = defaultdict(list)  # from clean classes to noisy classes
    for k, v in map_fn.items():
        map_fn_reverse[v].append(k)

    # get images names to use for loading the extracted features
    with open(args.split_file_noisy, 'r') as f:
        dataset = json.load(f)
    images_name = [i['file_name'][:-4] for i in dataset['images']]
    categories_noisy = {int(i['id']): i['name']  for i in dataset['categories']}
    with open(args.split_file_clean, 'r') as f:
        dataset = json.load(f)
    images_name = [i['file_name'][:-4] for i in dataset['images']]
    categories_clean = {int(i['id']): i['name']  for i in dataset['categories']}
    print("Number of images to consider:", len(images_name))

    # check if the folder exists
    if os.path.exists(args.features_noisy) and os.path.exists(args.features_clean):
        print('Loading all data.')
        all_data_noisy = load_data(args.features_noisy, "noisy", images_name)
        all_data_clean = load_data(args.features_clean, "cleaned", images_name)
        features_per_class_noisy, features_per_class_noisy2images = prepare_features(all_data_noisy, categories_noisy)
        features_per_class_clean, features_per_class_clean2images = prepare_features(all_data_clean, categories_clean)
        print("Start calculation")
        if args.analysis == 'knn_old':
            knn_analysis_old(features_per_class_noisy, args.output_folder, 'noisy', n_neighbors=args.k)
            knn_analysis_old(features_per_class_clean, args.output_folder, 'cleaned', n_neighbors=args.k)
        elif args.analysis == 'knn':
            knn_analysis(features_per_class_noisy, features_per_class_noisy2images, args.output_folder, 'noisy', args.ignore)
            knn_analysis(features_per_class_clean, features_per_class_clean2images, args.output_folder, 'cleaned', args.ignore)
    else:
        print("Folder not valid. ")
        exit(1)
    
