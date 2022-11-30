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


def load_data(img_folder, map_fn_reverse, classes_type, model_type, images_name):
    """
    Load extracted features fro the given folder.
    :param img_folder: folder where there are the extracted features
    :param map_fn_reverse: dict {CLASS: list of old classes}. CLASS values are in [1, 878].
    :param classes_type: indicates the type of classes selected among ['all', 'untouched', 'new']
    :param model_type: indicates if the features are from a model trained on the old classes or on the new classes
    :param images_name: list of all the images to consider according to the dataset. Validation set of VG for example.
    """
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
    # NOTE: from here, untouched_cls_idx is a list of index representing the indexes of noisy/cleaned classes that where never merged in the process


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
                # filtered_boxes.append(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
                # filtered_features.append(data_features[0])
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


def prepare_features(all_data_noisy, all_data_clean, categories_noisy, categories_clean):
    """
    This function prepare the features for the analysis.
    :param all_data_noisy: all noisy data. A dict of list. categories_clean.keys() -> 'image_id', 'num_boxes', 'features', 'cls'
    :param all_data_clean: all cleaned data
    :param categories_noisy: list of noisy categories
    :param categories_clean: list of clean categories
    """
    # group features by class, REMEMBER empty list when there are no bounding boxes extracted for some classes
    features_per_class_noisy = {k: [] for k in range(len(categories_noisy))}    # index starting from 0
    features_per_class_clean = {k: [] for k in range(len(categories_clean))}    # index starting from 0
    for cls, features in zip(all_data_noisy['cls'], all_data_noisy['features']):
        for c_index, f in zip(cls, features):
            # cls_label = categories_noisy[c_index]
            # features_per_class_noisy[cls_label].append(f)
            assert len(f) == 2048
            features_per_class_noisy[c_index].append(f)
    for cls, features in zip(all_data_clean['cls'], all_data_clean['features']):
        for c_index, f in zip(cls, features):
            # cls_label = categories_clean[c_index]
            # features_per_class_clean[cls_label].append(f)
            assert len(f) == 2048
            features_per_class_clean[c_index].append(f)  
    return features_per_class_noisy, features_per_class_clean


def knn_analysis(features, output_folder, type, n_neighbors=8):
    print("Start KNN fit with k={}.".format(n_neighbors))
    from sklearn.neighbors import KNeighborsClassifier
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
    KNN_classes = list(neigh.classes_)
    results = neigh.predict_proba(X_data)

    # each key corresponds to a class index and its value is a list of predictions.
    # so: key -> [[....], [....]]. Inner lists refer to predictions per class
    all_results_dict = defaultdict(list)
    for prob, y_label in zip(results, X_labels):
        all_results_dict[y_label].append(prob)
    
    # each key corresponds to a class index and its value represent the average value of proportions of NN belonging to the class
    # NOTE: the predictor sees only a subset of all classes, just those where there is a predicted bounding boxes. So it is needed a map from the 1600 to the app. 1269 
    results_dict = dict()
    for k, predictions in all_results_dict.items():
        current_predictor_index = KNN_classes.index(k)
        tmp = [el[current_predictor_index] for el in predictions]
        score = sum(tmp)/(len(tmp)+1e-9)
        results_dict[int(k)] = round(score * 100, 2)

    print(results_dict)

    # dump distances
    output_file = "{}knn_euclidean_distance_nn{}_feat_{}.json".format(output_folder, n_neighbors, type)
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
        print('Saved file: {}'.format(output_file))


# version more specific
def intra_class_pairs_analysis(features_per_class_noisy, features_per_class_clean, map_fn, output_folder):
    print("Intra class analysis")
    clusters_clean = {k: v for k, v in features_per_class_clean.items()}    # index starting from 0
    clusters_noisy = defaultdict(list) 
    for k, v in features_per_class_noisy.items():
        clean_index = map_fn[k+1]-1
        clusters_noisy[clean_index].append(v)                               # NOTE: list of list, one for each noisy class
    
    # clusters_clean -> dict, for each key there is a list of features or an empty list
    # clusters_noisy -> dict, for each key there is a list of list of features or an empty list


    # calculate cosine_similarity for each class to each class. No self
    distances_clean = dict()
    distances_noisy = defaultdict(list)
    for k in clusters_clean.keys():
        if len(clusters_clean[k]) > 0:
            feat_clean = np.stack(clusters_clean[k], axis=0)
            distances_clean[k] = cosine_distances(feat_clean, feat_clean)
            np.fill_diagonal(distances_clean[k], 0)        # DIAGONAL VALUES TO 0, inplace operation
        else: 
            distances_clean[k] = None
        feat_list_noisy = clusters_noisy[k]
        n_elements = len(feat_list_noisy)
        for i in range(n_elements):
            for j in range(n_elements):
                if i != j and len(feat_list_noisy[i]) > 0 and len(feat_list_noisy[j]) > 0 :
                    feat_list_noisy_i = np.stack(feat_list_noisy[i], axis=0)
                    feat_list_noisy_j = np.stack(feat_list_noisy[j], axis=0)
                    distances_noisy[k].append(cosine_distances(feat_list_noisy_i, feat_list_noisy_j))    
    
    # distances_clean -> dict, for each key there is a matrix of the distances or a NONE value
    # distances_noisy -> dict, for each key there is a list of matrix of the distances or an empty list
    
    results_clean = dict()
    results_noisy = dict()
    for k in distances_clean.keys():
        if distances_clean[k] is not None and len(distances_clean[k]) > 0:
            # remember that diagonal is 0
            res_clean = np.sum(distances_clean[k], axis=1) / (len(distances_clean[k])-1)    # [n, n] to [n]
            res_clean = float(np.mean(res_clean))
        else:
            res_clean = None
        results_clean[k] = res_clean
        # remember the different number of elements in each class
        res_noisy = [np.mean(list_distances) for list_distances in distances_noisy[k] if len(list_distances) > 0]  # NOTE: number of element not considered... macro average
        if len(res_noisy) > 0:
            res_noisy = float(np.mean(res_noisy))
        else: 
            res_noisy = None
        results_noisy[k] = res_noisy

    # dump distances
    output_file = "{}features_cleaned_intra_class_pairs_distance.json".format(output_folder)
    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
        print('Saved file: {}'.format(output_file))
    output_file = "{}features_noisy_intra_class_pairs_distance.json".format(output_folder)
    with open(output_file, 'w') as f:
        json.dump(results_noisy, f, indent=2)
        print('Saved file: {}'.format(output_file))


def intra_class_by_cleaned_classes_analysis(features_per_class_noisy, features_per_class_clean, map_fn, output_folder):
    print("Intra class analysis")
    clusters_clean = {k: v for k, v in features_per_class_clean.items()}    # index starting from 0
    clusters_noisy = defaultdict(list) 
    for k, v in features_per_class_noisy.items():
        clean_index = map_fn[k+1]-1
        clusters_noisy[clean_index].extend(v)                               # NOTE: list of list, one for each noisy class
    
    # clusters_clean -> dict, for each key there is a list of features or an empty list
    # clusters_noisy -> dict, for each key there is a list of list of features or an empty list


    # calculate cosine_similarity for each class to each class. No self
    distances_clean = dict()
    distances_noisy = dict()
    for k in clusters_clean.keys():
        if len(clusters_clean[k]) > 0:
            feat_clean = np.stack(clusters_clean[k], axis=0)
            distances_clean[k] = euclidean_distances(feat_clean, feat_clean)
            np.fill_diagonal(distances_clean[k], 0)        # DIAGONAL VALUES TO 0, inplace operation
        else: 
            distances_clean[k] = None

        if len(clusters_noisy[k]) > 0:
            feat_clean = np.stack(clusters_noisy[k], axis=0)
            distances_noisy[k] = euclidean_distances(feat_clean, feat_clean)
            np.fill_diagonal(distances_noisy[k], 0)        # DIAGONAL VALUES TO 0, inplace operation
        else: 
            distances_noisy[k] = None
    
    results_clean = dict()
    results_noisy = dict()
    for k in distances_clean.keys():
        if distances_clean[k] is not None and len(distances_clean[k]) > 1:  # 1 because if there is only one point, its distance  from itself is not interesting 
            # remember that diagonal is 0
            res_clean = np.sum(distances_clean[k], axis=1) / (len(distances_clean[k])-1)    # [n, n] to [n]
            res_clean = float(np.mean(res_clean))
        else:
            res_clean = None
        results_clean[k] = res_clean
        # remember the different number of elements in each class
        if distances_noisy[k] is not None and len(distances_noisy[k]) > 1:
            # remember that diagonal is 0
            res_noisy = np.sum(distances_noisy[k], axis=1) / (len(distances_noisy[k])-1)    # [n, n] to [n]
            res_noisy = float(np.mean(res_noisy))
        else:
            res_noisy = None
        results_noisy[k] = res_noisy

    # dump distances
    output_file = "{}features_cleaned_intra_class_distance.json".format(output_folder)
    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
        print('Saved file: {}'.format(output_file))
    output_file = "{}features_noisy_intra_class_distance.json".format(output_folder)
    with open(output_file, 'w') as f:
        json.dump(results_noisy, f, indent=2)
        print('Saved file: {}'.format(output_file))



def intra_class_analysis(features_per_class_noisy, features_per_class_clean, map_fn, output_folder):
    print("Intra class analysis")
    clusters_clean = {k: v for k, v in features_per_class_clean.items()}    # index starting from 0
    clusters_noisy = {k: v for k, v in features_per_class_noisy.items()}    # index starting from 0
    
    # clusters_clean -> dict, for each key there is a list of features or an empty list
    # clusters_noisy -> dict, for each key there is a list of list of features or an empty list


    # calculate cosine_similarity for each class to each class. No self
    distances_clean = dict()
    for k in clusters_clean.keys():
        if len(clusters_clean[k]) > 0:
            feat_clean = np.stack(clusters_clean[k], axis=0)
            distances_clean[k] = euclidean_distances(feat_clean, feat_clean)
            np.fill_diagonal(distances_clean[k], 0)        # DIAGONAL VALUES TO 0, inplace operation
        else: 
            distances_clean[k] = None
    distances_noisy = dict()
    for k in clusters_noisy.keys():
        if len(clusters_noisy[k]) > 0:
            feat_noisy = np.stack(clusters_noisy[k], axis=0)
            distances_noisy[k] = euclidean_distances(feat_noisy, feat_noisy)
            np.fill_diagonal(distances_noisy[k], 0)        # DIAGONAL VALUES TO 0, inplace operation
        else: 
            distances_noisy[k] = None

    # calculate mean
    results_clean = dict()
    for k in distances_clean.keys():
        if distances_clean[k] is not None and len(distances_clean[k]) > 1:  # 1 because if there is only one point, its distance  from itself is not interesting 
            # remember that diagonal is 0
            res_clean = np.sum(distances_clean[k], axis=1) / (len(distances_clean[k])-1)    # [n, n] to [n]
            res_clean = float(np.mean(res_clean))
        else:
            res_clean = None
        results_clean[k] = res_clean
    results_noisy = dict()
    for k in distances_noisy.keys():
        if distances_noisy[k] is not None and len(distances_noisy[k]) > 1:  # 1 because if there is only one point, its distance  from itself is not interesting 
            # remember that diagonal is 0
            res_noisy = np.sum(distances_noisy[k], axis=1) / (len(distances_noisy[k])-1)    # [n, n] to [n]
            res_noisy = float(np.mean(res_noisy))
        else:
            res_noisy = None
        results_noisy[k] = res_noisy

    # filter and calculate results
    results_clean = {k: v for k, v in results_clean.items() if v is not None}
    results_noisy = {k: v for k, v in results_noisy.items() if v is not None}
    for name, data in zip(['noisy', 'cleaned'], [results_noisy, results_clean]):
        tmp_mean = round(np.mean(list(data.values())), 3)
        tmp_std = round(np.std(list(data.values())), 3)
        print("Average intra distances {} classes. Mean: {} || STD: {} . ".format(name, tmp_mean, tmp_std ))



def inter_class_analysis(features_per_class_noisy, features_per_class_clean, map_fn, output_folder):
    print("Inter class analysis")
    clusters_clean = {k: np.mean(np.stack(v, axis=0), axis=0) if len(v)>0 else None for k, v in features_per_class_clean.items()}    # index starting from 0
    clusters_noisy = {k: np.mean(np.stack(v, axis=0), axis=0) if len(v)>0 else None for k, v in features_per_class_noisy.items()}    # index starting from 0

    # calculate mean
    results_clean = dict()
    for k in clusters_clean.keys():
        if clusters_clean[k] is not None:
            # remove self element
            tmp_copy = copy.deepcopy(clusters_clean)
            tmp_copy.pop(k)
            other_centroids = [v for v in tmp_copy.values() if v is not None]
            res_clean = euclidean_distances([clusters_clean[k]], np.stack(other_centroids, axis=0))
            res_clean = float(np.mean(res_clean))
        else:
            res_clean = None
        results_clean[k] = res_clean
    results_noisy = dict()
    for k in clusters_noisy.keys():
        if clusters_noisy[k] is not None:
            # remove self element
            tmp_copy = copy.deepcopy(clusters_noisy)
            tmp_copy.pop(k)
            other_centroids = [v for v in tmp_copy.values() if v is not None]
            res_noisy = euclidean_distances([clusters_noisy[k]], np.stack(other_centroids, axis=0))
            res_noisy = float(np.mean(res_noisy))
        else:
            res_noisy = None
        results_noisy[k] = res_noisy

    # # dump distances
    # output_file = "{}features_cleaned_inter_class_distance.json".format(output_folder)
    # with open(output_file, 'w') as f:
    #     json.dump(results_clean, f, indent=2)
    #     print('Saved file: {}'.format(output_file))
    # output_file = "{}features_noisy_inter_class_distance.json".format(output_folder)
    # with open(output_file, 'w') as f:
    #     json.dump(results_noisy, f, indent=2)
    #     print('Saved file: {}'.format(output_file))
    # filter and calculate results
    results_clean = {k: v for k, v in results_clean.items() if v is not None}
    results_noisy = {k: v for k, v in results_noisy.items() if v is not None}
    for name, data in zip(['noisy', 'cleaned'], [results_noisy, results_clean]):
        tmp_mean = round(np.mean(list(data.values())), 3)
        tmp_std = round(np.std(list(data.values())), 3)
        print("Average inter distances {} classes. Mean: {} || STD: {} . ".format(name, tmp_mean, tmp_std ))

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
                    default="./evaluation/objects_vocab.txt",
                    type=str)
    parser.add_argument('--classes', dest='classes',
                help='Classes to consider.',
                default='all',
                choices=['all', 'untouched', 'new'],
                type=str)
    parser.add_argument('--analysis', dest='analysis',
            help='Analysis to perform.',
            default='knn',
            choices=['knn', 'intra_distance', 'intra_distance_clean', 'intra_distance_pairs', 'inter_distance'],
            type=str)
    parser.add_argument('--k', dest='k',
        help='KNN value for k parameter',
        default='10',
        type=int)
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

    # check if the folder exists
    if os.path.exists(args.features_noisy) and os.path.exists(args.features_clean):
        print('Loading all data.')
        all_data_noisy = load_data(args.features_noisy, map_fn_reverse, args.classes, "noisy", images_name)
        all_data_clean = load_data(args.features_clean, map_fn_reverse, args.classes, "cleaned", images_name)
        features_per_class_noisy, features_per_class_clean = prepare_features(all_data_noisy, all_data_clean, categories_noisy, categories_clean)
        print("Start calculation")
        if args.analysis == 'knn':
            knn_analysis(features_per_class_noisy, args.output_folder, 'noisy', n_neighbors=args.k)
            knn_analysis(features_per_class_clean, args.output_folder, 'cleaned', n_neighbors=args.k)
        elif args.analysis == 'inter_distance':
            inter_class_analysis(features_per_class_noisy, features_per_class_clean, map_fn, args.output_folder)
        elif args.analysis == 'intra_distance':
            intra_class_analysis(features_per_class_noisy, features_per_class_clean, map_fn, args.output_folder)
        elif args.analysis == 'intra_distance_clean':
            intra_class_by_cleaned_classes_analysis(features_per_class_noisy, features_per_class_clean, map_fn, args.output_folder)
        elif args.analysis == 'intra_distance_pairs':
            intra_class_pairs_analysis(features_per_class_noisy, features_per_class_clean, map_fn, args.output_folder)
    else:
        print("Folder not valid. ")
        exit(1)
    
