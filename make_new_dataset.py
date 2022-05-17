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

def save_dataset(annotations_file, annotations):
    '''
    This function saves the new annotations.
    :param annotations_file: original name of the annotations file.
    :param annotations: new annotations.
    '''
    # save cleaned annotation file
    output_file_name = annotations_file.split('/')[-1]
    output_folder = "datasets/cleaned_visual_genome/annotations"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = "{}/cleaned_{}".format(output_folder, output_file_name)
    with open(output_file, 'w') as file:
        json.dump(annotations, file)
    print("Dataset saved in {} .".format(output_file))


def load_dataset(annotations_file):
    '''
    This function loads the Visual Genome annotations.
    :param annotations_file: annotations file.
    :return:
    '''
    print("Loading Visual Genome annotations in {} .".format(annotations_file))
    with open(annotations_file, 'r') as file:
        annotations = json.load(file)
    return annotations


def _update_dataset_info(annotations):
    '''
    This function updates the annotations information,
    :param annotations: annotations data.
    '''
    info = dict()
    info['description'] = "Visual Genome 2022 Dataset with cleaned classes."
    info['contributor'] = "Davide Rigoni, Stella Frank, Emanuele Bugliarello"
    info['date_created'] = "2022/05/05"
    annotations['info'] = info

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

def _update_class_labels(annotations, map_fn, cleaned_labels, old_labels):
    '''
    This function creates the new classes and check if it is all ok.
    :param annotations: the Visual Genome annotations.
    :param map_fn: the mapping function from the old to the new classes.
    :param cleaned_labels: the new cleaned classes.
    :param old_labels: the old classes.
    '''
    # first verify that the classes indexes are the same
    print('Checking indexes and classes.')
    annotations_classes = {ann['id']: ann['name'] for ann in annotations['categories']}     # [0, 1599]
    assert len(annotations_classes) == len(old_labels)
    # NOTE drigoni: our mapping start from index 1, while annotations from index 0.
    for label_id, label_str in annotations_classes.items():
        if old_labels[label_id+1] != label_str:
            print("Warning: label mismatch for {}:{} with {}. ".format(label_id, label_str, old_labels[label_id+1]))

    # make inverse mapping from annotations (because then we can check for errors)
    print('Checking supercategories.')
    map_fn_inv = defaultdict(list)     # [1, 1600]
    old_annotations = copy.deepcopy(annotations['categories'])
    for ann in old_annotations:
        ann['name'] = cleaned_labels[map_fn[ann['id']+1]]
        map_fn_inv[ann['name']].append(ann['id']+1)
    assert len(cleaned_labels) == len(map_fn_inv)

    # check if supercategories are the same among grouped classes otherwise print a warning
    annotations_supercategory = {ann['id']: ann['supercategory'] for ann in annotations['categories']}     # [0, 1599]
    cleaned_sup = defaultdict(list)
    for label_str, label_indexes in map_fn_inv.items():
        for label_id in label_indexes:
            cleaned_sup[label_str].append(annotations_supercategory[label_id-1])
    cleaned_supercategory_unique = {k: list(set(v)) for k, v in cleaned_sup.items()}     # [1, 1600]
    for k, v in cleaned_supercategory_unique.items():
        if len(v) > 1:
            print("Warning: supercategories mismatch among grouped classes {}:{}.".format(k, v))

    # change all the classes keeping the same supercategory. If more than one, the first is selected
    print('Generating new categories.')
    new_annotations = []     # [0, len(new_classes)-1]
    for label_id, label_str in cleaned_labels.items():
        tmp = {
            "id": label_id-1,
            "supercategory": cleaned_supercategory_unique[label_str][0],
            "name": label_str
        }
        new_annotations.append(tmp)
    assert len(new_annotations) == len(cleaned_labels) == len(map_fn_inv)
    # update
    annotations['categories'] = new_annotations


def _update_boxes_labels(annotations, map_fn, cleaned_labels, old_labels):
    '''
    This function updates the indexes of each bounding boxes according to the new classes.
    :param annotations: the Visual Genome annotations.
    :param map_fn: the mapping function from the old to the new classes.
    :param cleaned_labels: the new cleaned classes.
    :param old_labels: the old classes.
    '''
    print('Updating indexing for each GT bounding box.')
    for image in annotations['annotations']:
        # print(image)
        old_label_id = image['category_id']
        new_label_it = map_fn[old_label_id+1]
        # update class id
        image['category_id'] = new_label_it-1
        # print("Old_id is: {} with meaning: {} .".format(old_label_id, old_labels[old_label_id+1]))
        # print("new_id is {} with meaning: {} .".format(new_label_it, cleaned_labels[new_label_it]))


def create_new_dataset(annotations_file, map_fn, cleaned_labels, old_labels):
    '''
    This function update the classes and the link of each GT box.
    :param annotations_file: annotations file.
    :param map_fn: mapping function from the previusly classes to the cleaned ones.
    :param cleaned_labels: cleaned labels.
    :param old_labels: old labels.
    '''
    # read Visual Genome annotation file
    annotations = load_dataset(annotations_file)

    # dictionary
    # "info": {...},
    # "licenses": [...],
    # "images": [...],
    # "annotations": [...],
    # "categories": [...], <-- Not in Captions annotations
    # "segment_info": [...] <-- Only in Panoptic annotations

    # write new dataset information
    print("Processing. ")
    # update information about the dataset
    _update_dataset_info(annotations)
    # update labels of classes
    _update_class_labels(annotations, map_fn, cleaned_labels, old_labels)
    # update labels of boxes
    _update_boxes_labels(annotations, map_fn, cleaned_labels, old_labels)

    # save new dataset
    save_dataset(annotations_file, annotations)
    print("Dataset {} processed. ".format(annotations_file))


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
    parser.add_argument('--labels', dest='labels',
                        help='File containing the new cleaned labels.',
                        default="./evaluation/objects_vocab.txt",
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # create mapping function
    map_fn, cleaned_labels, old_labels = create_mapping(args.labels)
    create_new_dataset('datasets/visual_genome/annotations/visual_genome_train.json', map_fn, cleaned_labels, old_labels)
    create_new_dataset('datasets/visual_genome/annotations/visual_genome_val.json', map_fn, cleaned_labels, old_labels)
    create_new_dataset('datasets/visual_genome/annotations/visual_genome_test.json', map_fn, cleaned_labels, old_labels)
