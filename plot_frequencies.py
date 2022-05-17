#!/usr/bin/env python
"""
Created on 5/05/22
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file includes the code needed plotting scores.
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


def draw_plots_together(counting1, counting2, output):
    # plot first dictionary
    counting1_sorted = dict(sorted(counting1.items(), key=lambda i: float(i[0]), reverse=False))
    plt.plot(counting1_sorted.keys(), counting1_sorted.values())
    # plot second dictionary
    counting2_sorted = dict(sorted(counting2.items(), key=lambda i: float(i[0]), reverse=False))
    plt.plot(counting2_sorted.keys(), counting2_sorted.values())
    plt.title("AP scores by number of GT boxes")
    ax = plt.gca()
    ax.set_xlabel('Number of GT boxes')        
    ax.set_ylabel('AP scores')
    ax.legend(['post-processing', 'Cleaned classes'])
    plt.savefig(output)  
    print('Saved plot: {}'.format(output))

def draw_loglog_plots_together(counting1, counting2, output):
    # plot first dictionary
    counting1_sorted = dict(sorted(counting1.items(), key=lambda i: float(i[0]), reverse=False))
    plt.loglog(counting1_sorted.keys(), counting1_sorted.values(), base=10)
    # plot second dictionary
    counting2_sorted = dict(sorted(counting2.items(), key=lambda i: float(i[0]), reverse=False))
    plt.loglog(counting2_sorted.keys(), counting2_sorted.values(), base=10)
    plt.title("AP scores by number of GT boxes")
    ax = plt.gca()
    ax.set_xlabel('log(Number of GT boxes)')        
    ax.set_ylabel('log(AP scores)')
    ax.legend(['post-processing', 'Cleaned classes'])
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
                    help='Frequency file.',
                    default=None,
                    type=str)
    parser.add_argument('--file2', dest='file2',
                        help='Frequency file.',
                        default=None,
                        type=str)
    parser.add_argument('--output', dest='output',
                        help='Dataset file.',
                        default="./aps_scores.pdf",
                        type=str)
    parser.set_defaults(loglog=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.file1, 'r') as f:
        counting1 = json.load(f)
        # load only AP scores, not wAP scores
        counting1 = {k: v[0] for k, v in counting1.items()}
    with open(args.file2, 'r') as f:
        counting2 = json.load(f)
        # load only AP scores, not wAP scores
        counting2 = {k: v[0] for k, v in counting2.items()}
    # plots together the frequencies reported in two files
    if args.loglog is False:
        draw_plots_together(counting1, counting2, args.output)
    else:
        draw_loglog_plots_together(counting1, counting2, args.output)