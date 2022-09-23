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

def filter_classes(all_data, clean_cls_idx, classes_type):
    accepted_classes_similar = {
        "1": "1:yolk,525:egg,324:eggs",
        "3": "3:bathroom,1574:restroom",
        "7": "11:tail fin,1468:fin",
        "9": "15:toilet,85:toilet seat,302:toilet tank,385:toilet bowl,444:toilet lid",
        "12": "17:stop sign,17:stopsign,1437:sign post,941:traffic sign,589:street sign,817:signs,129:sign,245:stop",
        "14": "18:cone,576:cones,560:traffic cone,658:safety cone",
        "19": "25:kettle,67:tea kettle",
        "23": "29:bathtub,196:bath tub,306:tub",
        "24": "1168:blind,30:blinds",
        "25": "31:court,39:tennis court",
        "26": "314:urinals,32:urinal",
        "27": "34:bed,893:beds,947:bedding,660:bedspread,1343:bed frame",
        "29": "36:giraffe,38:giraffes,471:giraffe head",
        "31": "1229:laptops,41:laptop,1124:laptop computer",
        "32": "42:tea pot,562:teapot",
        "33": "43:horse,187:horses,1319:pony",
        "35": "1351:short,45:shorts",
        # "36": "46:manhole,1014:manhole cover",
        # "37": "47:dishwasher,148:washer",
        # "40": "51:man,1511:young man,683:men,774:guy,1441:male",
        # "41": "52:shirt,1404:tshirt,1404:t shirt,1404:t-shirt,1226:dress shirt,1099:tee shirt,1157:sweatshirt,653:undershirt,233:tank top,133:jersey,1288:blouse",
        # "42": "686:cars,53:car,955:passenger car,1334:sedan",
        # "44": "54:cat,185:cats,477:kitten,1117:kitty",
        # "46": "56:bus,380:buses",
        # "47": "57:radiator,1006:heater",
        # "53": "61:plate,956:plates,1378:paper plate,540:saucer,587:dishes,788:dish",
        # "54": "65:ocean,1214:sea",
        # "57": "1587:shoreline,816:shore",
        # "60": "68:wetsuit,217:wet suit",
        # "62": "70:sink,692:sinks,1123:bathroom sink,1424:basin",
        # "63": "815:trains,71:train,1448:passenger train,899:train front,626:train car,1182:train cars,490:carriage,637:locomotive,1275:caboose,1318:railroad",
        # "64": "73:sky,1217:weather",
        # "66": "75:train station,272:train platform,319:platform,387:station",
        # "68": "77:bats,301:bat,657:baseball bat",
        # "72": "142:elephants,84:elephant",
        # "73": "86:zebra,88:zebras",
        # "74": "87:skateboard,87:skate board,1224:skateboards",
        # "76": "91:woman,749:women,858:lady,996:she,1486:ladies,1245:mother,1539:bride",
        # "78": "685:bicycles,94:bicycle,506:bikes,100:bike",
        # "79": "95:magazines,1096:magazine",
        # "81": "495:umbrellas,97:umbrella,1523:parasol",
        # "82": "151:cows,98:cow,428:bull,793:cattle,583:ox,1202:calf",
        # "88": "1204:grapes,105:grape",
        # "90": "107:table,1301:tables,875:end table,200:coffee table",
        # "93": "110:orange,408:oranges",
        # "94": "219:teddy bears,111:teddy bear,1293:teddy,270:stuffed animals,767:stuffed animal,647:stuffed bear",
        # "95": "113:meter,1481:meters,211:parking meter",
        # "97": "262:ski boots,117:ski boot",
        # "98": "118:dog,338:dogs,1532:puppy",
        # "101": "120:hair,505:mane,1187:bangs",
        # "106": "128:mouse,486:computer mouse",
        # "107": "134:reigns,574:bridle,24:halter,1388:harness",
        # "108": "1321:hot dogs,1321:hotdogs,135:hot dog,135:hotdog,1384:sausage",
        # "109": "136:surfboard,136:surf board,351:surfboards",
        # "110": "163:glasses,138:glass",
        # "111": "1493:wine glasses,614:wine glass",
        # "112": "625:sunglasses,990:eye glasses,800:eyeglasses",
        # "113": "1327:shades,620:shade",
        # "114": "1139:snow board,139:snowboard",
        # "115": "140:girl,754:girls,953:little girl",
        # "120": "344:bears,147:bear,131:polar bear,283:cub",
        # "122": "150:bow tie,578:necktie,655:neck tie,268:tie",
        # "126": "156:truck,839:trucks",
        # "130": "162:sheep,164:ram,231:lamb",
        # "131": "705:kites,165:kite",
        # "132": "166:salad,868:lettuce,1398:greens",
        # "133": "167:pillow,332:pillows,842:pillow case,675:throw pillow",
        # "135": "169:mug,232:cup,850:coffee cup",
        # "137": "171:computer,1032:computers,1053:cpu",
        # "138": "172:swimsuit,1174:swim trunks,388:bikini,1008:bathing suit",
        # "139": "173:tomato,665:tomatoes,426:tomato slice",
        # "140": "174:tire,1456:tires",
        # "144": "1581:sandwiches,179:sandwich,1052:sandwhich",
        # "145": "180:weather vane,753:vane",
        # "146": "181:bird,1000:birds",
        # "147": "182:jacket,381:coat,1521:ski jacket,566:suit jacket,836:blazer",
        # "149": "184:water,1429:ocean water",
        # "153": "191:cake,12:birthday cake,273:cupcake,764:frosting",
        # "155": "193:head band,368:headband",
        # "156": "780:skiers,194:skier,1009:skiier",
        # "158": "197:bowl,1027:bowls",
        # "160": "1241:floors,201:floor,1556:tile floor,1310:flooring",
        # "161": "519:uniforms,202:uniform",
        # "162": "203:ottoman,424:sofa,137:couch,228:armchair",
        # "164": "205:olive,1148:olives",
        # "165": "206:mound,459:pitcher's mound",
        # "167": "208:food,703:meal",
        # "168": "209:paintings,346:painting",
        # "169": "210:traffic light,1347:traffic lights",
        # "170": "212:bananas,531:banana,554:banana peel,464:banana bunch,266:banana slice",
        # "172": "213:mountain,457:mountains,1304:mountain top,984:mountain range,1487:peak,1375:mountainside",
        # "176": "221:suitcase,221:suit case,429:suitcases,297:luggage",
        # "177": "507:drawer,222:drawers",
        # "178": "1069:grasses,223:grass,488:lawn,963:turf",
        # "179": "101:field,1286:grass field,418:pasture",
        # "181": "289:apples,224:apple",
        # "182": "226:goggles,1246:ski goggles",
        # "183": "510:boys,227:boy",
        # "185": "269:burners,230:burner",
        # "192": "241:bottle,379:bottles,1554:beer bottle,931:wine bottle,476:water bottle",
        # "193": "1267:surfers,244:surfer",
        # "194": "1203:back pack,246:backpack",
        # "196": "247:shin guard,876:shin guards",
        # "197": "248:wii remote,432:remotes,805:remote,348:remote control,723:controller,812:game controller,1208:controls,1589:control,1303:wii",
        # "199": "250:pizza slice,127:pizza,914:pizzas",
        # "200": "1466:slices,1005:slice",
    }

    accepted_classes_diverse = {
        "10": "16:batter,5:umpire,14:catcher,474:baseball player,1210:baseball players,794:players,92:player,78:tennis player,377:soccer player,207:pitcher",
        "39": "125:parasail,1569:parachute",
        "75": "89:floor lamp,1426:table lamp,1083:lamps,225:lamp,161:chandelier,905:light fixture",
        "84": "99:pants,1492:pant,781:trouser,1111:sweatpants,973:jean,48:jeans,651:snow pants,503:ski pants,1344:slacks",
        "99": "119:clock,1393:clocks,7:alarm clock,1274:clock hand,509:clock face",
        "116": "141:plane,532:planes,489:airplanes,132:airplane,536:aircraft,803:jets,545:jet",
        "117": "143:oven,679:oven door,198:stove",
        "119": "146:area rug,335:rug,467:carpet",
        "127": "158:boat,234:boats,59:sailboat,59:sail boat,421:ship,719:yacht,988:canoe,1143:kayak",
        "136": "170:tarmac,1495:asphalt,831:pavement",
        "143": "178:building,670:buildings,581:skyscraper,1193:second floor",
        "148": "183:chair,699:chairs,552:office chair,390:lounge chair,157:beach chair,504:seat,1022:seats,242:stool,1015:stools,1325:recliner",
        "150": "186:soccer ball,1235:balls,568:ball,481:tennis ball,674:baseball",
        "152": "190:engine,619:engines,567:train engine,1093:jet engine",
        "180": "667:soccer field,114:baseball field,763:infield,729:outfield,22:dugout",
        "186": "235:hat,798:cowboy hat,487:cap,721:baseball cap,1153:beanie,1149:ball cap",
        "198": "1101:walls,249:wall,62:rock wall,1220:stone wall,1279:brick wall",
    }
    all_data_filtered = {k: v for k, v in all_data.items() if int(k) in [int(i) for i in accepted_classes_similar.keys()]}
    # if classes_type != 'all':
    #     if classes_type == 'new':
    #         all_data_filtered = {k: v for k, v in all_data.items() if int(k) in clean_cls_idx}
    #     else:
    #         all_data_filtered = {k: v for k, v in all_data.items() if int(k) not in clean_cls_idx}
    # else:
    #     all_data_filtered = all_data
    # all_data_filtered = {k: v for k, v in all_data.items() if v[4] in accepted_classes}
    # remove classes with None values, i.e. classes that were not found in the extracted bounding boxes
    final_all_data_filtered = {k: v for k, v in all_data_filtered.items() if v[3] is not None}
    if len(final_all_data_filtered) != len(all_data_filtered):
        print('Filtered {} classes due to "None" values. '.format(len(all_data_filtered) - len(final_all_data_filtered)))
    return final_all_data_filtered


def visualize_tsne(all_data, noisy_cls_idx, clean_cls_idx, classes_type, output_file):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    plt.rcParams['legend.fontsize'] = 10

    print("Mean Average Distance on noisy classes, classes type {}: {} .".format(classes_type, np.mean([v[2] for k, v in all_data.items()])))
    print("Mean Standard Deviation on noisy classes, classes type {}: {} .".format(classes_type, np.mean([v[3] for k, v in all_data.items()])))
    print("Mean Average Distance on clean classes, classes type {}: {} .".format(classes_type, np.mean([v[0] for k, v in all_data.items()])))
    print("Mean Standard Deviation on clean classes, classes type {}: {} .".format(classes_type, np.mean([v[1] for k, v in all_data.items()])))

    print("Saving the plot.")
    width = 0.35
    num_labels = len(all_data.keys())
    x_pos = range(num_labels)
    fig, ax = plt.subplots()
    tmp_coordinates = [i-width/2 for i in x_pos]
    tmp_mean = [i[2] for i in all_data.values()]
    tmp_dvs = [i[3] for i in all_data.values()]
    ax.bar(tmp_coordinates, tmp_mean, width=width, yerr=tmp_dvs, 
            align='center', alpha=1, ecolor='black', capsize=3, label='Noisy', color='#1f77b4')  # '#1f77b4'
    tmp_coordinates = [i+width/2 for i in x_pos]
    tmp_mean = [i[0] for i in all_data.values()]
    tmp_dvs = [i[1] for i in all_data.values()]
    ax.bar(tmp_coordinates, tmp_mean, width=width, yerr=tmp_dvs, 
            align='center', alpha=1, ecolor='black', capsize=3, label='Clean', color='#ff7f0e')  # '#1f77b4'
    ax.set_xticks(x_pos, all_data.keys())
    ax.legend(loc='best')
    # finally, show the plot
    plt.savefig(output_file, dpi=1000)


def parse_args():
    """
    Parse input arguments
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parsing
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--comparison_file', type=str, default='./backup/proposals_features_comparison.json', help='Folder of extracted features')
    parser.add_argument('--output_folder', type=str, default='./proposals_features_t-sne.pdf', help='Folder where to save the output file.')
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
    noisy_cls_idx = [k for k, v in map_fn_reverse.items() if len(v) == 1]
    clean_cls_idx = [k for k, v in map_fn_reverse.items() if len(v) > 1]

    # get features comparison results 
    print('Loading all data.')
    with open(args.comparison_file, 'r') as f:
        all_data = json.load(f)

    all_data = filter_classes(all_data, clean_cls_idx, args.classes)

    # check if the folder exists
    if os.path.exists(args.comparison_file):
        print("Start plotting")
        visualize_tsne(all_data, noisy_cls_idx, clean_cls_idx, args.classes, args.output_folder)
    else:
        print("Folder not valid: ", args.comparison_file)
        exit(1)
    
