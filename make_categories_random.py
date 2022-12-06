import os
import argparse
import random
import copy


def load_objects(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    # cleaning
    data = [i.strip() for i in data]
    return data


def save_objects(output_folder, data):
    file_name = 'objects_vocab_random.txt'
    path = "{}{}".format(output_folder, file_name)
    with open(path, 'w') as f:
        f.writelines('\n'.join(data))
    print("new random set of categories saved: ", path)


def make_random_set(noisy_labels, N_CLASS):
    print("Generating the new random set of labels")
    assert len(noisy_labels) == 1600
    new_random_set_deepcopy = copy.deepcopy(noisy_labels)

    # add number in front of label
    new_random_set = []
    for index in range(len(new_random_set_deepcopy)):
        # like 44:television,44:tv
        label = new_random_set_deepcopy[index]
        sublabels = label.split(',')
        name = ''
        for i, sublabel in enumerate(sublabels):
            if i > 0:
                name += ','
            name += str(index+1) + ":" + sublabel
        new_random_set.append(name)
    while len(new_random_set) > N_CLASS:
        current_n_elements = len(new_random_set)
        selected_idx_class = random.randint(0, current_n_elements-1)
        joined_idx_class = random.randint(0, current_n_elements-1)
        if selected_idx_class != joined_idx_class:
            new_random_set[selected_idx_class] += "," + new_random_set[joined_idx_class]
            # remove class
            del new_random_set[joined_idx_class]
    return new_random_set
    


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
                        default="./evaluation/objects_vocab_cleaned.txt",
                        type=str)
    parser.add_argument('--output_folder', dest='output_folder',
                        help='Output folder.',
                        default="./",
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data = load_objects(args.labels)
    data = make_random_set(data, N_CLASS=878)
    save_objects(args.output_folder, data)
    # print(data)