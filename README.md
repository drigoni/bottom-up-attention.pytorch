# Cleaned Classes for Visual Genome
This repository contains the code used to generate the results reported in the paper: [TODO]() \
Most of our code is based on [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch). Thanks! \
Follow the list of classes:

- Cleaned classes: [./evaluation/objects_vocab.txt](./evaluation/objects_vocab.txt) 
- Old classes: [./evaluation/old_objects_vocab.txt](./evaluation/old_objects_vocab.txt) 
- Random classes: [./evaluation/objects_vocab_random.txt](./evaluation/objects_vocab_random.txt) 


# Dependencies, Structure, Usage
For more details on repository structure, dependencies, datasets and usage, refer to the original [README](./README_bu.md)


# Datasets
To download the Visual Genome dataset, follow the instruction presented here: [README](./README_bu.md) \
To generate the new datasets:
```
python make_dataset_cleanv3.py --labels ./evaluation/objects_vocab.txt
python make_dataset_random.py --labels ./evaluation/objects_vocab_random.txt
```
or, download them from [here](https://drive.google.com/file/d/1tYn6TlOyMb2WXEek6xL-Fig433xZWhIZ/view?usp=share_link). 

Use this command to generate the list of random labels:
```
python make_categories_random.py    --labels ./evaluation/objects_vocab.txt --output_folder ./
```

# Repository Branches
This repository has the following branches:

- **master**: this branch includes all the experiments regarding the new 878 classes;
- [develop](https://github.com/drigoni/bottom-up-attention.pytorch/tree/develop): this branch includes all the experiments performed with the old 1600 classes; 
- [develop_postmapping](https://github.com/drigoni/bottom-up-attention.pytorch/tree/develop): this branch includes all the experiments performed with the old 1600 classes which are then post-processed to map to the new 878 classes;


# Checkpoints
Checkpoint of the model trained on the new 878 cleaned classes: [weight](https://drive.google.com/file/d/1obS7chZg3a-huEHtxaYYJvWcaq_q_Yxb/view?usp=share_link) 


# Example of commands
Follow some examples:
## Tran and Test
Use the following command to train the model with either cleaned or random labels:
```
# for random labels
# CONFIG_FILE=./configs/d2/train-d2-r101_random.yaml
# OUTPUT_FOLDER=./output/output_random/
# for cleaned labels
CONFIG_FILE=./configs/d2/train-d2-r101_cleaned.yaml
OUTPUT_FOLDER=./output/output_new_classes_v3/
python train_net.py \
                    --mode d2 \
                    --config  ${CONFIG_FILE} \
                    --num-gpus 1 \
                    OUTPUT_DIR ${OUTPUT_FOLDER}
```

Use the following command to test the model with either cleaned or random labels:
```
# for random labels
# CONFIG_FILE=./configs/d2/test-d2-r101_random.yaml
# OUTPUT_FOLDER=./output/output_random/
# for cleaned labels
CONFIG_FILE=./configs/d2/test-d2-r101_cleaned.yaml
OUTPUT_FOLDER=./output/output_new_classes_v3/
python train_net.py --mode d2 \
                        --config-file ${CONFIG_FILE} \
                        --num-gpus 4 \
                        --eval-only \
                        MODEL.WEIGHTS ${OUTPUT_FOLDER}model_final.pth \
                        OUTPUT_DIR ${OUTPUT_FOLDER}
```

## Extract Features
Use the following command to extract features the features:
```
# for random labels
# CONFIG_FILE=./configs/d2/test-d2-r101_random.yaml
# OUTPUT_FOLDER=./output/output_random/
# OUTPUT_FOLDER_FEATURES=./extracted_features/extracted_features_clean_VG_th02/
# for cleaned labels
CONFIG_FILE=./configs/d2/test-d2-r101_cleaned.yaml
OUTPUT_FOLDER=./output/output_new_classes_v3/
OUTPUT_FOLDER_FEATURES=./extracted_features/extracted_features_random_VG_th02/

python extract_features.py  --mode d2 \
                            --num-cpus 32 \
                            --extract-mode roi_feats \
                            --min-max-boxes 10,100 \
                            --config-file ${CONFIG_FILE} \
                            --image-dir ./datasets/visual_genome/images/ \
                            --out-dir ${OUTPUT_FOLDER_FEATURES} \
                            MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.2  \
                            OUTPUT_DIR ${OUTPUT_FOLDER}  \
                            MODEL.WEIGHTS .${OUTPUT_FOLDER}model_final.pth 
```

## Plot Dataset Frequencies
Use the following command to plot the frequencies per datasets:
```
python plot_freq_class_dataset.py   --file ./analysis/noisy_classes_frequency.json \
                                    --compare ./analysis/cleaned_classes_frequency.json \
                                    --loglog
```

**NOTE**: See the folder `./cluster` for more commands.


# Information
For any questions and comments about the new classes, contact [Davide Rigoni](mailto:davide.rigoni.2@phd.unipd.it).

# License
Apache License