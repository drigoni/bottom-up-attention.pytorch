# Cleaned Classes for Visual Genome
This repository contains the code used to generate the results reported in the paper: [TODO]() \
Most of our code is based on [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch). Thanks! \
Follow the list of classes:

- Cleaned classes: [./evaluation/objects_vocab.txt](./evaluation/objects_vocab.txt) 
- Old classes: [./evaluation/old_objects_vocab.txt](./evaluation/old_objects_vocab.txt) 
- Random classes: [./evaluation/objects_vocab_random.txt](./evaluation/objects_vocab_random.txt) 


# Dependencies, Structure, Datasets, Usage
For more details on repository structure, dependencies, datasets and usage, refer to the original [README](./README_bu.md)

# Repository Branches
This repository has the following branches:

- master: this branch includes all the experiments regarding the new 878 classes;
- develop: this branch includes all the experiments performed with the old 1600 classes; 
- develop_postmapping: this branch includes all the experiments performed with the old 1600 classes which are then post-processed to map to the new 878 classes;

# Checkpoints
Checkpoint of the model trained on the new 878 cleaned classes: [weight](https://drive.google.com/file/d/1obS7chZg3a-huEHtxaYYJvWcaq_q_Yxb/view?usp=share_link) 

# Information
For any questions and comments about the new classes, contact [Davide Rigoni](mailto:davide.rigoni.2@phd.unipd.it).

# License
Apache License