# Cleaned Classes for Visual Genome
In this branch it is possible to post-process the object detector results to map the classes to the 878 cleaned (or random) labels:

```
python train_net.py --mode d2 \
                        --config-file configs/d2/test-d2-r101_clean.yaml \
                        --num-gpus 1 \
                        --eval-only \
                        MODEL.WEIGHTS ./results/output_develop/model_final.pth \
                        OUTPUT_DIR ./output/output_develop_postprocessing_clean/

python train_net.py --mode d2 \
                        --config-file configs/d2/test-d2-r101_random.yaml \
                        --num-gpus 1 \
                        --eval-only \
                        MODEL.WEIGHTS ./results/output_develop/model_final.pth \
                        OUTPUT_DIR ./output/output_develop_postprocessing_random/
```