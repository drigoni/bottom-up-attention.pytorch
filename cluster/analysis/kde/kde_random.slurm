#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='random_kde'     # create a short name for your job
#SBATCH --account=p200049                           
#SBATCH --mail-user=davide.rigoni.2@phd.unipd.it
#SBATCH --output=cluster/out/random_labels_kde.txt
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --mem=100GB				                    # total memory per node
#SBATCH --time=06:00:00                           # total run time limit (DD-HH:MM:SS)
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=128		                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=4                           # number of gpus per node


### ============ VARIABLES SETTING
SLURM_MASTER_PORT=$(expr 10000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
SLURM_MASTER_NODE="$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)" # such as gn11
SLURM_MASTER_ADDR="${SLURM_MASTER_NODE}"
SLURM_MASTER_URL="tcp://${SLURM_MASTER_ADDR}:${SLURM_MASTER_PORT}"
export SLURM_MASTER_PORT=$SLURM_MASTER_PORT
export SLURM_MASTER_NODE=$SLURM_MASTER_NODE
export SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR

# model variables
NUM_GPUS=${SLURM_GPUS_PER_NODE}
NUM_MACHINES=${SLURM_NNODES}
BATCH_SIZE=$(expr 5 \* $(echo -n ${SLURM_JOB_NUM_NODES}) \* $(echo -n ${SLURM_GPUS_PER_NODE})) # max 5 per A100
MAX_ITER=$(expr 32 \* 90000 \/ $(echo -n ${BATCH_SIZE}))

### ============ SOME PRINT COMMANDS
echo ''
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo 'Date: ' $(date)
echo 'Directory: ' $(pwd)
echo "Nodelist: " ${SLURM_JOB_NODELIST}                  # like gn11 or gn[37,58]
echo "Number of nodes: " ${SLURM_JOB_NUM_NODES}
echo "Ntasks per node: "  ${SLURM_NTASKS_PER_NODE}
echo "NGPUs per node: "  ${SLURM_GPUS_PER_NODE}
echo "CUDA_VISIBLE_DEVICES: " ${CUDA_VISIBLE_DEVICES}
echo "TORCH_DEVICE_COUNT: " $(python -c 'import torch; print(torch.cuda.device_count())')
echo "SLURM_MASTER_PORT: " ${SLURM_MASTER_PORT}
echo "SLURM_MASTER_NODE: " ${SLURM_MASTER_NODE}
echo "SLURM_MASTER_ADDR: " ${SLURM_MASTER_ADDR}
echo "SLURM_MASTER_URL: " ${SLURM_MASTER_URL}
echo "--------------------------------------------- "
echo "MODEL_NUM_GPUS: " ${NUM_GPUS}
echo "MODEL_NUM_MACHINES: " ${NUM_MACHINES}
echo "MODEL_BATCH_SIZE: " ${BATCH_SIZE}
echo "MODEL_MAX_ITER: " ${MAX_ITER}
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ''

module load CUDA

srun python plot_pred_bestclass_scores_kdeplot.py   --file1 ./results/output_develop_postprocessing_random/inference/instances_predictions.pth \
                                                    --file2 ./results/output_random/inference/instances_predictions.pth \
                                                    --inf_type score \
                                                    --labels ./evaluation/objects_vocab_random.txt \
                                                    --output ./predicted_scores_kde_random_all.pdf \
                                                    --classes all

srun python plot_pred_bestclass_scores_kdeplot.py   --file1 ./results/output_develop_postprocessing_random/inference/instances_predictions.pth \
                                                    --file2 ./results/output_random/inference/instances_predictions.pth \
                                                    --inf_type score \
                                                    --labels ./evaluation/objects_vocab_random.txt \
                                                    --output ./predicted_scores_kde_random_old.pdf \
                                                    --classes old

srun python plot_pred_bestclass_scores_kdeplot.py   --file1 ./results/output_develop_postprocessing_random/inference/instances_predictions.pth \
                                                    --file2 ./results/output_random/inference/instances_predictions.pth \
                                                    --inf_type score \
                                                    --labels ./evaluation/objects_vocab_random.txt \
                                                    --output ./predicted_scores_kde_random_new.pdf \
                                                    --classes new


# python plot_pred_bestclass_scores_kdeplot.py --file1 ./results/output_develop_postmapping/inference/instances_predictions.pth --file2 ./results/output_new_classes_v3/inference/instances_predictions.pth --output ./diff_new_predicted_scores_kde.pdf --inf_type score --classes all"
# bash container.bash exec bu-drigoni-dkm 00 "python plot_freq_class_dataset.py --file ./analysis/cleaned_classes_frequency.json --compare ./analysis/random_classes_frequency.json --loglog"
# 

# last default print
echo ''
echo ''
echo 'Job done.'
echo 'Date: ' $(date)