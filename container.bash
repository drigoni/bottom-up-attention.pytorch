#!/usr/bin/env bash
# inputs
MODE=$1
VERSION=$2
GPU=$3
CMD=$4

# default paths
DATASETS_PATH="$(pwd)/../../datasets"
CURRENT_FOLDER="$(pwd)"
WANDB_KEY=06de2b089b5d98ee67dcf4fdffce3368e8bac2e4
USER=dkm
USER_ID=1003
USER_GROUP=dkm
USER_GROUP_ID=1003

# variables
DOCKER_FOLDER="/home/drigoni/repository/bottom-up-attention.pytorch"


if [[ $MODE == "build" ]]; then
  # build container
  docker build ./ -t $VERSION
  docker run -v $CURRENT_FOLDER/:${DOCKER_FOLDER}/ \
    -u ${USER}:${USER_GROUP} \
    --runtime=nvidia \
    $VERSION \
    python setup.py build develop
elif [[ $MODE == "exec" ]]; then
  echo "Remove previous container: "
  docker container rm ${VERSION}-${GPU//,}
  # execute container
  echo "Execute container:"
  docker run \
    -u ${USER}:${USER_GROUP} \
    --env CUDA_VISIBLE_DEVICES=${GPU} \
    --env WANDB_API_KEY=${WANDB_KEY}\
    --name ${VERSION}-${GPU//,} \
    --runtime=nvidia \
    --ipc=host \
    -it  \
    -v ${CURRENT_FOLDER}/:${DOCKER_FOLDER}/ \
    -v ${CURRENT_FOLDER}/datasets:${DOCKER_FOLDER}/datasets \
    -v ${DATASETS_PATH}/VisualGenome/images/:${DOCKER_FOLDER}/datasets/visual_genome/images \
    -v ${DATASETS_PATH}/VisualGenome/annotations/:${DOCKER_FOLDER}/datasets/visual_genome/annotations \
    -v ${DATASETS_PATH}/flickr30k/:${DOCKER_FOLDER}/datasets/flickr30k/ \
    $VERSION \
    $CMD
elif [[ $MODE == "interactive" ]]; then
  docker run \
    -v $CURRENT_FOLDER/:${DOCKER_FOLDER}/ \
    -u ${USER}:${USER_GROUP}\
    --runtime=nvidia \
    -it \
    $VERSION \
    '/bin/bash'
else
  echo "To be implemented."
fi