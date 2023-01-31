#!/bin/bash

cd ..

# custom config
DATA=/home/data
TRAINER=DAPL

DATASET=$1
CFG=$2  # config file
T=$3 # temperature
TAU=$4 # pseudo label threshold
U=$5 # coefficient for loss_u
NAME=$6 # job name


for SEED in 2
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --model-dir ${DIR} \
    --output-dir ${DIR} \
    --eval-only
done