#!/bin/bash
RECORD=2102
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=./config/train.yaml

START_EPOCH=50
EPOCH_NUM=100
BATCH_SIZE=32
WARM_UP=5
SEED=777

python main.py --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 1 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED

