#!/bin/bash

RECORD=2996
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=./config/test.yaml

WEIGHTS=runs/2102-67-34884.pt # 71.6/93.4


BATCH_SIZE=32

python main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS

python pkl2npy.py
