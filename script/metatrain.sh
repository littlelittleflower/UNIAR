#!/bin/sh

dataset='ace_2005' # 'ace_2005', 'eventkg'
version=1 # 1, 2, 3, or 4
kge='RotatE' # 'TransE', 'DistMult', 'ComplEx', or 'RotatE'
gpu=0

python main.py --data_name ${dataset}_v${version} --name ${dataset,,}_v${version}_${kge,,} \
        --step meta_train --kge ${kge} --gpu cuda:${gpu}