#!/bin/bash

function train {
    python $run_file \
        --path_config $path_config_extra \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train \
        --task 'train' \
        --config $path_config
}

function test {
    python $run_file \
        --path_config $path_config_extra \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --task 'eval' \
        --config $path_config
}

folder_data='../../compositional-scene-representation-datasets'
run_file='main.py'

for name in 'mnist' 'dsprites' 'abstract' 'clevr' 'shop' 'gso'; do
    path_config='configs/'$name'.yaml'
    path_config_extra='configs/'$name'_extra.yaml'
    path_data=$folder_data'/'$name'.h5'
    folder_log='../output/logs/'$name
    folder_out='../output/outs/'$name
    train
    test
done
