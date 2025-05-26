#!/bin/bash
for dataset in IMDBBINARY IMDBMULTI MUTAG PTC; do
    for L in 1; do
        for num_layers in 1 2 3 4 5; do
            for kernel in SGTK SGNK; do
                out_dir=./outputs
                path=${out_dir}/${dataset}-K-${num_layers}-L-${num_mlp_layers}
                mkdir -p ${path}
                python gram.py --dataset ${dataset} --K ${K} --path ${path} --kernel ${kernel} --L ${L}
            done
        done
    done
done