#!/bin/bash
for dataset in IMDBBINARY IMDBMULTI MUTAG PTC; do
    for kernel in SGTK SGNK; do
        for L in 1; do
            for K in 1 2 3 4 5; do
                python search.py --dataset ${dataset} --K ${K} --L ${L} --kernel ${kernel} --data_dir ./outputs
            done
        done
    done
done