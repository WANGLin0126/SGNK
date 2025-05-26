#!/bin/bash
for dataset in Cora Citeseer Pubmed Photo Computers; do
    for kernel in SGTK SGNK; do
        for K in 1 2 3 4 5; do
            python search_krr.py  --dataset ${dataset} --kernel ${kernel} --K ${K} 
        done
    done
done