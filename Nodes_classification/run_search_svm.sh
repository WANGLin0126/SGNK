#!/bin/bash
for dataset in Cora Citeseer Pubmed Photo Computers; do
    for kernel in SGTK SGNK; do
        echo "dataset: ${dataset}, kernel: ${kernel}"
        python search_svm.py  --dataset ${dataset} --kernel ${kernel}
    done
done