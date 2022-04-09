#!/bin/bash

for c in {1..4}
do
    python3 train.py --config pcpnet --id_cluster $c
done
