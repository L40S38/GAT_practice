#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda create -n torch_env python==3.10.6 -y
conda activate torch_env 
conda info -e
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install matplotlib graphviz python-graphviz -c conda-forge -y
conda install -c anaconda networkx -y
conda install -c pyg pyg -y
conda install biopandas -y