# kiedra-exp
This repository contains comparison experiments between Kiedra and other Feature Selection algorithms

# Prerequisites
It requires the following repo to be installed next to this repo: https://github.com/narg95/model-free-data. 
This repo contains the datasets used in the experiments.

# To Run

Please run `python -W ignore run.py`. The results will be saved in file `./results.tab`. 
Note that this file will be deleted if it exist before executing the experiments.
The experiments run in parallel taking in to account the amount of available cores in your cpu.