# PSRNet
PyTorch implementation for **One-shot Learning for Population Mapping**

# Datasets
- data.zip : population and POI distribution (500m x 500m) from CITY1 to CITY4.

# Requirements
- Provided in **environment.yaml**

# Project Structure
- run.sh : bash to run all experiments
- STNet_train.py : Training codes for STNet
- PGNet_train.py : Training codes for PGNet
- STNet.py : Architecture codes for STNet
- PGNet.py : Architecture codes for PGNet


# Usage
> unzip data.zip
> bash run.sh
- The training is time consuming so you may run all run.sh's command parallelly except the last one to accelerate. The last Fine-tuning.py depends on the checkpoints of previous experiments.
