mkdir ../pop_sr
mkdir ../checkpoint
mkdir ../checkpoint/PGNetGenerator
conda env create -f environment.yaml
source activate torch-1.6-py3

############################## Pre-training of STNet ##############################

# Super-resolution Pre-training : CITY1 (1000m -> 500m)
python STNet_train.py -source "['CITY1']" -target "['CITY1']" -hr_downscale 0 -lr_downscale 1

# Super-resolution Pre-training : CITY1 (2000m -> 500m)
python STNet_train.py -source "['CITY1']" -target "['CITY1']" -hr_downscale 0 -lr_downscale 2

# Super-resolution Pre-training : CITY1 (4000m -> 500m)
python STNet_train.py -source "['CITY1']" -target "['CITY1']" -hr_downscale 0 -lr_downscale 3

# Super-resolution Pre-training : CITY2 (2000m -> 1000m)
python STNet_train.py -source "['CITY2']" -target "['CITY2']" -hr_downscale 1 -lr_downscale 2

# Super-resolution Pre-training : CITY3 (2000m -> 1000m)
python STNet_train.py -source "['CITY3']" -target "['CITY3']" -hr_downscale 1 -lr_downscale 2

############################## Pre-training and Data Augmentation of PGNet ##############################

# Data Augmentation across Cities : CITY1 (1000m -> 500m) => [CITY2, CITY3] (1000m -> 500m)
python PGNet_train.py -source "['CITY1']" -target "['CITY2', 'CITY3']" \
-source_lr_downscale 1 -source_hr_downscale 0 \
-target_lr_downscale 1 -target_hr_downscale 0

# Data Augmentation across Cities : CITY1 (2000m -> 500m) => [CITY2, CITY3] (2000m -> 500m)
python PGNet_train.py -source "['CITY1']" -target "['CITY2', 'CITY3']" \
-source_lr_downscale 2 -source_hr_downscale 0 \
-target_lr_downscale 2 -target_hr_downscale 0

# Data Augmentation across Cities : CITY1 (4000m -> 500m) => [CITY2, CITY3] (4000m -> 500m)
python PGNet_train.py -source "['CITY1']" -target "['CITY2', 'CITY3']" \
-source_lr_downscale 3 -source_hr_downscale 0 \
-target_lr_downscale 3 -target_hr_downscale 0

# Data Augmentation across Granularities : CITY2 (2000m -> 1000m) => CITY2 (1000m -> 500m)
python PGNet_train.py -source "['CITY2']" -target "['CITY2']" \
-source_lr_downscale 2 -source_hr_downscale 1 \
-target_lr_downscale 1 -target_hr_downscale 0

# Data Augmentation across Granularities : CITY3 (2000m -> 1000m) => CITY3 (1000m -> 500m)
python PGNet_train.py -source "['CITY3']" -target "['CITY3']" \
-source_lr_downscale 2 -source_hr_downscale 1 \
-target_lr_downscale 1 -target_hr_downscale 0

############################## Fine-tuning ##############################

python Fine-tuning.py
