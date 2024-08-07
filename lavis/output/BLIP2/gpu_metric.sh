#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 1

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_memory=24G

# Specify hard time limit
#$ -l h_rt=04:00:00

# get email when job begins
#$ -m beas

# name experiment
#$ -N blrtwc

module load miniconda
module load cuda/11.6
conda activate lavis

export PYTHONPATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/bin:$PATH



python fresh_metric.py --res_file_path "/projectnb2/ivc-ml/aburns4/LAVIS/lavis/output/BLIP2/widget_caption/flant5/20230802*_9/result/*_vqa_result.json" \
                       --ann_file_path "/projectnb2/ivc-ml/aburns4/widget-caption" \
                       --val_split_name "dev" \
                       --metrics "bleurt"
