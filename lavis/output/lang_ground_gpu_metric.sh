#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 1

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_memory=48G

# Specify hard time limit
#$ -l h_rt=06:30:00

# get email when job begins
#$ -m beas

# name experiment
#$ -N blrtlg

module load miniconda
module load cuda/11.6
conda activate lavis

export PYTHONPATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/bin:$PATH

python language_ground_caption_eval.py --metric bleurt --constraint "<=" --res_file_path "/projectnb2/ivc-ml/aburns4/LAVIS/lavis/output/BLIP2/language_ground/flant5/202405161305_1/*/result/test_vqa_result.json"
