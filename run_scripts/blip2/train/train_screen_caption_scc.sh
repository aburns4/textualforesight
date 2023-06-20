#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
# -pe omp 3

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_type=A40|A100|A6000
#$ -l gpu_memory=48G

# Specify hard time limit
#$ -l h_rt=20:00:00

# get email when job begins
#$ -m beas

p="${1:-lavis/projects/blip2/train/caption_screen_ft_flant5.yaml}"

echo "training yaml path $p"

module load miniconda
module load cuda/11.6
conda activate lavis

export PYTHONPATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/bin:$PATH
export TORCH_HOME=/projectnb2/ivc-ml/aburns4/LAVIS/model_cache/
export HF_HOME=/projectnb/ivc-ml/aburns4/huggingface_cache/

python train.py --cfg-path $p