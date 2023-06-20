#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_type=A100
#$ -l gpu_memory=48G

# Specify hard time limit
#$ -l h_rt=25:00:00

# get email when job begins
#$ -m beas

s="${1:-test}"
t="${2:-language_ground}"
i="${3:-/projectnb2/ivc-ml/aburns4/mug}"
b="${4:-1}"

echo "split $s | task $t | input directory $i | batch size $b"

module load miniconda
module load cuda/11.6
conda activate lavis

export PYTHONPATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/bin:$PATH
export TORCH_HOME=/projectnb2/ivc-ml/aburns4/LAVIS/model_cache/
export HF_HOME=/projectnb/ivc-ml/aburns4/huggingface_cache/

python zero_shot_eval_rico_tap.py --data_split $s --task $t --input_data_dir $i --batch_size $b