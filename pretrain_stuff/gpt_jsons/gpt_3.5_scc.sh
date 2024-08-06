#!/bin/bash -l

# Set SCC project
# -P ivc-ml
#$ -pe omp 12

# Specify hard time limit
#$ -l h_rt=00:01:00

# get email when job begins
#$ -m beas

# name experiment
#$ -N querygpt

module load miniconda
module load cuda/11.6
conda activate lavis

set -x

export PYTHONPATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/bin:$PATH
export TORCH_HOME=/projectnb2/ivc-ml/aburns4/LAVIS/model_cache/
export HF_HOME=/projectnb/ivc-ml/aburns4/huggingface_cache/
export TOKENIZERS_PARALLELISM=false
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export OPENAI_API_KEY=""

python gpt3_5_turbo_async.py --start_range 0 \
                             --end_range 500 \
                             --increment 300 \
                             --increment_fraction 4 \
                             --input_path "/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/set_samples_all_final.txt"
