#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 6
#$ -l mem_per_core=16G

# Request 1 GPU
#$ -l gpus=3
#$ -l gpu_type="L40|A40"
#$ -l gpu_memory=48G

# Specify hard time limit
#$ -l h_rt=16:00:00

# get email when job begins
#$ -m beas

# name experiment
#$ -N fewgpulr

# -t 1

module load miniconda
module load cuda/11.6
conda activate lavis

export PYTHONPATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/bin:$PATH
export TORCH_HOME=/projectnb2/ivc-ml/aburns4/LAVIS/model_cache/
export HF_HOME=/projectnb/ivc-ml/aburns4/huggingface_cache/
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG="INFO"

export NCCL_P2P_DISABLE=1

mp=24518
python -m torch.distributed.run --nproc_per_node=3 --master_port=$mp train.py \
                                                   --cfg-path "lavis/projects/blip2/train/pretrain_stage1_continued.yaml" \
                                                   --options run.distributed=True \
                                                             run.world_size=3 \
                                                             run.init_lr="5e-5" \
                                                             run.num_workers=0 \
                                                             run.batch_size_train=100 \