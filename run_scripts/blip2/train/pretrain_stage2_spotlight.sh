#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 8
# -l mem_per_core=10G

# Request 1 GPU
#$ -l gpus=4
#$ -l gpu_type="L40"
#$ -l gpu_memory=48G

# Specify hard time limit
#$ -l h_rt=48:00:00

# get email when job begins
#$ -m beas

# name experiment
#$ -N spotlightnopreog

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

# export CUDA_LAUNCH_BLOCKING=1 # for debug
# export OMP_NUM_THREADS=${NSLOTS}
# export TF_NUM_INTEROP_THREADS=${NSLOTS}
# export TF_NUM_INTRAOP_THREADS=1

set -x

mp=17428
python -m torch.distributed.run --nproc_per_node=4 --master_port=$mp train.py \
                                                   --cfg-path "lavis/projects/blip2/train/pretrain_stage2_spotlight_flant5.yaml" \
                                                   --options run.distributed=True \
                                                             run.world_size=4 \
                                                             run.num_workers=1 \
                                                             run.batch_size_train=60 \
                                                             datasets.motif_pretrain.type="stage2" \
                                                             datasets.aitw_pretrain.type="stage2" \
                                                             datasets.longitudinal_pretrain.type="stage2" \
                                                             model.pretrained="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth" \
                                                             run.resume_ckpt_path="/projectnb/ivc-ml/aburns4/LAVIS/lavis/output/BLIP2/stage2_spotlight/202312050624/checkpoint_1.pth"
