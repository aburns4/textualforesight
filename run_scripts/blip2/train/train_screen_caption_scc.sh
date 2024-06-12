#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 4

# Request 1 GPU 
#$ -l gpus=4
#$ -l gpu_type="A40|L40|A100|A6000"
#$ -l gpu_memory=48G
# -l mem_per_core=10G

# Specify hard time limit
#$ -l h_rt=10:00:00

# get email when job begins
#$ -m beas

# name experiment
#$ -N scgptredos2pre

#$ -t 1-3

module load miniconda
module load cuda/11.6
conda activate lavis

export PYTHONPATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/bin:$PATH
export TORCH_HOME=/projectnb2/ivc-ml/aburns4/LAVIS/model_cache/
export HF_HOME=/projectnb/ivc-ml/aburns4/huggingface_cache/
export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1

mp=12410
count=0
for model in "flant5"; do
    p="lavis/projects/blip2/train/caption_screen_ft_$model.yaml"
    for warmup in "1000" "2500" "4919"; do # "1000" "2500" "4919"
        for initlr in "1e-5"; do
            (( count++ ))
            (( mp++ ))
            if [[ $count -eq $SGE_TASK_ID ]]; then
                echo ${p}
                echo ${warmup}
                echo ${initlr}
                echo ${count}
                python -m torch.distributed.run --nproc_per_node=4 --master_port=$mp train.py \
                       --sge-task-id $SGE_TASK_ID --cfg-path $p \
                       --options run.distributed=True \
                                 run.world_size=4 \
                                 run.num_workers=1 \
                                 run.warmup_steps=${warmup} \
                                 run.init_lr=${initlr} \
                                 datasets.screen_caption.vis_processor.train.name="blip_image_eval" \
                                 model.pretrained="/projectnb/ivc-ml/aburns4/LAVIS/lavis/output/BLIP2/stage2_spotlight/202311221948/checkpoint_4.pth"
            fi
        done
    done
done