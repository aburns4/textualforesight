#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 4

# Request 1 GPU 
#$ -l gpus=2
#$ -l gpu_type="A40|A100|A6000"
#$ -l gpu_memory=48G

# Specify hard time limit
#$ -l h_rt=20:00:00

# get email when job begins
#$ -m beas

# name experiment
#$ -N noprompt

#$ -t 1

module load miniconda
module load cuda/11.6
conda activate lavis

export PYTHONPATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/bin:$PATH
export TORCH_HOME=/projectnb2/ivc-ml/aburns4/LAVIS/model_cache/
export HF_HOME=/projectnb/ivc-ml/aburns4/huggingface_cache/
export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1

mp=27698
count=0
for model in "flant5"; do
    p="lavis/projects/blip2/train/no_prompt_caption_screen_ft_$model.yaml"
    for warmup in "1000"; do
        for initlr in "1e-5"; do
            (( count++ ))
            (( mp++ ))
            if [[ $count -eq $SGE_TASK_ID ]]; then
                echo ${p}
                echo ${warmup}
                echo ${initlr}
                echo ${count}
                # model.image_size=490 datasets.screen_caption.vis_processor.train.image_size=490 datasets.screen_caption.vis_processor.eval.image_size=490
                python -m torch.distributed.run --nproc_per_node=2 --master_port=$mp train.py --sge-task-id $SGE_TASK_ID --cfg-path $p --options datasets.screen_caption.text_processor.train.prompt="" run.distributed=True run.world_size=2 run.num_workers=4 run.warmup_steps=${warmup} run.init_lr=${initlr}
            fi
        done
    done
done