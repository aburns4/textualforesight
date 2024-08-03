#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 8

# Request 1 GPU 
#$ -l gpus=4
#$ -l gpu_type="A40|A100|A6000"
#$ -l gpu_memory=48G

# Specify hard time limit
#$ -l h_rt=02:00:00

# get email when job begins
#$ -m beas

# name experiment
#$ -N evalgroundforesight

#$ -t 1

module load miniconda
module load cuda/11.6
conda activate lavis

export PYTHONPATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/bin:$PATH
export TORCH_HOME=/projectnb2/ivc-ml/aburns4/LAVIS/model_cache/
export HF_HOME=/projectnb/ivc-ml/aburns4/huggingface_cache/
export TOKENIZERS_PARALLELISM=false

# needed so that torch distributed barrier does not hang
export NCCL_P2P_DISABLE=1

mp=27834
count=0
p="lavis/projects/blip2/eval/language_grounding_captions.yaml"
dirs='/projectnb2/ivc-ml/aburns4/LAVIS/lavis/output/BLIP2/language_ground/flant5/*'

for i in $dirs; do
    len_i="${#i}"
    odir="${i:39:len_i}"
    (( count++ ))
    (( mp++ ))
    if [[ $count -eq $SGE_TASK_ID ]]; then
        python -m torch.distributed.run --nproc_per_node=4 --master_port=$mp evaluate.py \
               --sge-task-id $SGE_TASK_ID --cfg-path $p --options run.num_workers=1 \
                                                                  run.distributed=True \
                                                                  run.world_size=4 \
                                                                  run.output_dir=${odir} \
                                                                  run.text_condition_qformer=True \
                                                                  model.arch="blip2_t5" \
                                                                  model.model_type="vqa_screen_flant5xl" \
                                                                  run.batch_size_eval=32
    fi
done