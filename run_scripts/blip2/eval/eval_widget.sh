#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 3

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_type="A40|A100|A6000"
#$ -l gpu_memory=48G

# Specify hard time limit
#$ -l h_rt=00:25:00

# get email when job begins
#$ -m beas

# name experiment
#$ -N t5again

#$ -t 1

module load miniconda
module load cuda/11.6
conda activate lavis

export PYTHONPATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/lib/python3.8/site-packages/:$PYTHONPATH
export PATH=/projectnb2/ivc-ml/aburns4/LAVIS/pythonlibs/bin:$PATH
export TORCH_HOME=/projectnb2/ivc-ml/aburns4/LAVIS/model_cache/
export HF_HOME=/projectnb/ivc-ml/aburns4/huggingface_cache/
export TOKENIZERS_PARALLELISM=false

dirs='/projectnb2/ivc-ml/aburns4/LAVIS/lavis/output/BLIP2/widget_caption/flant5/202307180*_11'
p="lavis/projects/blip2/eval/widget_caption_flant5xl_eval.yaml"
count=0
for i in $dirs; do
    # "output/BLIP2/widget_caption/opt2/20230719055_19" "output/BLIP2/widget_caption/opt2/20230719060_20" "output/BLIP2/widget_caption/opt2/20230719081_21" "output/BLIP2/widget_caption/opt2/20230719081_22"; do # $dirs; do
    len_i="${#i}"
    odir="${i:39:len_i}"
    (( count++ ))
    if [[ $count -eq $SGE_TASK_ID ]]; then #  && [[ ! $i == "_10" || ! $i == "_11" ]]
        echo $odir
        python evaluate.py --sge-task-id $SGE_TASK_ID --cfg-path $p --options model.text_condition_qformer=False run.num_workers=3 run.output_dir=$odir
    fi
done
