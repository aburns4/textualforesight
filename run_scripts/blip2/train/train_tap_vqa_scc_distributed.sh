#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 4

# Request 1 GPU 
#$ -l gpus=4
#$ -l gpu_type="A6000|A40|A100|L40"
#$ -l gpu_memory=48G

# Specify hard time limit
#$ -l h_rt=04:30:00

# get email when job begins
#$ -m beas

# name experiment
#$ -N tapblip

#$ -t 1-3

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

mp=16321
count=0
for model in "flant5"; do
    p="lavis/projects/blip2/train/caption_tap_vqa_ft_$model.yaml"
    for tcond in "True"; do
        for warmup in "500" "1000" "1224"; do # "1224"
            for initlr in "5e-5"; do
                (( count++ ))
                (( mp++ ))
                if [[ $count -eq $SGE_TASK_ID ]]; then
                    echo ${p}
                    echo ${tcond}
                    echo ${warmup}
                    echo ${initlr}
                    echo ${count}
                    echo $mp
                    python  -m torch.distributed.run --nproc_per_node=4 --master_port=$mp train.py --sge-task-id $SGE_TASK_ID \
                                                                                                   --cfg-path $p \
                                                                                                   --options datasets.tap_vqa.vis_processor.train.name="blip_image_eval" \
                                                                                                             datasets.tap_vqa.type="caption_quad" \
                                                                                                             model.text_condition_qformer=${tcond} \
                                                                                                             run.num_workers=1 \
                                                                                                             run.distributed=True \
                                                                                                             run.world_size=4 \
                                                                                                             run.warmup_steps=${warmup} \
                                                                                                             run.init_lr=${initlr} \
                                                                                                             run.metric_type="caption" \
                                                                                                             run.max_len=10 \
                                                                                                             model.pretrained="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xl.pth"
                fi
            done
        done
    done
done

# datasets.tap_vqa.vis_processor.train.image_size=540 \
# datasets.tap_vqa.vis_processor.eval.image_size=540 \
# model.image_size=540 \