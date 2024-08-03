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
#$ -N tapforesight

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

mp=16321
p="lavis/projects/blip2/train/caption_tap_vqa_ft_flant5.yaml"
python  -m torch.distributed.run --nproc_per_node=4 --master_port=$mp train.py --sge-task-id $SGE_TASK_ID \
                                                                                --cfg-path $p \
                                                                                --options datasets.tap_vqa.vis_processor.train.name="blip_image_eval" \
                                                                                            datasets.tap_vqa.type="caption_quad" \
                                                                                            model.text_condition_qformer=True \
                                                                                            run.num_workers=1 \
                                                                                            run.distributed=True \
                                                                                            run.world_size=4 \
                                                                                            run.warmup_steps="1124" \
                                                                                            run.init_lr="5e-5" \
                                                                                            run.metric_type="caption" \
                                                                                            run.max_len=10 \
                                                                                            model.pretrained="/projectnb/ivc-ml/aburns4/LAVIS/lavis/output/BLIP2/stage2_fortune/202310220230/checkpoint_4.pth"