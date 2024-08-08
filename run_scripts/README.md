# Model Checkpoints and Example Run Scripts

## Model Checkpoints

Under `lavis/output/BLIP2` lies the folders for each experiment type, which can be either for pretraining or fine-tuning. We provide the pretrained checkpoints for the element list captioning and textual foresight pretrained models, as from our open source models and baselines, element list captioning pretraining resulted in the highest screen summarization performance, and textual foresight resulted in the highest performance for element list captioning, language grounding, and tappability prediction. 

Model checkpoints are available along with pretraining data [here](https://drive.google.com/drive/folders/1vsX1BJKEFJmX7s92CpDHsHK0Wi6x8RIh?usp=drive_link).

## Example Experiment Scripts

In this folder we provide example scripts (within `blip2/`) for pretraining all models used in our paper as well as scripts for finetuning each downstream task. See the `pretrain_stuff` folder for information on setting up the pretraining and fine-tuning data. Below we outline the scripts that we provide, as well as how to modify them for different ablations or other experiments in our paper. Note that our bash scripts have syntax for launching jobs on a shared computing cluster (in particular a Sun Grid Engine (SGE) job scheduler), but they can be modified as needed to run on your GPU set up. Currently, most scripts are set up to run in a distributed setting, but can be modified to single-gpu setting and the parameters (number of GPUs, GPU specs, etc.) can of course be adjusted as needed.

| Bash Script                                        | Experiment                                 |
| -------------------------------------------------- | ------------------------------------------ |
| `train/pretrain_stage2_gpt.sh`                     | Screen captioning pretraining script       |
| `train/pretrain_stage2_elem_list.sh`               | Element list captioning pretraining script |
| `train/pretrain_stage2_textual_foresight.sh`       | Textual Foresight pretraining script       |
| `train/train_screen_caption_scc.sh`                | Screen summarization finetuning script     |
| `train/train_widget_vqa_scc_distributed.sh`        | Element captioning finetuning script       |
| `train/train_language_captions_scc_distributed.sh` | Language grounding finetuning script       |
| `eval/eval_language_ground_captions.sh`            | Language grounding evaluation script       |
| `train/train_tap_vqa_scc_distributed.sh`           | Tappability prediction finetuning script   |

Note that language grounding is the only finetuning/downstream task that requires a second step for evaluation, because there is a difference in the finetuning and evaluation regime. In particular, only "positive" samples are trained on, in which the model learns from true command, UI, and element grounding output. In contrast, at test time, the model must look at all possible elements that could match the command (i.e., comparing many negative elements and the ground truth positive). As a result, there is a different eval setup and the test data is on the larger side (instead of # samples == # UIs, it's # samples == (# Elems / UI) * # UIs).

## Other Things to Note

### Script Arguments

In each finetuning script we put the best parameters for learning rate and number of warmup steps. You can change the `model.pretrained` parameter to other pretrained models to finetune downstream tasks with either element list captioning or screen captioning upstream checkpoints. Note that our finetuning scripts by default load from `"https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xl.pth"` - which begins training from an existing BLIP2 checkpoint. We experimentally found this to result in the best performance. If you care to start from a BLIP2 stage 1 checkpoint instead, you can replace this path with `"https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"`.

Each script corresponds to a `yaml` config which can be found under `LAVIS/lavis/projects/blip2/`. Screen captioning (*aka* gpt captioning pretraining) and element list captioning both use the `spot_caps` config (e.g., `pretrain_stage2_spot_caps_flant5.yaml`). As a result, it is necessary to set the `datasets.aitw_spotlight_stage2_caption.type` field when running either pretraining script. This already done for you in the example scripts, we just care to point it out as it is a notable difference.

### Element Captioning / Spotlight Pretraining
Finally, we note that we provide a pretraining script for "Spotlight" pretraining, which is an attempt at reproducing the original element captioning objective used in prior work Spotlight. While we can run this training script, we were unable to successfully train an element captioning pretraining objective (as in, the loss does not reduce or learn anything meaningful). We believe this is in part due to the BLIP architecture which was designed to learn multiple visual features over an input image, and also due to the much shorter form element captions that are used for training Spotlight. Nonetheless we release the data used and the script set up for this in case others want to modify it or use the data for other purposes.

We tried training with element caption data during stage 1 and/or stage 2 BLIP2 pretraining to ensure it wasn't simply that it required stage 1 training. This, along with the massive number of element captions processed from OpenApp sources, is why there are `stage1` and `stage2` subsets of the data. We also deduped the samples by (app, element text, element bbox) triplets to make training more feasible in our low-resource academic set up. Under `pretrain_stuff` you can find more information on the data processing used to build OpenApp as well as the preprocessing code for formulating the finetuning files.