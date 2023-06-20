import argparse
import json
import math
import os
import time
import torch

from collections import defaultdict
from PIL import Image

from lavis.models import load_model_and_preprocess

parser = argparse.ArgumentParser('BLIP-2 screen summarization')
parser.add_argument('--batch_size',
                    type=int,
                    default=10,
                    help='batch size for evaluation')
parser.add_argument('--input_data_dir',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/screen2words",
                    help='specify path to evaluation dataset')
parser.add_argument('--input_image_dir',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/motif_ricosca_images",
                    help='specify path to raw images')
parser.add_argument('--data_split',
                    type=str,
                    default="test",
                    help='specify which dataset split to evaluate')                  
parser.add_argument('--model_output_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/zero_shot_outputs",
                    help='specify path to output caption dir')

def load_split(split, anns_dir):
    split_path = os.path.join(anns_dir, "split", split) + "_screens.txt"
    ann_path = os.path.join(anns_dir, "screen_summaries.csv")

    with open(split_path, "r") as f:
        split_ids = f.readlines()
        split_ids = [x.strip() for x in split_ids]

    with open(ann_path, "r") as f:
        anns = f.readlines()[1:]
        anns = [x.split(',') for x in anns]

    split_anns = defaultdict(lambda: defaultdict(str))
    for sample in anns:
        if sample[0] in split_ids:
            if sample[0] in split_anns:
                split_anns[sample[0]]["target"].append(sample[1].strip())
            else:
                split_anns[sample[0]]["target"] = [sample[1].strip()]

    print("Loaded %d %s samples for zero shot evaluation" % (len(split_anns.keys()), split))

    return split_anns

def load_images(image_ids, image_dir, vis_preprocessor, device):
    processed = []
    for image in image_ids:
        image_path = os.path.join(image_dir, image + '.jpg')
        raw_image = Image.open(image_path).convert('RGB')
        processed_image = vis_preprocessor["eval"](raw_image).to(device)
        processed.append(processed_image)
    
    processed = torch.stack(processed)
    return processed

def generate_zero_shot_summaries(images, image_ids, model, target_dict):
    images.cuda()
    no_prompt = model.generate({"image": images})
    prompt1 = model.generate({"image": images, "prompt": "Question: What is the description of the screen? Answer:"})
    prompt2 = model.generate({"image": images, "prompt": "Question: What best summarizes the UI? Answer:"})
    model_outputs = zip(image_ids, no_prompt, prompt1, prompt2)
    for x in model_outputs:
        target_dict[x[0]].update({"no_prompt": x[1],
                                  "prompt1": x[2],
                                  "prompt2": x[3]})
    return target_dict

def __main__():
    global args
    args = parser.parse_args()
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size
    target_anns = load_split(args.data_split, args.input_data_dir)
    assert len(target_anns) % batch_size == 0

    # we associate a model with its preprocessors to make it easier for inference.
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
    )

    split_img_ids = list(target_anns.keys())
    num_batches = math.ceil(len(split_img_ids) / batch_size)
    print('Number of batches to evaluate %d' % num_batches)
    batched_ids = [split_img_ids[i:i+batch_size] for i in range(0, len(split_img_ids), batch_size)]
    batched_images = [load_images(batch, args.input_image_dir, vis_processors, device) for batch in batched_ids]
    
    processed_batches = 0
    for img_batch, id_batch in zip(batched_images, batched_ids):
        target_anns = generate_zero_shot_summaries(img_batch, id_batch, model, target_anns)
        if processed_batches % 10 == 0:
            print('Batch %d complete...' % processed_batches)
        processed_batches += 1

    with open(os.path.join(args.model_output_path, "screen_summarization_zero_shot_" + args.data_split + ".json"), "w") as f:
        json.dump(target_anns, f)


start_time = time.time()
__main__()
end_time = time.time()
seconds = end_time - start_time
minutes = seconds / 60
hours = minutes / 60
print('Time to evaluate %.2f seconds / %.2f minutes / %.2f / hours' % (seconds, minutes, hours))