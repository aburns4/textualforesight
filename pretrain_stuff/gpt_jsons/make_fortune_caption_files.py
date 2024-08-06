import argparse
import json
import glob
import os

from collections import defaultdict

parser = argparse.ArgumentParser('')
parser.add_argument('--input_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/aitw/gpt_captions/*",
                    help='path to read gpt caption samples from')
parser.add_argument('--dataset_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/spotlight_jsons/aitw/st_at_st1/triplets_clean.txt",
                    help='path to read valid state-action triplets from')

NUM_JSONS_TO_STORE = {"aitw": 20,
                      "motif": 10,
                      "longitudinal": 10}

QUESTION = "What does the screen show if the UI object found at [%s, %s, %s, %s] is interacted with?"

def screen_to_caption_map(caption_path):
    map_dict = defaultdict(str)
    fis = glob.glob(caption_path)
    for fi in fis:
        with open(fi) as f:
            d = json.load(f)
        for sample in d:
            map_dict[sample["image"]] = sample["caption"]
    return map_dict

def create_aitw_samples(caption_map, state_action_samples):
    samples = []
    missed = []
    for sa in state_action_samples:
        new_sample = {}
        
        app, root_id, st, st1, bbox = sa.strip().split(",")
        bbox = bbox.split(" ")
        curr_state_image = "_".join([root_id, st + ".jpg"])
        next_state_image = "_".join([root_id, st1 + ".jpg"])
        # print(next_state_image)
        new_sample["image"] = curr_state_image
        new_sample["question"] = QUESTION % (bbox[0], bbox[1], bbox[2], bbox[3])
        if next_state_image not in caption_map:
            missed.append(",".join([app, next_state_image]))
            # print(next_state_image)
            continue
        new_sample["caption"] = caption_map[next_state_image]
        samples.append(new_sample)
    print("AITW missed %d samples" % len(missed))
    with open("aitw_cant_be_used_fortune.txt", "w") as f:
        f.write("\n".join(missed))
    return samples

def create_motif_samples(caption_map, state_action_samples):
    samples = []
    for sa in state_action_samples:
        new_sample = {}
        
        root_id, st, st1, bbox = sa.strip().split(",")
        bbox = bbox.split(" ")
        curr_state_image = "/screens/".join([root_id, st])
        next_state_image = "/screens/".join([root_id, st1])
        
        new_sample["image"] = curr_state_image
        new_sample["question"] = QUESTION % (bbox[0], bbox[1], bbox[2], bbox[3])
        assert next_state_image in caption_map
        new_sample["caption"] = caption_map[next_state_image]
        samples.append(new_sample)
    return samples

def create_longitudinal_samples(caption_map, state_action_samples):
    samples = []
    for sa in state_action_samples:
        new_sample = {}
        
        root_id, st, st1, bbox = sa.strip().split(",")
        bbox = bbox.split(" ")
        curr_state_image = os.path.join(root_id, st) + ".png"
        next_state_image = os.path.join(root_id, st1) + ".png"
        
        new_sample["image"] = curr_state_image
        new_sample["question"] = QUESTION % (bbox[0], bbox[1], bbox[2], bbox[3])
        assert next_state_image in caption_map
        new_sample["caption"] = caption_map[next_state_image]
        samples.append(new_sample)
    return samples

def __main__():
    global args
    args = parser.parse_args()

    gpt_captions = screen_to_caption_map(args.input_path)
    dname = args.dataset_path.split("/")[-3]
    assert dname in NUM_JSONS_TO_STORE

    with open(args.dataset_path) as f:
        sa_data = [x.strip() for x in f.readlines()]
        sa_data = set(sa_data)

    if dname == "aitw":
        fortune_samples = create_aitw_samples(gpt_captions, sa_data)
    elif dname == "longitudinal":
        fortune_samples = create_longitudinal_samples(gpt_captions, sa_data)
    elif dname == "motif":
        fortune_samples = create_motif_samples(gpt_captions, sa_data)
    else:
        raise ValueError("Dataset is not supported.")

    write_path = args.input_path[:-1].replace("gpt_captions", "fortune_captions")
    print("Writing to %s..." % write_path)
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    
    increment = len(fortune_samples) // NUM_JSONS_TO_STORE[dname]
    for i in range(0, len(fortune_samples), increment):
        print(i, i+increment, len(fortune_samples))
        subset = fortune_samples[i:i+increment]
        with open(os.path.join(write_path, str(i//increment) + "_captions.json"), "w") as f:
            json.dump(subset, f)

__main__()