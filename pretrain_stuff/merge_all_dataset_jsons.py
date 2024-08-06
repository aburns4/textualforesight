import json
import glob
import time
import re
import os
import argparse
import random

from collections import defaultdict
from PIL import Image

parser = argparse.ArgumentParser('Create pretrain annotations for Spotlight objective')       
parser.add_argument('--input_json_pattern',
                    nargs='+',
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/spotlight_jsons/aitw/grouped_by_app/*.json",
                    help='specify where to load intermediate json from')
parser.add_argument('--output_json_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/spotlight_jsons/aitw/subsampled",
                    help='specify where to store final jsons')

def clean_and_dedup(dataset, dname):
    bad_text = 'Google Play Store keeps stopping'
    bad_count = 0

    abt_to_sample = defaultdict(list)
    for sample in dataset:
        if bad_text.lower() not in sample['caption'].lower():
            norm_bbox = sample['screen_norm_bbox']
            formatted_bbox = '%.2f, %.2f, %.2f, %.2f' % (norm_bbox[0], norm_bbox[1], norm_bbox[2], norm_bbox[3])
            key =  "_".join([sample['app_id'], formatted_bbox, sample['caption'].lower()])
            abt_to_sample[key].append(sample)
        else:
            bad_count += 1
    print('Bad text found in %d samples' % bad_count)

    stage1 = []
    stage2 = []
    print(len(abt_to_sample.keys()))
    avg_v_len = []
    for k, v in abt_to_sample.items():
        avg_v_len.append(len(v))
        ridx1 = random.randint(0, len(v)-1)
        ridx2 = random.randint(0, len(v)-1)

        s1 = v[ridx1]
        s2 = v[ridx2]
        s1["dataset"] = dname
        s2["dataset"] = dname
        stage1.append(s1)
        stage2.append(s2)

    avg_repeats = sum(avg_v_len) / len(avg_v_len)
    print("Average number of repeats %f" % avg_repeats)
    return stage1, stage2, avg_repeats

def __main__():
    global args
    args = parser.parse_args()

    full_data_stage1 = []
    full_data_stage2 = []
    if "aitw" in args.output_json_path:
        dname = "aitw"
    else:
        print(args.input_json_pattern)
        dname = args.input_json_pattern[0].split('/')[-3]
    print(dname)
    
    unique_images_by_app = []
    if dname == "aitw":
        total_repeats = 0
        num = 0
    for dataset in args.input_json_pattern:
        dataset_paths = glob.glob(dataset)  
        for d in dataset_paths:
            print(d)
            with open(d) as f:
                data = json.load(f)

            st1, st2, avg_r = clean_and_dedup(data, dname)
            if dname == "aitw":
                total_repeats += avg_r
                num += 1
            both = st1 + st2
            imgs = set([x["image_id"] for x in both])
            unique_images_by_app.extend(list(imgs))
            print('Number of samples %d' % (len(st1)))
            print('Number of unique images %d' % len(imgs))

            full_data_stage1.extend(st1)
            full_data_stage2.extend(st2)
    
    if dname == "aitw":
        avg_r = total_repeats / num
        print("Average number of repeats over all apps %f" % avg_r)

    print('All datasets merged constitute %d' % (len(full_data_stage1)))
    both = full_data_stage1 + full_data_stage2
    imgs = set([x["image_id"] for x in both])
    print('Number of unique images %d' % len(imgs))
    print('Number of unique images by app before and after final set operation %d | %d' % (len(unique_images_by_app), len(set(unique_images_by_app))))
    if dname == "aitw":
        with open("to_store_imgs_aitw.txt", "w") as f:
            f.write('\n'.join(imgs))
    
    dist1 = int(len(full_data_stage1) / 10)
    dist2 = int(len(full_data_stage2) / 10)
    if not os.path.exists(os.path.join(args.output_json_path, "stage1")):
        os.makedirs(os.path.join(args.output_json_path, "stage1"))
    if not os.path.exists(os.path.join(args.output_json_path, "stage2")):
        os.makedirs(os.path.join(args.output_json_path, "stage2"))
    
    for i in range(0, len(full_data_stage1), dist1):
        with open(os.path.join(args.output_json_path, "stage1", "pretrain_data" + str(int(i/dist1)) + ".json"), 'w') as f:
            print(i, i+dist1)
            json.dump(full_data_stage1[i:i+dist1], f)
    
    for i in range(0, len(full_data_stage2), dist2):
        with open(os.path.join(args.output_json_path, "stage2", "pretrain_data" + str(int(i/dist2)) + ".json"), 'w') as f:
            print(i, i+dist2)
            json.dump(full_data_stage2[i:i+dist2], f)
__main__()