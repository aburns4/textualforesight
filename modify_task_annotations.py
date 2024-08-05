import argparse
import glob
import json
import math
import os
import time
import torch
import tensorflow as tf 
import re

from collections import defaultdict
from PIL import Image

from lavis.models import load_model_and_preprocess

parser = argparse.ArgumentParser('Update fine-tune task annotations')
parser.add_argument('--input_data_dir',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/combined",
                    help='specify path to evaluation dataset')
parser.add_argument('--input_anns_dir',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/taperception",
                    help='specify path to evaluation dataset')
parser.add_argument('--data_split',
                    type=str,
                    default="test",
                    help='specify which dataset split to annotate')       
parser.add_argument('--task',
                    type=str,
                    default="tappability",
                    help='specify which downstream task to annotate')             

def convert_view_to_screen_dims(bbox, scale_x, scale_y):
    # need to convert to screen localization
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    new_x1 = bbox[0] * scale_x
    new_y1 = (bbox[1] * scale_y)
    new_x2 = (bbox[0] + bbox_width) * scale_x
    new_y2 = ((bbox[1] + bbox_height) * scale_y)
    return [new_x1, new_y1, new_x2, new_y2]

def get_bbox_scale(image):
    vh_w = 1440
    vh_h = 2560

    im_w = image.width
    im_h = image.height
    scale_x = im_w / vh_w
    scale_y = im_h / vh_h
    return scale_x, scale_y

def get_nodes_from_id(root, sample_id):
    ui_obj = []
    def _get_id_nodes(node):
        if node is not None:
            if node['pointer'] == sample_id:
                ui_obj.append(node)
            if 'children' in node:
                for n in node['children']:
                    _get_id_nodes(n)
    _get_id_nodes(root)
    return ui_obj[0]

def get_nodes_from_index_list(obj_id, root):
    index_list = obj_id.split('.')[1:]
    index_list = [int(i) for i in index_list]

    curr_node = root
    for index in index_list:
        print(index)
        curr_node = curr_node['children'][index]
    print(curr_node)
    return curr_node

def normalize_bbox(bbox, image):
    w = image.width
    h = image.height
    return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

def load_tap_anns(anns_dir):
    ann_path = os.path.join(anns_dir, "rico_tap_annotations_idsonly.csv")

    with open(ann_path, "r") as f:
        anns = f.readlines()[1:]
        anns = [x.strip().split(',') for x in anns]

    split_anns = defaultdict(lambda: defaultdict(str))
    for sample in anns:
        imgid_objid = '_'.join(sample[:2])
        split_anns[imgid_objid]["target"] = ("yes" if sample[1].strip() == "1" else "no")

    print("Loaded %d samples for zero shot evaluation of tappability" % len(split_anns.keys()))

    return split_anns

def load_widget_anns(anns_dir):
    ann_path = os.path.join(anns_dir, "widget_captions.csv")
    split_path = os.path.join(anns_dir, "split")
    all_split_files = glob.glob(os.path.join(split_path, "*.txt"))

    imgid_split = {}
    for split_f in all_split_files:
        s_key = split_f.split('/')[-1].split('.')[0]
        with open(split_f) as f:
            s = f.readlines()
            for sample in s:
                imgid_split[sample.strip()] = s_key

    with open(ann_path, "r") as f:
        anns = f.readlines()[1:]
        anns = [x.strip().split(',') for x in anns]

    split_anns = defaultdict(lambda: defaultdict(str))
    for sample in anns:
        if sample[0] != 'rc':
            continue
        imgid_objid = '_'.join(sample[1:3])
        split_anns[imgid_objid]["target"] = sample[-1]

    print("Loaded %d samples for zero shot evaluation of widget captioning" % len(split_anns.keys()))

    return split_anns, imgid_split

def load_screen_anns(anns_dir):
    new_anns = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(anns_dir, 'split', 'train_screens.txt')) as f:
        train_samples = [x.strip() for x in f.readlines()]
    with open(os.path.join(anns_dir, 'split', 'dev_screens.txt')) as f:
        val_samples = [x.strip() for x in f.readlines()]
    with open(os.path.join(anns_dir, 'split', 'test_screens.txt')) as f:
        test_samples = [x.strip() for x in f.readlines()]
    with open(os.path.join(anns_dir, 'screen_summaries.csv')) as f:
        all_anns = [x.strip() for x in f.readlines()]
        all_anns = all_anns[1:]

    for entry in all_anns:
        img_id = entry.split(',')[0]
        cap = ','.join(entry.split(',')[1:])
        if img_id in train_samples:
            new_anns['train'][img_id].append(cap)
        elif img_id in val_samples:
            new_anns['val'][img_id].append(cap)
        else:
            assert img_id in test_samples
            new_anns['test'][img_id].append(cap)

    for split in new_anns:
        split_formatted = []
        for sample in new_anns[split]:
            if split == 'train':
                for cap in new_anns[split][sample]:
                    curr_sample = {'image_id': sample, 'image': sample + '.jpg', 'caption': cap}
                    split_formatted.append(curr_sample)
            else:
                curr_sample = {'image_id': sample, 'image': sample + '.jpg', 'caption': new_anns[split][sample]}
                split_formatted.append(curr_sample)
        with open(os.path.join(anns_dir, split + '.json'), 'w') as f:
            json.dump(split_formatted, f)

def load_ground_anns(anns_dir, split):
    split_anns_path = os.path.join(anns_dir, 'mug_objects_v1.1.' + split + '.json')
    with open(split_anns_path) as f:
        all_jsons = json.load(f)

    instruct_path = os.path.join(anns_dir, 'mug_v1.1.' + split + '.csv')
    instruct_dict = {}
    with open(instruct_path) as f:
        instr_data = f.readlines()[1:]
        instr_data = [re.sub(',(?=[^"]*"[^"]*(?:"[^"]*"[^"]*)*$)', "", x) for x in instr_data]
        instr_data = [x.strip().strip("\"").strip(',').split(',') for x in instr_data]
    for sample in instr_data:
        key_id = '_'.join(sample[:2])
        instruct_dict[key_id] = sample[2:]

    new_dict = defaultdict(lambda: defaultdict(str))
    for sample in all_jsons:
        imgid = sample['screen_id']
        target_objid = sample['target_id']
        all_objids = sample['object_ids']
        img_tobj_id = '_'.join([imgid, target_objid])
        new_dict[img_tobj_id]['object_ids'] = all_objids
        new_dict[img_tobj_id]['instruction'] = instruct_dict[img_tobj_id]
        new_dict[img_tobj_id]['single_step'] = (False if len(instruct_dict[img_tobj_id]) > 2 else True)
    print("Loaded %d %s samples for zero shot evaluation of language grounding" % (len(new_dict.keys()), split))
    return new_dict

def get_updated_label_bbox(og_idx, bboxes):
    # sort bboxes and assign index based off of sorted order
    # to ensure no duplicate image id - target obj idx pairs
    curr_idxs = range(len(bboxes))
    idx_bbox = list(zip(curr_idxs, bboxes))
    sorted_idx_bbox = sorted(idx_bbox, key=lambda x: x[1])

    new_idx = None
    for i in range(len(sorted_idx_bbox)):
        if sorted_idx_bbox[i][0] == og_idx:
            new_idx = i
            break
    # print(og_idx)
    # print(sorted_idx_bbox)
    # print(new_idx)
    assert new_idx is not None

    sorted_just_bbox = [x[1] for x in sorted_idx_bbox]
    return new_idx, sorted_just_bbox

def load_ref_exp_anns(anns_dir, split):
    split_anns_path = os.path.join(anns_dir, split + '.tfrecord')
    raw_dataset = tf.data.TFRecordDataset(split_anns_path)

    feature_description = {
        'image/id': tf.io.FixedLenFeature([], tf.string),
        'image/ref_exp/label': tf.io.FixedLenFeature([], tf.int64),
        'image/ref_exp/text': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/num': tf.io.FixedLenFeature([], tf.float32),
        'image/view_hierarchy/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/view_hierarchy/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/view_hierarchy/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/view_hierarchy/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/view_hierarchy/class/label': tf.io.VarLenFeature(tf.int64),
        'image/view_hierarchy/class/name': tf.io.VarLenFeature(tf.string),
        'image/view_hierarchy/description': tf.io.VarLenFeature(tf.string),
        'image/view_hierarchy/id/name': tf.io.VarLenFeature(tf.string),
        'image/view_hierarchy/text': tf.io.VarLenFeature(tf.string),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)

    new_dict = defaultdict(lambda: defaultdict(str))
    img_bbox = {} 
    iters = 0
    for sample in parsed_dataset:
        iters+=1
        imgid = sample['image/id'].numpy().decode('utf-8')
        og_target_objidx = sample['image/ref_exp/label'].numpy()

        xmin = [float(x) for x in sample['image/object/bbox/xmin'].values.numpy()]
        ymin = [float(x) for x in sample['image/object/bbox/ymin'].values.numpy()]
        xmax = [float(x) for x in sample['image/object/bbox/xmax'].values.numpy()]
        ymax = [float(x) for x in sample['image/object/bbox/ymax'].values.numpy()]
        bboxes = list(zip(xmin, ymin, xmax, ymax))

        new_target_objidx, sorted_bboxes = get_updated_label_bbox(og_target_objidx, bboxes)
        img_tobj_id = '_'.join([imgid, str(new_target_objidx)])
        if imgid not in img_bbox:
            img_bbox[imgid] = sorted_bboxes
        else:
            if img_bbox[imgid] != sorted_bboxes:
                print(imgid)
                print(img_bbox[imgid])
                print(sorted_bboxes)
                break

        # assert img_tobj_id not in new_dict
        new_dict[img_tobj_id]['target_idx'] = new_target_objidx
        new_dict[img_tobj_id]['screen_norm_bboxes'] = sorted_bboxes
        new_dict[img_tobj_id]['ref_exp'] = sample['image/ref_exp/text'].numpy().decode('utf-8')
    
    return new_dict

def load_view_hierarchy(image_paths, image_dir):
    loaded = []
    for image_path in image_paths:
        json_path = os.path.join(image_dir, image_path + '.json')
        with open(json_path) as f:
            d = json.load(f)
        loaded.append(d['activity']['root'])
    return loaded

def load_images(image_ids, image_dir):
    loaded = []
    for image in image_ids:
        image_path = os.path.join(image_dir, image + '.jpg')
        raw_image = Image.open(image_path).convert('RGB')
        loaded.append(raw_image)
    return loaded

def __main__():
    global args
    args = parser.parse_args()

    if args.task == 'tappability':
        target_anns = load_tap_anns(args.input_anns_dir)
    elif args.task  == 'widget_caption':
        target_anns, split_id = load_widget_anns(args.input_anns_dir)
        print(len(target_anns))
    elif args.task == 'language_ground':
        target_anns = load_ground_anns(args.input_anns_dir, args.data_split)
        all_objs_ids = [target_anns[x]['object_ids'] for x in target_anns]
    elif args.task == 'ref_exp':
        target_anns = load_ref_exp_anns(args.input_anns_dir, args.data_split)

        new_fp = args.data_split + '_modified.json'
        with open(os.path.join(args.input_anns_dir, new_fp), "w") as f:
            json.dump(target_anns, f)
        return
    elif args.task == 'screen_caption':
        load_screen_anns(args.input_anns_dir)
        return

    assert args.task != 'ref_exp' and args.task != 'screen_caption'
    image_ids = [x.split('_')[0] for x in target_anns]
    obj_ids = [x.split('_')[1] for x in target_anns]
    

    vh_bbox = []
    screen_bbox = []
    normalized_bbox = []
    split_info = [] # only needed for language ground
    for i in range(len(target_anns)):
        if i % 1000 == 0:
            print(i)
        image = load_images([image_ids[i]], args.input_data_dir)[0]
        vh = load_view_hierarchy([image_ids[i]], args.input_data_dir)[0]
        
        if args.task == 'tappability':
            target_node = get_nodes_from_id(vh, obj_ids[i])
        elif args.task == 'language_ground':
            target_node = get_nodes_from_id(vh, obj_ids[i])
            all_nodes = [get_nodes_from_id(vh, elem) for elem in all_objs_ids[i]]
        elif args.task == 'widget_caption':
            target_node = get_nodes_from_index_list(obj_ids[i], vh)
            split_info.append(split_id[image_ids[i]])

        if args.task != 'language_ground':
            vh_bb = target_node['bounds']
            sx, sy = get_bbox_scale(image)

            screen_bb = convert_view_to_screen_dims(vh_bb, sx, sy)
            screen_norm_bb = normalize_bbox(screen_bb, image)

            vh_bb = ' '.join([str(x) for x in vh_bb])
            screen_bb = ' '.join([str(x) for x in screen_bb])
            screen_norm_bb = ' '.join([str(x) for x in screen_norm_bb])

            vh_bbox.append(vh_bb)
            screen_bbox.append(screen_bb)
            normalized_bbox.append(screen_norm_bb)
        else:
            objid_bboxes = defaultdict(lambda: defaultdict(str))
            sx, sy = get_bbox_scale(image)
            for idx in range(len(all_objs_ids[i])):
                node = all_nodes[idx]
                vh_bb = node['bounds']
                screen_bb = convert_view_to_screen_dims(vh_bb, sx, sy)
                screen_norm_bb = normalize_bbox(screen_bb, image)
                
                objid_bboxes[all_objs_ids[i][idx]]['vh_bbox'] = vh_bb
                objid_bboxes[all_objs_ids[i][idx]]['screen_bbox'] = screen_bb
                objid_bboxes[all_objs_ids[i][idx]]['screen_norm_bbox'] = screen_norm_bb
            sample_key = '_'.join([image_ids[i], obj_ids[i]])
            target_anns[sample_key]['object_bboxes'] = objid_bboxes

    if args.task == 'tappability':
        old_fp = "rico_tap_annotations_idsonly.csv"
        new_fp = "rico_tap_annotations_modified.csv"
        to_write = zip(vh_bbox, screen_bbox, normalized_bbox)
        to_write = [','.join(x) for x in to_write]
        to_write = ['vh_bbox,screen_bbox,norm_screen_bbox'] + to_write
    elif args.task == 'widget_caption':
        old_fp = "widget_captions.csv"
        new_fp = "widget_captions_modified.csv"
        to_write = zip(split_info,vh_bbox, screen_bbox, normalized_bbox)
        to_write = [','.join(x) for x in to_write]
        to_write = ['split,vh_bbox,screen_bbox,norm_screen_bbox'] + to_write
    
    if args.task != 'language_ground' and args.task != 'ref_exp':
        # CSV format
        with open(os.path.join(args.input_anns_dir, old_fp)) as f:
            original_anns = f.readlines()
            original_anns = [x.strip() for x in original_anns]

        new_anns = zip(original_anns, to_write)
        new_anns = [','.join(x) for x in new_anns]
        full = '\n'.join(new_anns)

        with open(os.path.join(args.input_anns_dir, new_fp), "w") as f:
            f.write(full)
    elif args.task == 'language_ground':
        # JSON format
        new_fp = "mug_objects_v1_modified." + args.data_split + '.json'
        with open(os.path.join(args.input_anns_dir, new_fp), "w") as f:
            json.dump(target_anns, f)

__main__()