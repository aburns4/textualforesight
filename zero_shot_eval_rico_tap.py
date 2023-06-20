import argparse
import json
import math
import os
import time
import torch

from collections import defaultdict
from PIL import Image

from lavis.models import load_model_and_preprocess

parser = argparse.ArgumentParser('BLIP-2 zero shot evaluation')
parser.add_argument('--batch_size',
                    type=int,
                    default=10,
                    help='batch size for evaluation')
parser.add_argument('--input_data_dir',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/taperception",
                    help='specify path to evaluation dataset')
parser.add_argument('--input_image_dir',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/combined",
                    help='specify path to raw images')
parser.add_argument('--data_split',
                    type=str,
                    default="test",
                    help='specify which dataset split to evaluate')       
parser.add_argument('--task',
                    type=str,
                    default="tappability",
                    help='specify which downstream task to evaluate')             
parser.add_argument('--model_output_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/zero_shot_outputs",
                    help='specify path to output caption dir')

def convert_view_to_screen_dims(ui_bbox, scale_x, scale_y):
    # need to convert to screen localization
    bbox_width = bbox.x2 - bbox.x1
    bbox_height = bbox.y2 - bbox.y1
    new_x1 = bbox.x1 * scale_x
    new_y1 = (bbox.y1 * scale_y)
    new_x2 = (bbox.x1 + bbox_width) * scale_x
    new_y2 = ((bbox.y1 + bbox_height) * scale_y)
    return [new_x1, new_y1, new_x2, new_y2]

def get_bbox_scale(image, root):
    if root['bounds'][:2] != [0, 0]:
        raise ValueError('Root bounds [%d, %d, %d, %d]do not begin at upper left corner coordinate' % root['bounds'])
    _, _, vh_w, vh_y = root['bounds']
    
    im_w = image.width
    im_h = image.height
    scale_x = im_w / vh_w
    scale_y = im_h / vh_y # (im_h + 65)
    return scale_x, scale_y

def get_nodes_from_id(root, sample_id):
    def _get_id_nodes(node):
        if node is not None:
            if 'children' not in node:
                if node['pointer'] == sample_id:
                    return node
            else:
                for n in node['children']:
                    _get_id_nodes(n)
    _get_id_nodes(root)

def get_nodes_from_index_list(sample, root):
    index_list = sample['nodeId'].split('.')[1:]
    index_list = [int(i) for i in index_list]

    curr_node = root
    for index in index_list:
        curr_node = curr_node['children'][index]
    return curr_node['bounds']

def get_bbox_widget(sample, root, scale_x, scale_y, sample_id=None, use_index=True, use_pointer=False):
    if use_index:
        bbox = get_nodes_from_index_list(sample, root)
    else:
        bbox = get_nodes_from_id(root, sample_id)
    return convert_view_to_screen_dims(bbox, scale_x, scale_y)

def illustrate_bbox(image, bbox, img_name, save_dir):
    temp = image.copy()
    temp_draw = ImageDraw.Draw(img)
    temp_draw.rectangle(bbox)
    temp_draw.save(os.path.join(save_dir, img_name) + '_ui_bboxes.jpg')

def normalize_bbox(bbox, image):
    w = image.width
    h = image.height
    return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

def load_split_tap(split, anns_dir):
    ann_path = os.path.join(anns_dir, "rico_tap_annotations_modified.csv")

    with open(ann_path, "r") as f:
        anns = f.readlines()[1:]
        anns = [x.split(',') for x in anns]

    split_anns = defaultdict(lambda: defaultdict(str))
    for sample in anns:
        if sample[4] == split:
            imgid_objid = '_'.join(sample[:2])
            split_anns[imgid_objid]["target"] = ("yes" if sample[2].strip() == "1" else "no")
            split_anns[imgid_objid]["normalized_bbox"] = [float(x) for x in sample[-1].split(' ')]
    print("Loaded %d %s samples for zero shot evaluation of tappability" % (len(split_anns.keys()), split))

    return split_anns

def load_split_widget(split, anns_dir):
    ann_path = os.path.join(anns_dir, "widget_captions_modified.csv")

    with open(ann_path, "r") as f:
        anns = f.readlines()[1:]
        anns = [x.split(',') for x in anns]

    split_anns = defaultdict(lambda: defaultdict(str))
    for sample in anns:
        if sample[4] == split:
            imgid_objid = '_'.join(sample[1:3])
            split_anns[imgid_objid]["target"] = sample[-5].split('|')
            split_anns[imgid_objid]["normalized_bbox"] = [float(x) for x in sample[-1].split(' ')]
    print("Loaded %d %s samples for zero shot evaluation of widget captioning" % (len(split_anns.keys()), split))

    return split_anns

def load_split_ground(split, anns_dir):
    ann_path = os.path.join(anns_dir, 'mug_objects_v1_modified.' + split + '.json')
    with open(ann_path) as f:
        split_anns = json.load(f)
    print("Loaded %d %s samples for zero shot evaluation of language grounding" % (len(split_anns.keys()), split))
    return split_anns

def load_images_and_process(image_obj_ids, image_dir, vis_preprocessor, device):
    processed = []
    image_ids = [x.split('_')[0] for x in image_obj_ids]
    for image in image_ids:
        image_path = os.path.join(image_dir, image + '.jpg')
        raw_image = Image.open(image_path).convert('RGB')
        processed_image = vis_preprocessor["eval"](raw_image).to(device)
        processed.append(processed_image)
    
    processed = torch.stack(processed)
    # processed.type(torch.float16)
    return processed

def load_images(image_obj_ids, image_dir):
    loaded = []
    image_ids = [x.split('_')[0] for x in image_obj_ids]
    for image in image_ids:
        image_path = os.path.join(image_dir, image + '.jpg')
        raw_image = Image.open(image_path).convert('RGB')
        loaded.append(raw_image)
    return loaded

def generate_zero_shot_tap(images, image_obj_ids, bboxes, model, target_dict):
    images.cuda()
    formatted_bboxes = ['%.2f, %.2f, %.2f, %.2f' % (bb[0], bb[1], bb[2], bb[3]) for bb in bboxes]
    p1 = ["Question: Is the app element located at [%s] tappable? Answer:" % bb for bb in formatted_bboxes]
    p2 = ["Question: Is the UI object found at [%s] tappable? Answer:" % bb for bb in formatted_bboxes]
    p3 = ["Question: Can the screen object found at [%s] be tapped on? Answer:" % bb for bb in formatted_bboxes]

    prompt1 = model.generate({"image": images, "prompt": p1})
    prompt2 = model.generate({"image": images, "prompt": p2})
    prompt3 = model.generate({"image": images, "prompt": p3})
    model_outputs = zip(image_obj_ids, prompt1, prompt2, prompt3)
    for x in model_outputs:
        target_dict[x[0]].update({"prompt1": x[1],
                                  "prompt2": x[2],
                                  "prompt3": x[3]})
    return target_dict

def generate_zero_shot_widget(images, image_obj_ids, bboxes, model, target_dict):
    images.cuda()
    formatted_bboxes = ['%.2f, %.2f, %.2f, %.2f' % (bb[0], bb[1], bb[2], bb[3]) for bb in bboxes]
    
    p1 = ["Question: What is the description for the app element located at [%s]? Answer:" % bb for bb in formatted_bboxes]
    p2 = ["Question: What caption explains the UI object found at [%s]? Answer:" % bb for bb in formatted_bboxes]
    p3 = ["Question: What describes the functionality of the UI object found at [%s]? Answer:" % bb for bb in formatted_bboxes]
    
    prompt1 = model.generate({"image": images, "prompt": p1})
    prompt2 = model.generate({"image": images, "prompt": p2})
    prompt3 = model.generate({"image": images, "prompt": p3})
    model_outputs = zip(image_obj_ids, prompt1, prompt2, prompt3)
    for x in model_outputs:
        target_dict[x[0]].update({"prompt1": x[1],
                                  "prompt2": x[2],
                                  "prompt3": x[3]})
    return target_dict

def generate_zero_shot_ground(images, image_obj_ids, bboxes, bbox_ids, model, target_dict, batch_size=16, dsplit='test'):
    images.cuda()
    instruction = target_dict[image_obj_ids[0]]['instruction'][0]
    
    target_obj_id = image_obj_ids[0].split('_')[1]
    tidx = bbox_ids[0].index(target_obj_id)
    target_dict[image_obj_ids[0]].update({"target_idx": tidx})

    formatted_bboxes = ['%.2f, %.2f, %.2f, %.2f' % (bb[0], bb[1], bb[2], bb[3]) for bb in bboxes[0]]

    p1 = ["Question: Does the command '%s' match the UI object found at [%s]? Answer: " % (instruction, sample) for sample in formatted_bboxes]
    p2 = ["Question: Does the app element located at [%s] complete the command '%s'? Answer:" % (instruction, sample) for sample in formatted_bboxes]

    max_yes_logit = -10000   
    max_yes_bbox_id = None
    max_yes_bbox_idx = -1
    bbox_idx = 0
    for i in range(0, len(p1), batch_size):
        bbox_batch = p1[i:i+batch_size]
        image_batch = images.expand(len(bbox_batch), -1, -1, -1)
        logs = model.generate_yes_logits({"image": image_batch, "prompt": bbox_batch}, scores=True, return_dict=True)
        max_of_mini = torch.argmax(logs)
        if logs[max_of_mini] > max_yes_logit:
            max_yes_logit = logs[max_of_mini]
            max_yes_bbox_id = bbox_ids[0][bbox_idx + max_of_mini]
            max_yes_bbox_idx = bbox_idx + max_of_mini
        bbox_idx += len(bbox_batch)
    target_dict[image_obj_ids[0]].update({"prompt1": max_yes_bbox_id,
                                          "prompt1_idx": max_yes_bbox_idx.cpu().numpy().item()})   

    max_yes_logit2 = -10000   
    max_yes_bbox_id2 = None
    max_yes_bbox_idx2 = -1
    bbox_idx = 0
    for i in range(0, len(p2), batch_size):
        bbox_batch = p2[i:i+batch_size]
        image_batch = images.expand(len(bbox_batch), -1, -1, -1)
        logs = model.generate_yes_logits({"image": image_batch, "prompt": bbox_batch}, scores=True, return_dict=True)
        max_of_mini = torch.argmax(logs)
        if logs[max_of_mini] > max_yes_logit2:
            max_yes_logit2 = logs[max_of_mini]
            max_yes_bbox_id2 = bbox_ids[0][bbox_idx + max_of_mini]
            max_yes_bbox_idx2 = bbox_idx + max_of_mini
        bbox_idx += len(bbox_batch)
    target_dict[image_obj_ids[0]].update({"prompt2": max_yes_bbox_id2,
                                          "prompt2_idx": max_yes_bbox_idx2.cpu().numpy().item()})   

    with open('/projectnb2/ivc-ml/aburns4/LAVIS/zero_shot_outputs/zero_shot_language_ground_intermediate_' + dsplit +'.txt', 'a') as f:
        f.write(','.join([image_obj_ids[0], str(tidx), str(max_yes_bbox_idx.cpu().numpy()), str(max_yes_bbox_idx2.cpu().numpy())]) + '\n')

    return target_dict

def __main__():
    global args
    args = parser.parse_args()
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size
    if args.task == 'tappability':
        target_anns = load_split_tap(args.data_split, args.input_data_dir)
    elif args.task == 'widget_caption':
        target_anns = load_split_widget(args.data_split, args.input_data_dir)
    elif args.task == 'language_ground':
        target_anns = load_split_ground(args.data_split, args.input_data_dir)

    # we associate a model with its preprocessors to make it easier for inference.
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
    )

    split_img_obj_ids = list(target_anns.keys())
    num_batches = math.ceil(len(split_img_obj_ids) / batch_size)
    print('Number of batches to evaluate %d' % num_batches)
    batched_ids = [split_img_obj_ids[i:i+batch_size] for i in range(0, len(split_img_obj_ids), batch_size)] # 
    if args.task != 'language_ground':
        batched_bboxes = [[target_anns[sample_id]['normalized_bbox'] for sample_id in batch] for batch in batched_ids]
    else:
        batched_bboxes = [[[target_anns[sample_id]['object_bboxes'][node_id]['screen_norm_bbox'] for node_id in target_anns[sample_id]['object_bboxes']] for sample_id in batch] for batch in batched_ids]
        batched_bbox_ids = [[[node_id for node_id in target_anns[sample_id]['object_bboxes']] for sample_id in batch] for batch in batched_ids]
    batched_images = [load_images_and_process(batch, args.input_image_dir, vis_processors, device) for batch in batched_ids]
    
    processed_batches = 0
    for img_batch, id_batch, bbox_batch in zip(batched_images, batched_ids, batched_bboxes):
        if args.task == 'tappability':
            target_anns = generate_zero_shot_tap(img_batch, id_batch, bbox_batch, model, target_anns)
        elif args.task == 'widget_caption':
            target_anns = generate_zero_shot_widget(img_batch, id_batch, bbox_batch, model, target_anns)
        elif args.task == 'language_ground':
            curr_batch_all_bbox_ids = batched_bbox_ids[processed_batches]
            target_anns = generate_zero_shot_ground(img_batch, id_batch, bbox_batch, curr_batch_all_bbox_ids, model, target_anns, dsplit=args.data_split)

        if processed_batches % args.batch_size == 0:
            print('Batch %d complete...' % processed_batches)
        processed_batches += 1

    if args.task == 'tappability':
        new_fp = "tappability_zero_shot_" + args.data_split + ".json"
    elif args.task == 'widget_caption':
        new_fp = "widget_captioning_zero_shot_" + args.data_split + ".json"
    elif args.task == 'language_ground':
        new_fp = "language_grounding_zero_shot_" + args.data_split + ".json"

    with open(os.path.join(args.model_output_path, new_fp), "w") as f:
        json.dump(target_anns, f)


start_time = time.time()
__main__()
end_time = time.time()
seconds = end_time - start_time
minutes = seconds / 60
hours = minutes / 60
print('Time to evaluate %.2f seconds / %.2f minutes / %.2f / hours' % (seconds, minutes, hours))