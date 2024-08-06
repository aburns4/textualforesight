import json
import glob
import time
import re
import os
import argparse

from collections import defaultdict
from PIL import Image

GENERIC_WORDS = ['action', 'bar', 'menu', 'title', 'and', 'ans', 'app', 'icon', 'name',
                 'arg', 'background', 'element', 'btn', 'but', 'bottom', 'button', 'content',
                 'desc', 'text', 'item', 'empty', 'fab', 'image', 'grid', 'header', 'img',
                 'imgfile', 'lbutton', 'label', 'letter', 'list', 'view', 'pic', 'placeholder',
                 'random', 'row', 'single', 'raw', 'small', 'large', 'sub', 'template', 'navbar', 
                 'banner', 'test', 'textinput', 'error', 'texto', 'todo', 'toolbar', 'tool', 'track',
                 'txt', 'unknown', 'stub', 'web', 'left', 'right', 'tlb', 'nan', 'page', 'feature',
                 'menugrid', 'picture', 'tabs', 'number', 'node', 'iconimage', 'entity', 'webview',
                 'heading', 'logo', 'tbl', 'tab', 'primary', 'footer']

parser = argparse.ArgumentParser('Create pretrain annotations for Spotlight objective')
parser.add_argument('--dataset',
                    type=str,
                    default="motif",
                    help='specify which dataset to process')
parser.add_argument('--start_range',
                    type=int,
                    default=0,
                    help='specify start of range to process')
parser.add_argument('--end_range',
                    type=int,
                    default=5000,
                    help='specify end of range to process')
parser.add_argument('--index',
                    type=int,
                    default=0,
                    help='specify split index of current dataset to process')

def normalize_bbox(bbox, image):
    w = image.width
    h = image.height
    return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

def convert_view_to_screen_dims(bbox, scale_x, scale_y, adjustment=0):
    # need to convert to screen localization
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    new_x1 = bbox[0] * scale_x
    new_y1 = (bbox[1] * scale_y) - adjustment
    new_x2 = (bbox[0] + bbox_width) * scale_x
    new_y2 = ((bbox[1] + bbox_height) * scale_y) - adjustment
    return [new_x1, new_y1, new_x2, new_y2]

def get_bbox_scale(image_path,
                   root_path='/projectnb2/ivc-ml/aburns4/combined',
                   vh_w = 1440, vh_h = 2560, adjustment = 0):
    full_im_path = (os.path.join(root_path, image_path) if image_path not in root_path else root_path)
    image = Image.open(full_im_path).convert('RGB')

    im_w = image.width
    im_h = image.height

    scale_x = im_w / vh_w
    scale_y = (im_h + adjustment) / vh_h
    return scale_x, scale_y, image

def get_motif_vh_dims(vh_bounds, fp, exceptions="widget_exception_dims.json"):
    trace = fp.split('/')[-3]
    with open(exceptions) as f:
        widget_exceptions = json.load(f)

    if trace in widget_exceptions:
        vh_dims = widget_exceptions[trace]
        vh_dims = [int(x) for x in vh_dims]
        return vh_dims
    elif vh_bounds[0] == vh_bounds[1] == 0:
        return vh_bounds[2:]
    else:
        return None

def is_good_bbox(img_bbox, img):
    im_w, im_h = img.size
    x1, y1, x2, y2 = img_bbox

    ok_x_wrt_img = (0 <= x1 <= im_w and
                    0 <= x2 <= im_w and
                    x1 < x2)
    ok_y_wrt_img = (0 <= y1 <= im_h and
                    0 <= y2 <= im_h and 
                    y1 < y2)

    if not (ok_x_wrt_img and ok_y_wrt_img):
        return False

    ui_crop = img.copy().crop(img_bbox)
    extrema = ui_crop.convert("L").getextrema()
    if extrema is None:
        return False
    min_px, max_px = extrema
    one_value = (min_px == max_px)
    if one_value:
        return False
    return True

# global data_dict
class UI:
    def __init__(self, generic_words, dataset):
        self.dataset = dataset
        self.leaves = []
        self.image_ids = []
        self.app_ids = []
        self.vh_dims = []
        self.vh_paths = []

        self.clean_leaves = []
        self.clean_image_ids = []
        self.clean_app_ids = []
        self.clean_vh_dims = []
        self.clean_vh_paths = []
        self.clean_leaves_text = []
        
        self.generic_words = generic_words

    def recurse(self, node, image_id, app_id, root_bounds=None, vh_path=None):
        if node:
            if 'children' in node and len(node['children']) > 0:
                for child in node['children']:
                    # changed this to allow non leaf nodes
                    # which will later be further processed
                    self.leaves = self.leaves + [node]
                    self.image_ids = self.image_ids + [image_id]
                    self.app_ids = self.app_ids + [app_id]
                    if root_bounds is not None:
                        assert self.dataset == 'motif'
                        self.vh_dims = self.vh_dims + [root_bounds]
                        self.vh_paths = self.vh_paths + [vh_path]

                    self.recurse(child, image_id, app_id, root_bounds, vh_path)
            else:
                self.leaves = self.leaves + [node]
                self.image_ids = self.image_ids + [image_id]
                self.app_ids = self.app_ids + [app_id]
                if root_bounds is not None:
                    assert self.dataset == 'motif'
                    self.vh_dims = self.vh_dims + [root_bounds]
                    self.vh_paths = self.vh_paths + [vh_path]

    def filter_leaves(self):
        bad_classes = ["android.view.View", "android.widget.Image"]
        for i in range(len(self.leaves)):
            lv = self.leaves[i]
            im_id = self.image_ids[i]
            app_id = self.app_ids[i]
            # remove invisible objects
            if self.dataset != "longitudinal":
                visible = lv['visible-to-user']
                class_name = lv["class"]
            else:
                visible = (False if 'isVisibleToUser' not in lv else lv['isVisibleToUser'])
                if 'isVisibleToUser' not in lv:
                    print(im_id, app_id)
                class_name = lv["className"]

            if visible and class_name not in bad_classes:
                self.clean_leaves = self.clean_leaves + [lv]
                self.clean_image_ids = self.clean_image_ids + [im_id]
                self.clean_app_ids = self.clean_app_ids + [app_id]
                if self.dataset == 'motif':
                    vh_id = self.vh_paths[i]
                    vh_dims = self.vh_dims[i]
                    self.clean_vh_paths = self.clean_vh_paths + [vh_id]
                    self.clean_vh_dims = self.clean_vh_dims + [vh_dims]

    def get_text_fields(self, lv):
        text = ('' if 'text' not in lv else lv['text'])
        if args.dataset != 'longitudinal':
            content = ('' if 'content-desc' not in lv else lv['content-desc'])
            if isinstance(content, list):
                assert len(content) == 1
                content = content[0]
            content = ('' if content is None else content)
        else:
            content = ('' if 'contentDesc' not in lv else lv['contentDesc'])
            if isinstance(content, list):
                assert len(content) == 1
                content = content[0]
            content = ('' if content is None else content)
        
        raw_fields = [text, content] # decided to not include rid
        return raw_fields

    def filter_text(self):
        # select text, content_description, and resource_id fields
        for lv in self.clean_leaves:
            raw_fields = self.get_text_fields(lv)
            
            to_clean = []
            # filter out object text consisting of only generic words
            # filter out object text that is unicode 
            # or nonalphabetical TODO: not sure if I want alphabet or alpha numeric
            # or only a single char
            # or URL
            for field in raw_fields:
                field = field.replace("\n", "")
                toks = field.strip(" ").split(' ')
                only_gen = (len(set(toks).difference(set(self.generic_words))) == 0)
                single_or_empty_char = (len(field) <= 1)
                is_url = (len(toks) == 1 and 'http' in field)
                transformed_field = field.encode('unicode-escape').decode('ascii')
                is_alpha = all(x.isalpha() or x.isspace() for x in transformed_field) # TODO: I think I fixed this, not sure if I want to allow numbers though
                if (not only_gen) and (not single_or_empty_char) and (not is_url) and is_alpha:
                    to_clean.append(field)
            
            cleaned_fields = []
            # clean by replacing continuous spaces and underscores with a single space
            # lowercase text - I changed this to not be lower for GPT purposes
            for field in to_clean:
                # cln_field = field.lower()
                cln_field = field.replace("_", " ")
                cln_field = re.sub(' +', ' ', cln_field)
                cleaned_fields.append(cln_field) # also this was field and not cln_field, so that was a bug
            self.clean_leaves_text = self.clean_leaves_text + [cleaned_fields]
   
    def remove_infrequent(self, threshold=5):
        # remove text that occurs fewer than 5 times
        all_entries = [y for x in self.clean_leaves_text for y in x]
        counts = defaultdict(int)
        for entry in all_entries:
            counts[entry.lower()] += 1
        
        final_kept = []
        for leaf_texts in self.clean_leaves_text:
            curr_final_texts = []
            for leaf_t in leaf_texts:
                if counts[leaf_t.lower()] >= threshold:
                    curr_final_texts.append(leaf_t)
            final_kept.append(curr_final_texts)
        self.final_leaves_text = final_kept

    def create_pretrain_samples(self, idx):
        # keys: image, image_id, bbox_id, caption
        samples = defaultdict(lambda: defaultdict(list))
        removed_by_bbox = 0
        assert len(self.final_leaves_text) == len(self.clean_image_ids)  == len(self.clean_app_ids) == len(self.clean_leaves)
        print('within pretrain samples')

        i = 0
        inputs = zip(self.final_leaves_text, self.clean_leaves, self.clean_image_ids, self.clean_app_ids)
        for text, node, img, app in inputs:
            if i % 1000 == 0:
                print(i)
            vh_bb = node['bounds']

            if self.dataset == 'motif':
                assert len(self.final_leaves_text) == len(self.clean_vh_dims) == len(self.clean_vh_paths)
                vh_w, vh_h = self.clean_vh_dims[i]
                screen_path = self.clean_vh_paths[i].replace('view_hierarchies', 'screens')[6:]
                try:
                    sx, sy, loaded_image = get_bbox_scale(
                    screen_path, 
                    root_path="/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/motif",
                    vh_w=vh_w, vh_h=vh_h, adjustment=65)
                except:
                    print('bbox scale error')
                    i+=1
                    continue
                screen_bb = convert_view_to_screen_dims(vh_bb, sx, sy, adjustment=65)
                screen_norm_bb = normalize_bbox(screen_bb, loaded_image)
            elif self.dataset == 'rico':
                sx, sy, loaded_image = get_bbox_scale(img)
                screen_bb = convert_view_to_screen_dims(vh_bb, sx, sy)
                screen_norm_bb = normalize_bbox(screen_bb, loaded_image)
                screen_path = img
            elif self.dataset == 'longitudinal':
                screen_bb = [node["rect"]["left"],
                             node["rect"]["top"],
                             node["rect"]["right"],
                             node["rect"]["bottom"]]
                screen_norm_bb = [screen_bb[0] / node["screenWidth"],
                                  screen_bb[1] / node["screenHeight"],
                                  screen_bb[2] / node["screenWidth"],
                                  screen_bb[3] / node["screenHeight"]]
                screen_path = os.path.join(app, img + '.png')
                if not os.path.exists(os.path.join("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/longitudinal", screen_path)):
                    print("Missing image: %s" % screen_path)
                    continue
                loaded_image = Image.open(
                    os.path.join("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/longitudinal", screen_path)).convert('RGB')
            else:
                raise ValueError("Dataset %s not supported. Please choose amongst [rico, longitudinal, motif]" % self.dataset)

            if not is_good_bbox(screen_bb, loaded_image):
                removed_by_bbox += 1
                i+=1
                continue

            img_id = img.split('.')[0]

            for e in text:
                if e not in samples[app][img_id]:
                    samples[app][img_id].append(e)
            # samples[app][img_id].extend(text)
            i+=1

        print("Samples removed due to bbox conditions %d" % removed_by_bbox)
        print("Number of apps in this subset processed %d" % len(samples.keys()))
        all_imgs = sum([len(samples[app].keys()) for app in samples])
        print("Number of images in this subset processed %d" % all_imgs)
        output_intermediate_dir = os.path.join('gpt_jsons', self.dataset, "elements_final")
        if not os.path.exists(output_intermediate_dir):
            os.makedirs(output_intermediate_dir)
        with open(os.path.join(output_intermediate_dir, str(idx) + '_pretrain.json'), 'w') as f:
            json.dump(samples, f)
        # to_write = ["[*]".join((app, imid, " | ".join(samples[app][imid]))) for app in samples for imid in samples[app]]
        # with open(self.dataset + "_reattempt_leaves_only.txt", "w") as f:
        #     f.write("\n".join(to_write))

def finetune_ids(json_paths):
    all_ids = []
    for jp in json_paths:
        with open(jp) as f:
            data = json.load(f)
        for sample in data:
            all_ids.append(sample['image'])
    return set(all_ids)

def get_rico_files():
    all_rico_ids = glob.glob('/projectnb2/ivc-ml/aburns4/combined/*.jpg')
    all_rico_ids = set([x.split('/')[-1] for x in all_rico_ids])

    rico_root = '/projectnb2/ivc-ml/aburns4/combined/'
    root_path = '/projectnb2/ivc-ml/aburns4/'
    ann_paths = ['mug/eval_1.json', 'mug/test_1.json', 'mug/train_1.json',
                'screen2words/val.json', 'screen2words/test.json', 'screen2words/train.json',
                'widget-caption/dev.json', 'widget-caption/test.json', 'widget-caption/train.json',
                'taperception/eval.json', 'taperception/test.json', 'taperception/train.json',]
    json_paths = [root_path + x for x in ann_paths]

    ann_rico_ids = finetune_ids(json_paths)

    for_pretrain = list(all_rico_ids.difference(ann_rico_ids))
    return for_pretrain, rico_root

def get_motif_files():
    pretrain_traces = glob.glob('motif/*/*/*')
    set_traces = [(fp, '/'.join(fp.split('/')[-2:])) for fp in pretrain_traces]
    dict_traces = defaultdict(list)

    for entry in set_traces:
        dict_traces[entry[1]].append(entry[0])
    for_pretrain = [glob.glob(x[0] + '/view_hierarchies/*') for x in dict_traces.values()]
    for_pretrain = [y for x in for_pretrain for y in x]

    return for_pretrain

def get_longitudinal_files():
    pretrain_traces = glob.glob('longitudinal/*/*.json')
    filtered = []
    for x in pretrain_traces:
        if "graph.json" not in x:
            filtered.append(x)
    return filtered

def __main__():
    start_time = time.time()
    global args
    args = parser.parse_args()
    error_count = 0
    bounds_issues = 0
    assert args.start_range >= 0 and args.end_range >= 0 and (args.start_range < args.end_range)
    print((args.start_range, args.end_range, args.index))
    if args.dataset == 'rico':
        for_pretrain, dataset_root = get_rico_files()
    elif args.dataset == 'motif':
        for_pretrain = get_motif_files()
    elif args.dataset == 'longitudinal':
        for_pretrain = get_longitudinal_files()
    else:
        raise ValueError("Dataset %s not supported. Please choose amongst [rico, motif]" % args.dataset)
    
    print(len(for_pretrain))
    ui_class = UI(GENERIC_WORDS, args.dataset)
    for i in range(args.start_range, min(args.end_range, len(for_pretrain))):
        if i % 1000 == 0:
            print(i)
        sample = for_pretrain[i]
        if args.dataset == 'rico':
            vh_path = dataset_root + sample
            vh_path = vh_path.replace('.jpg', '.json')
            img_id = sample
        else:
            vh_path = for_pretrain[i]
            img_id = sample.split('/')[-1].split('.')[0]
        try:
            with open(vh_path) as f:
                ui = json.load(f)
                if 'activity' in ui:
                    ui_root = ui['activity']['root']
                else:
                    ui_root = ui
        except:
            error_count += 1
            continue

        if args.dataset == 'rico':
            ui_app_name = ui['activity_name'].split('/')[0]
            ui_class.recurse(ui_root, img_id, ui_app_name)
        elif args.dataset == 'motif':
            ui_app_name = vh_path.split('/')[-4]
            bounds = get_motif_vh_dims(ui_root['bounds'], vh_path)
            if bounds is not None:
                ui_class.recurse(ui_root, img_id, ui_app_name, bounds, vh_path)
            else:
                # can't make use of this screen
                bounds_issues += 1
                continue
        elif args.dataset == 'longitudinal':
            ui_app_name = vh_path.split('/')[-2]
            ui_class.recurse(ui_root, img_id, ui_app_name)

    ui_class.filter_leaves()
    ui_class.filter_text()
    ui_class.remove_infrequent(threshold=1)
    print("Number of JSON errors %d" % error_count)
    print("Number of bounds errors %d" % bounds_issues)

    ui_class.create_pretrain_samples(args.index)
    
    end_time = time.time()
    seconds = end_time - start_time
    minutes = seconds / 60
    hours = minutes / 60
    print('Time to process elements of %s: %.2f seconds / %.2f minutes / %.2f / hours' % (args.dataset, seconds, minutes, hours))

__main__()