import argparse
import json
import glob
import os

from collections import defaultdict

parser = argparse.ArgumentParser('')
parser.add_argument('--input_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/gpt3_5_captions/*",
                    help='path to read gpt captions from')
parser.add_argument('--dataset_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/motif/elements_final/*",
                    help='path to write gpt captions to')

NUM_JSONS_TO_STORE = {"aitw": 20,
                      "motif": 10,
                      "longitudinal": 10}

GENERIC_WORDS = ['action', 'bar', 'menu', 'title', 'and', 'ans', 'app', 'icon', 'name',
                 'arg', 'background', 'element', 'btn', 'but', 'bottom', 'button', 'content',
                 'desc', 'text', 'item', 'empty', 'fab', 'image', 'grid', 'header', 'img',
                 'imgfile', 'lbutton', 'label', 'letter', 'list', 'view', 'pic', 'placeholder',
                 'random', 'row', 'single', 'raw', 'small', 'large', 'sub', 'template', 'navbar', 
                 'banner', 'test', 'textinput', 'error', 'texto', 'todo', 'toolbar', 'tool', 'track',
                 'txt', 'unknown', 'stub', 'web', 'left', 'right', 'tlb', 'nan', 'page', 'feature',
                 'menugrid', 'picture', 'tabs', 'number', 'node', 'iconimage', 'entity', 'webview',
                 'heading', 'logo', 'tbl', 'tab', 'primary', 'footer']

def is_good_ocr(field):
    toks = field.split(' ')
    only_gen = (len(set(toks).difference(set(GENERIC_WORDS))) == 0)
    single_or_empty_char = (len(field) <= 1)
    is_url = (len(toks) == 1 and 'http' in field)
    transformed_field = field.encode('unicode-escape').decode('ascii')
    is_alpha = all(x.isalpha() or x.isspace() for x in transformed_field)
    if (not only_gen) and (not single_or_empty_char):
        return True
    return False

def format_elem_list(elem_list, dataset):
    if dataset != "aitw":
        return " | ".join(elem_list).strip()
    
    cleaned_elem_list = []
    for e in elem_list:
        if is_good_ocr(e):
            cleaned_elem_list.append(e)
    return " | ".join(cleaned_elem_list) 

def handle_edge_case(curr_split, next_split, reasons):
    if len(curr_split) == 3 and len(next_split) == 1:
        reasons["repeated_elems"] += 1
    elif len(curr_split) == 2 and len(next_split) == 1:
        reasons["just_newline"] += 1
    else:
        reasons["unknown"] += 1
    try:
        app = curr_split[0]
        elem_list = curr_split[1]
        cap = next_split[0].strip()
        return app, elem_list, cap, reasons
    except:
        print(curr_split)
        print(next_split)

def clean_caption_edge_case(caption):
    elems = caption.split(" | ")
    cleaned_elems = []
    for e in elems:
        if e not in cleaned_elems:
            cleaned_elems.append(e)
    return ", ".join(cleaned_elems)

def app_to_caption_map(output_list):
    map_dict = defaultdict(lambda: defaultdict(str))
    reasons = defaultdict(int)
    issues = 0
    might_have_repeated_still = 0
    last_was_issue = False
    newline_issues = 0
    for i in range(len(output_list)):
        if last_was_issue:
            last_was_issue = False
            continue
        sample = output_list[i]
        if i < len(output_list) - 1:
            next_sample = output_list[i+1]
        else:
            next_sample = sample

        sample = sample.strip()
        split_sample = sample.split("[*]")
        if len(split_sample) < 2:
            print(split_sample)
            newline_issues+=1
            continue
        next_sample = next_sample.strip()
        split_next_sample = next_sample.split("[*]")
        if len(split_sample) != 3 or len(split_next_sample) != 3:
            issues += 1
            last_was_issue=True # skip next

            app, elem_list, output_caption, reasons = handle_edge_case(split_sample, split_next_sample, reasons)
            output_caption = output_caption.strip()
            if " | " in output_caption:
                might_have_repeated_still+=1
                output_caption = clean_caption_edge_case(output_caption)
            map_dict[app.strip()][elem_list.strip()] = output_caption

        else:
            app, elem_list, output_caption = split_sample
            output_caption = output_caption.strip()
            if " | " in output_caption:
                might_have_repeated_still+=1
                output_caption = clean_caption_edge_case(output_caption)
            map_dict[app.strip()][elem_list.strip()] = output_caption
    print("Issues %d" % issues)
    print("Edge cases: [1] repeated elems first %d [2] had newline prior to caption %d [3] unknown other cases %d" % (
        reasons["repeated_elems"], reasons["just_newline"], reasons["unknown"]))
    print("Might have repeated elements still: %d" % might_have_repeated_still)
    print("Newline issues: %d" % newline_issues)
    return map_dict

def __main__():
    global args
    args = parser.parse_args()

    all_gpt_captions = []
    all_gpt_files = glob.glob(args.input_path)
    for fi in all_gpt_files:
        print(fi)
        with open(fi) as f:
            d = f.readlines()
        for line in d:
            if line.strip() != "":
                all_gpt_captions.append(line)

    dataset_elem_files = glob.glob(args.dataset_path)
    dname = args.dataset_path.split("/")[-3]
    assert dname in NUM_JSONS_TO_STORE

    empty_list = []
    caption_mapping = app_to_caption_map(all_gpt_captions)
    blip_samples = []
    missing_samples = []
    for fi in dataset_elem_files:
        print(fi)
        with open(fi) as f:
            dataset_elems_samples = json.load(f)
        for app in dataset_elems_samples:
            for screen_id in dataset_elems_samples[app]:
                elems = dataset_elems_samples[app][screen_id]
                elems_str = format_elem_list(elems, dname)
                if elems_str == "":
                    empty_list.append((app, elems_str))
                    continue
                try:
                    assert app.strip() in caption_mapping
                    assert elems_str in caption_mapping[app.strip()]
                except:
                    missing_samples.append("[*]".join([app, elems_str]))
                    print([app, elems_str])
                    continue
                cap = caption_mapping[app][elems_str]

                if dname == "longitudinal":
                    full_screen = os.path.join(app, screen_id + ".png")
                elif dname == "aitw":
                    full_screen = screen_id + ".jpg"
                else:
                    full_screen = glob.glob(os.path.join(
                        "/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/motif/*",
                        app,
                        "*/screens",
                        screen_id + ".jpg"))
                    assert len(full_screen) > 0
                    full_screen = full_screen[0].split("motif/")[-1]
                new_sample = {"image": full_screen, "caption": cap}
                blip_samples.append(new_sample)
    
    missing_samples = set(missing_samples)
    empty_list = set(empty_list)
    print("Randomly missing %d samples, %d empty element list samples" % (len(missing_samples), len(empty_list)))
    missed_path = "samples_still_needed_" + dname + ".txt"
    if missing_samples and not os.path.exists(missed_path):
        with open(missed_path, "w") as f:
            f.write("\n".join(missing_samples))
    
    write_path = args.dataset_path.split("/")[:-2] + ["gpt_captions"]
    write_path = "/".join(write_path)
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    print("Writing to %s..." % write_path)
    
    increment = len(blip_samples) // NUM_JSONS_TO_STORE[dname]
    for i in range(0, len(blip_samples), increment):
        subset = blip_samples[i:i+increment]
        with open(os.path.join(write_path, str(i//increment) + "_captions.json"), "w") as f:
            json.dump(subset, f)

__main__()