import argparse
import glob
import json
import os

from collections import defaultdict

parser = argparse.ArgumentParser('Create pretrain annotations for Fortune objective')
parser.add_argument('--dataset',
                    type=str,
                    default="aitw",
                    help='specify which dataset to process')
parser.add_argument('--element_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/aitw/elements_raw/*",
                    help='specify output_folder')

def __main__():
    global args
    args = parser.parse_args()

    elems = glob.glob(args.element_path)
    triplets = os.path.join("/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/spotlight_jsons", args.dataset, "st_at_st1/triplets.txt")
    print(triplets)
    issues = defaultdict(int)

    cleaned = []
    with open(triplets) as f:
        samples = [x.strip().split(",") for x in f.readlines()]
        if args.dataset != "aitw":
            set_ids = [x[2:4] for x in samples]
            set_ids = set([y.split(".")[0] for x in set_ids for y in x])
        else:
            set_ids = [x[1:4] for x in samples]
            set_ids = [["_".join([x[0], x[1]]), "_".join([x[0], x[2]])] for x in set_ids]
            set_ids = set([y.split(".")[0] for x in set_ids for y in x])
    set_elems = {}
    for fi in elems:
        print(fi)
        with open(fi) as f:
            data = json.load(f)
        for app in data:
            for uid in data[app]:
                if uid in set_ids:
                    set_elems[uid] = data[app][uid]

    # print(samples[:5])
    for sample in samples:
        # print(sample)
        if args.dataset != "aitw":
            app, st, st1, str_bbox = sample
        else:
            app, epid, st, st1, str_bbox = sample
            st = "_".join([epid, st])
            st1 = "_".join([epid, st1])

        bbox = [float(x) for x in str_bbox.split(" ")]
        if ((bbox[2] - bbox[0]) == 1) or ((bbox[3] - bbox[1]) == 1):
            issues["bbox_full_w_or_h"] += 1
            continue
        if ((bbox[2] - bbox[0]) == 0) or ((bbox[3] - bbox[1]) == 0):
            issues["bbox_zero_area"] += 1
            continue
        if args.dataset == "motif":
            st = st.split(".")[0]
            st1 = st1.split(".")[0]
        if st == st1:
            issues["same_state_id"] += 1
            continue
        if st not in set_elems:
            issues["missing_curr_state_elems"] += 1
            continue
        if st1 not in set_elems:
            issues["missing_next_state_elems"] += 1
            continue
        if set_elems[st] == set_elems[st1]:
            issues["same_state_elems"] += 1
            continue
        cleaned.append(sample)

    print("%s triplets before and after cleaning %d --> %d" % (args.dataset, len(samples), len(cleaned)))
    with open(triplets.replace("triplets.txt", "triplets_clean.txt"), "w") as f:
        to_write = "\n".join([",".join(x) for x in cleaned])
        f.write(to_write)

    for k,v in issues.items():
        print(k,v)

__main__()