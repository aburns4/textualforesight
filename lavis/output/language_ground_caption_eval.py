import json
import argparse
import evaluate
import glob

from collections import defaultdict
from pycocoevalcap.cider.cider import Cider

parser = argparse.ArgumentParser('Final accuracy computation for language grounding')
parser.add_argument('--res_file_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/lavis/output/BLIP2/language_ground/flant5/20231013144_1/20231015170/result/test_vqa_result.json",
                    help='file patterns to load model outputs')
parser.add_argument('--ann_object_file_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/mug/mug_objects_v1_modified.test.json",
                    help='file pattern to load ground truth annotations')
# parser.add_argument('--ann_caption_file_path',
#                     type=str,
#                     default="/projectnb2/ivc-ml/aburns4/mug/mug_captions_test.json",
#                     help='file pattern to load ground truth annotations')
parser.add_argument('--metric',
                    type=str,
                    default="rouge",
                    help='metric type to use')
parser.add_argument('--instr_type',
                    type=str,
                    choices=["first", "last", "all"],
                    help='which instruction type to compare')
parser.add_argument('--bleurt_model',
                    type=str,
                    default="BLEURT-20-D12",
                    help='if BLEURT is the selected metric, which model variant should be used')
parser.add_argument('--constraint',
                    type=str,
                    choices=["<", "<="],
                    help='if target caption score has to be strictly greater than other objects or at least as large')

def format_res_outputs(res_list):
    new_dict = defaultdict(lambda : defaultdict(list))
    for sample in res_list:
        new_key = "_".join([str(sample["image_id"]), sample["object_id"]])
        new_dict[new_key] = sample["pred_ans"]
    return new_dict

def get_bleurt(targets, predictions, model):
    assert len(targets) == len(predictions)
    bleurt = evaluate.load("bleurt", model)
    bleurt_score = bleurt.compute(predictions=predictions, references=targets)['scores']
    return bleurt_score

def get_bertscore(targets, predictions):
    assert len(targets) == len(predictions)
    bert = evaluate.load("bertscore")
    bert_score = bert.compute(predictions=predictions, references=targets, lang='en')['f1']
    return bert_score

def get_rouge(targets, predictions):
    assert len(targets) == len(predictions)
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=predictions, references=targets, use_aggregator=False)['rougeL']
    return rouge_score

def get_cider(targets, predictions):
    assert len(targets) == len(predictions)
    cider = Cider()
    _, cider_scores = cider.compute_score(targets, predictions)
    return cider_scores

def get_prediction(target_scores, object_scores, constraint):
    pred_booleans = []
    for tscore, oscores in zip(target_scores, object_scores):
        if constraint == "<=":
            comp = [o <= tscore for o in oscores]
        else:
            assert constraint == "<"
            comp = [o < tscore for o in oscores]
        if all(comp):
            pred_booleans.append(1)
        else:
            pred_booleans.append(0)
    return pred_booleans

def get_formatted_instruction(instr_list, instr_type):
    instrs = []
    for i in range(len(instr_list)):
        if i % 2 == 0:
            instrs.append(instr_list[i])

    if instr_type == "all":
        caption = " ".join(instrs)
    elif instr_type == "last":
        caption = instrs[-1]
    elif instr_type == "first":
        caption = instrs[0]
    else:
        raise ValueError("Instruction type %s not supported. Please choose one of [all, last, first]" % instr_type)
    return caption

def get_instr_type(res_file_path):
    log_file_path = res_file_path.split('/')[:-3] + ['log.txt']
    log_file_path = "/".join(log_file_path)
    with open(log_file_path) as f:
        data = f.readlines()
    for li in data:
        if "\"url\":" in li:
            if "captions" not in li:
                raise ValueError("Outdated result file type, prior to when captions were used")
            if "captions_full_instr" in li:
                return "all"
            elif "captions_last_instr" in li:
                return "last"
            else:
                return "first"

def __main__():
    global args
    args = parser.parse_args()

    res_paths = glob.glob(args.res_file_path)
    with open(args.ann_object_file_path) as f:
        object_mapping = json.load(f)

    for res_file_path in res_paths:
        try:
            instr_type = get_instr_type(res_file_path)
            print(res_file_path)

            with open(res_file_path) as f:
                outputs = json.load(f)
            outputs_dict = format_res_outputs(outputs)
            # print(len(outputs_dict))

            # with open(args.ann_caption_file_path) as f:
            #     gt_anns = json.load(f)

            print("Loaded all input files...")
            gt_captions = []
            target_output_captions = []
            object_output_captions = []
            
            gt_dict = defaultdict(list)
            target_output_dict = defaultdict(list)
            for sample_key in object_mapping:
                image_id, tkey = sample_key.split("_")
                sample = object_mapping[sample_key]

                target_output_captions.append(outputs_dict[sample_key])
                target_output_dict[sample_key].append(outputs_dict[sample_key])

                gt_instr = get_formatted_instruction(object_mapping[sample_key]["instruction"], instr_type)
                gt_captions.append(gt_instr)
                gt_dict[sample_key].append(gt_instr)

                current_objects = []
                for okey in sample["object_ids"]:
                    if okey != tkey:
                        output_okey = "_".join([image_id, okey])
                        current_objects.append(outputs_dict[output_okey])

                object_output_captions.append(current_objects)
            
            print("Getting target object scores...")
            if args.metric == "bleurt":
                target_object_scores = get_bleurt(gt_captions, target_output_captions, args.bleurt_model)
            elif args.metric == "bertscore":
                target_object_scores = get_bertscore(gt_captions, target_output_captions)
            elif args.metric == "rouge":
                target_object_scores = get_rouge(gt_captions, target_output_captions)
            elif args.metric == "cider":
                # print(len(gt_dict), len(target_output_dict))
                target_object_scores = get_cider(gt_dict, target_output_dict)
            else:
                raise ValueError("%s is not supported, please choose metric from [bleurt, bertscore, rouge, cider]" % args.metric)
            print("Done!")
            other_object_scores = []
            obj_gt_captions = []
            obj_gt_dict = defaultdict(list)
            object_output_dict = defaultdict(list)
            obj_batch_lens = []
            object_output_captions_flat = []
            print("Getting other object scores...")
            dict_idx = 0
            for i in range(len(object_output_captions)):
                # print(i)
                obj_batch = object_output_captions[i]
                gt_batch = [gt_captions[i]]*len(obj_batch)
                obj_batch_lens.append(len(obj_batch))
                obj_gt_captions.extend(gt_batch)
                object_output_captions_flat.extend(obj_batch)

                for j in range(len(obj_batch)):
                    obj_gt_dict[str(dict_idx)].append(gt_captions[i])
                    object_output_dict[str(dict_idx)].append(obj_batch[j])
                    dict_idx += 1

            if args.metric == "bleurt":
                obj_flat_scores = get_bleurt(obj_gt_captions, object_output_captions_flat, args.bleurt_model)
            elif args.metric == "bertscore":
                obj_flat_scores = get_bertscore(obj_gt_captions, object_output_captions_flat)
            elif args.metric == "rouge":
                obj_flat_scores = get_rouge(obj_gt_captions, object_output_captions_flat)
            elif args.metric == "cider":
                # print(len(obj_gt_dict), len(object_output_dict))
                obj_flat_scores = get_cider(obj_gt_dict, object_output_dict)
            else:
                raise ValueError("%s is not supported, please choose metric from [bleurt, bertscore, rouge, cider]" % args.metric)
            print("Done!")
            # print(obj_flat_scores)
            for i in range(len(obj_batch_lens)):
                curr_idx = sum(obj_batch_lens[:i])
                curr_batch_scores = obj_flat_scores[curr_idx : curr_idx + obj_batch_lens[i]]
                other_object_scores.append(curr_batch_scores)

            predictions = get_prediction(target_object_scores, other_object_scores, args.constraint)
            acc = sum(predictions)*100 / len(predictions) 
            # f1_metric = evaluate.load("f1")
            # f1 = f1_metric.compute(predictions=predictions, references=[1]*len(predictions))['f1']
            acc_str = "Language grounding accuracy with %s metric %s: %.1f" % (args.metric, args.constraint, acc)
            print(acc_str)
            res_str = [acc_str] + [",".join([str(x) for x in predictions]) + "\n"]
            save_file_path = res_file_path.replace("result.json", "accuracy.txt")
            with open(save_file_path, "a+") as f:
                f.write("\n".join(res_str))
        except:
            continue
__main__()