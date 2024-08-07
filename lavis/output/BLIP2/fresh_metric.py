import argparse
import json
import evaluate
import glob
import os

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge

parser = argparse.ArgumentParser('Model output text metric computation')
parser.add_argument('--metrics',
                    nargs='+',
                    default=['bleu', 'rouge', 'cider'],
                    help='metric to use for evaluation')
parser.add_argument('--res_file_path',
                    nargs='+',
                    help='file patterns to load model outputs')
parser.add_argument('--ann_file_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/screen2words",
                    help='file pattern to load ground truth annotations')
parser.add_argument('--val_split_name',
                    type=str,
                    default='val',
                    help='what val file is named')

class EvalTask():
    def __init__(self, result_path, ann_path, metrics):
        self.res_path = result_path
        self.gt_path = ann_path
        self.metrics = metrics

        targets, t_im_ids, t_q_ids = self.load_targets()
        preds, p_im_ids, p_q_ids = self.load_predictions()
        self.targets = targets
        self.predictions = preds
        self.t_q_ids = t_q_ids
        self.p_q_ids = p_q_ids
        assert t_im_ids == p_im_ids
        assert t_q_ids == p_q_ids

    def load_targets(self):
        gt_tuples = []
        with open(self.gt_path) as f:
            gt_data = json.load(f)
    
        for sample in gt_data:
            gt_tuples.append((sample["caption"], int(sample["image_id"]), int(sample["question_id"])))
        
        gt_tuples = sorted(gt_tuples, key = lambda x : x[2])
        gts = [x[0] for x in gt_tuples]
        im_ids = [x[1] for x in gt_tuples]
        q_ids = [x[2] for x in gt_tuples]

        return gts, im_ids, q_ids

    def load_predictions(self):
        pred_tuples = []
        with open(self.res_path) as f:
            pred_data = json.load(f)
    
        for sample in pred_data:
            pred_tuples.append((sample["pred_ans"], int(sample["image_id"]), int(sample["question_id"])))
        
        pred_tuples = sorted(pred_tuples, key = lambda x : x[2])
        preds = [x[0] for x in pred_tuples]
        im_ids = [x[1] for x in pred_tuples]
        q_ids = [x[2] for x in pred_tuples]

        return preds, im_ids, q_ids

    def compute_metrics(self):
        raise NonImplementedError

def get_caption_type(res_path):
    split = ('val' if 'val' in res_path else 'test')

    res_path = res_path.split('/')[:-2]
    res_path.append('log.txt')
    log_path = '/'.join(res_path)


    with open(log_path) as f:
        data = f.readlines()

    for li in data:
        if "\"url\":" in li and "test" in li:
            test_caption_type = li.strip().split('/')[-1].split('.')[0]
        if "\"url\":" in li and split in li:
            updated_ann_path = li.strip().split("\"")[-2]
            updated_ann_path = updated_ann_path.replace("_4", "").replace("_2", "").replace("_10", "") # TODO: does this cause issues... think about it more
            print(updated_ann_path)
    if test_caption_type == 'test':
        return 'yes_no', updated_ann_path
    elif test_caption_type == 'test_tap_caption':
        return 'yes_no_caption', updated_ann_path
    else:
        assert test_caption_type == 'test_tap_desc_caption'
        return 'yes_no_desc_caption', updated_ann_path

class Tappability(EvalTask):
    def __init__(self, result_path, ann_path, pos_label=1, metrics=['accuracy', 'f1']):
        self.pos_label = pos_label
        caption_type, updated_ann_path = get_caption_type(result_path)
        self.caption_type = caption_type

        super().__init__(result_path, updated_ann_path, metrics)
        self.compute_metrics()

    def preprocess_caption(self, caption):
        if self.caption_type == 'yes_no' or self.caption_type == 'yes_no_caption':
            return caption.lower()
        else:
            assert self.caption_type == 'yes_no_desc_caption'
            return caption.lower().split('.')[-1].strip() 
        
    def convert_caption_to_label(self, caption):
        caption = self.preprocess_caption(caption)
        if "no" not in caption and "yes" not in caption:
            print(caption)
        if "no" in caption:
            return 1
        else:
            return 0

    def load_targets(self):
        gt_tuples = []
        with open(self.gt_path) as f:
            gt_data = json.load(f)
    
        for sample in gt_data:
            gt_tuples.append((self.convert_caption_to_label(sample["caption"]), int(sample["image_id"]), int(sample["question_id"])))
        
        gt_tuples = sorted(gt_tuples, key = lambda x : x[2])
        gts = [x[0] for x in gt_tuples]
        im_ids = [x[1] for x in gt_tuples]
        q_ids = [x[2] for x in gt_tuples]

        return gts, im_ids, q_ids

    def load_predictions(self):
        pred_tuples = []
        with open(self.res_path) as f:
            pred_data = json.load(f)
    
        for sample in pred_data:
            pred_tuples.append((self.convert_caption_to_label(sample["pred_ans"]), int(sample["image_id"]), int(sample["question_id"])))
        
        pred_tuples = sorted(pred_tuples, key = lambda x : x[2])
        preds = [x[0] for x in pred_tuples]
        im_ids = [x[1] for x in pred_tuples]
        q_ids = [x[2] for x in pred_tuples]

        return preds, im_ids, q_ids

    def compute_metrics(self):
        # print(self.metrics)
        if 'f1' in self.metrics:
            f1_score = get_f1(self.targets, self.predictions, self.pos_label) * 100
            print('F1 Score: %.1f' % f1_score)
        if 'accuracy' in self.metrics:
            acc_score = get_accuracy(self.targets, self.predictions) * 100
            print('Accuracy: %.1f\n' % acc_score)


class VQACaptionTask(EvalTask):
    def __init__(self, result_path, ann_path, metrics=['cider', 'rouge', 'bleu']):
        super().__init__(result_path, ann_path, metrics)
        
        self.convert_to_dict(self.t_q_ids, self.p_q_ids)
        self.compute_metrics()

    def convert_to_dict(self, gt_ids, pred_ids):
        tdict = {}
        pdict = {}
        for gt, qid in zip(self.targets, gt_ids):
            assert isinstance(gt, list)
            tdict[qid] = gt
        for pred, qid in zip(self.predictions, pred_ids):
            assert isinstance(pred, str)
            pdict[qid] = [pred]

        self.target_dict = tdict
        self.prediction_dict = pdict

    def compute_metrics(self):
        if 'bleu' in self.metrics:
            bleu_score = get_bleu2(self.targets, self.predictions)
            print('BLEU:', bleu_score)
        
        if 'rouge' in self.metrics:
            rouge_score = get_rouge2(self.targets, self.predictions)
            print('ROUGE:', rouge_score)
        
        if 'cider' in self.metrics:
            cider_score = get_cider(self.target_dict, self.prediction_dict)
            print('CIDEr:', cider_score)

        if 'bleurt' in self.metrics:
            bleurt_score = get_bleurt(self.targets, self.predictions)
            print('BLEURT:', bleurt_score)

        if 'bertscore' in self.metrics:
            bert_score = get_bertscore(self.targets, self.predictions)
            print('BERTSCORE:', bert_score)

class ScreenCaption(VQACaptionTask):
    def __init__(self, result_path, ann_path, metrics=['cider', 'rouge', 'bleu']):
        self.res_path = result_path
        self.gt_path = ann_path
        self.metrics = metrics

        targets, t_im_ids = self.load_targets()
        preds, p_im_ids = self.load_predictions()
        self.targets = targets
        self.predictions = preds
        self.t_im_ids = t_im_ids
        self.p_im_ids = p_im_ids

        assert t_im_ids == p_im_ids
        
        self.convert_to_dict(self.t_im_ids, self.p_im_ids)
        self.compute_metrics()

    def load_targets(self):
        gt_tuples = []
        with open(self.gt_path) as f:
            gt_data = json.load(f)
    
        for sample in gt_data:
            gt_tuples.append((sample["caption"], int(sample["image_id"])))
        
        gt_tuples = sorted(gt_tuples, key = lambda x : x[1])
        gts = [x[0] for x in gt_tuples]
        im_ids = [x[1] for x in gt_tuples]

        return gts, im_ids

    def load_predictions(self):
        pred_tuples = []
        with open(self.res_path) as f:
            pred_data = json.load(f)
    
        for sample in pred_data:
            pred_tuples.append((sample["caption"], int(sample["image_id"])))
        
        pred_tuples = sorted(pred_tuples, key = lambda x : x[1])
        preds = [x[0] for x in pred_tuples]
        im_ids = [x[1] for x in pred_tuples]

        return preds, im_ids

def get_f1(targets, predictions, pos_label):
    f1_metric = evaluate.load("f1")
    results = f1_metric.compute(predictions=predictions, references=targets, pos_label=pos_label)
    return results['f1']

def get_accuracy(targets, predictions):
    acc_metric = evaluate.load("accuracy")
    results = acc_metric.compute(predictions=predictions, references=targets)
    return results['accuracy']

def get_bleu(targets, predictions):
    all_bleu = {}
    avg, all_scores = Bleu().compute_score(targets, predictions)
    for i in range(4):
        key = 'bleu'+str(i)
        all_bleu[key] = round(100*avg[i], 2)

    return all_bleu

def get_bleu2(targets, predictions):
    bleu = evaluate.load('bleu')
    all_bleu = {}
    for i in range(1, 5):
        key = 'bleu'+str(i)
        all_bleu[key] = round(100*bleu.compute(references=targets, predictions=predictions,  max_order=i)['bleu'], 2)
    return all_bleu

def get_cider(targets, predictions):
    avg, all_scores = Cider().compute_score(targets, predictions)
    return round(100*avg, 2)

def get_rouge(targets, predictions):
    all_rouge = {}
    avg, all_scores = Rouge().compute_score(targets, predictions)
    return avg

def get_rouge2(targets, predictions):
    rouge = evaluate.load('rouge')
    score_dict = rouge.compute(references=targets, predictions=predictions)
    for k in score_dict:
        score_dict[k] = round(100*score_dict[k], 2)
    return score_dict

def get_bertscore(targets, predictions):
    assert len(targets) == len(predictions)
    bert = evaluate.load("bertscore")
    all_bert = []
    # bert_score = bert.compute(predictions=predictions, references=targets, lang='en')['f1']
    for t, p in zip(targets, predictions):
        sample_scores = []
        for ref in t:
            assert isinstance(p, str)
            assert isinstance(ref, str)
            curr_bert = bert.compute(predictions=[p], references=[ref], lang='en')['f1']
            sample_scores.append(curr_bert[0])
        all_bert.append(sum(sample_scores)/len(sample_scores))
    return round(sum(all_bert) / len(all_bert), 2)

def get_bleurt(targets, predictions):
    bleurt = evaluate.load("bleurt", "BLEURT-20-D12")
    all_bleurt = []
    for t, p in zip(targets, predictions):
        sample_scores = []
        for ref in t:
            assert isinstance(p, str)
            assert isinstance(ref, str)
            curr_bleurt = bleurt.compute(predictions=[p], references=[ref])['scores']
            sample_scores.append(curr_bleurt[0])
        all_bleurt.append(max(sample_scores))
    return round(sum(all_bleurt) / len(all_bleurt), 2)

def __main__():
    global args
    args = parser.parse_args()

    ress =[]
    for fi in args.res_file_path:
        print(fi)
        ress.extend(glob.glob(fi))
    print('Computing %s metrics for %d results' % (args.metrics, len(ress)))
    for res in ress:
        if 'test' in res:
            ann = os.path.join(args.ann_file_path, 'test.json')
        else:
            ann = os.path.join(args.ann_file_path, args.val_split_name + '.json')

        info = res.split('/')
        print(' '.join([info[-4],info[-3],info[-1]]))
        if 'screen2words' in args.ann_file_path:
            ScreenCaption(res, ann, args.metrics)
        elif 'widget-caption' in args.ann_file_path:
            VQACaptionTask(res, ann, args.metrics)
        elif 'taperception' in args.ann_file_path:
            try:
                Tappability(res, ann, 1, args.metrics)
            except:
                print("assertion error\n")
                continue
        else:
            raise ValueError(
                """Input annotation path does not match one of our supported tasks.
                You entered %s but we support \{screen captioning, widget captioning,
                and tappability prediction\}""" % args.ann_file_path)
        print('')

__main__()