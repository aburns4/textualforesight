"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import copy
import logging
import json
import os
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.tasks.base_task import BaseTask
from lavis.tasks.captioning import coco_caption_eval

@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        answer_list=None,
        prompt="",
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = (None if answer_list is None else OmegaConf.to_object(answer_list))

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )

            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics

@registry.register_task("rico_vqa")
class RicoVQATask(VQATask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="generate",
        answer_list=None,
        prompt="",
        metric_type="acc_vqa",
        anns_path="",
    ):
        super().__init__(num_beams,
                         max_len,
                         min_len,
                         evaluate,
                         num_ans_candidates,
                         inference_method=inference_method,
                         answer_list=answer_list,
                         prompt=prompt)
        self.metric_type = metric_type
        self.anns_path = anns_path

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "generate")
        answer_list = run_cfg.get("answer_list", None)
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")
        eval_metric = run_cfg.get("metric_type", "acc_vqa")
        anns_path = run_cfg.get("anns_path", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            answer_list=answer_list,
            inference_method=inference_method,
            prompt=prompt,
            metric_type=eval_metric,
            anns_path=anns_path,
        )

    def valid_step(self, model, samples):
        # print(self.answer_list)
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )

        if self.inference_method == "rank" and self.answer_list is not None:
            answers = [self.answer_list[index[-1]] for index in answers]
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["text_output"]
        img_ids = samples["image_id"]

        for answer, ques_id, gt_answer, img_id in zip(answers, question_id, gt_answers, img_ids):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer,
                                  "caption": answer, "image_id": int(img_id)})

        return pred_qa_pairs

    def get_annotations_file(self, split):
        assert 'widget-caption' in self.anns_path
        filenames = {
            "val": "eval_dev.json",
            "test": "eval_test.json",
        }
        annotation_file = os.path.join(self.anns_path, filenames[split])
        return annotation_file

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        results = json.load(open(result_file, "r"))

        if self.metric_type == "acc_vqa":
            acc = []
            vqa_tool = VQAEval()

            for res in results:
                if res["gt_ans"] is None:
                    # prepare test results for leaderboard evaluation
                    self._save_result_leaderboard(results)
                    return

                gt_ans = res["gt_ans"]
                pred = res["pred_ans"]

                if self.inference_method == "generate":
                    pred = vqa_tool.processPunctuation(pred)
                    pred = vqa_tool.processDigitArticle(pred)

                vqa_acc = 1 if pred == gt_ans else 0

                acc.append(vqa_acc)

            accuracy = sum(acc) / len(acc) * 100
            metrics = {"agg_metrics": accuracy, "acc": accuracy}
        elif self.metric_type == "f1_vqa_flipped":
            refs = []
            preds = []
            vqa_tool = VQAEval()
            for res in results:
                if res["gt_ans"] is None:
                    # prepare test results for leaderboard evaluation
                    self._save_result_leaderboard(results)
                    return

                gt_ans = res["gt_ans"]
                pred = res["pred_ans"]

                if self.inference_method == "generate":
                    pred = vqa_tool.processPunctuation(pred)
                    pred = vqa_tool.processDigitArticle(pred)

                ref = 1 if gt_ans == "no" else 0
                pred = 1 if pred == "no" else 0

                refs.append(ref)
                preds.append(pred)

            f1 = f1_score(refs, preds)  * 100
            metrics = {"agg_metrics": f1, "f1": f1}
        elif self.metric_type == "f1_vqa":
            refs = []
            preds = []
            vqa_tool = VQAEval()
            for res in results:
                if res["gt_ans"] is None:
                    # prepare test results for leaderboard evaluation
                    self._save_result_leaderboard(results)
                    return

                gt_ans = res["gt_ans"]
                pred = res["pred_ans"]

                if self.inference_method == "generate":
                    pred = vqa_tool.processPunctuation(pred)
                    pred = vqa_tool.processDigitArticle(pred)

                ref = 1 if gt_ans == "yes" else 0
                pred = 1 if pred == "yes" else 0

                refs.append(ref)
                preds.append(pred)

            f1 = f1_score(refs, preds)  * 100
            metrics = {"agg_metrics": f1, "f1": f1}
        else:
            assert self.metric_type == "caption"
            annotation_file = self.get_annotations_file(split)
            coco_val = coco_caption_eval(annotation_file, result_file)

            agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_1"]
            log_stats = {split: {k: v for k, v in coco_val.eval.items()}}

            # with open(
            #     os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            # ) as f:
            #     f.write(json.dumps(log_stats) + "\n")

            coco_res = {k: v for k, v in coco_val.eval.items()}
            coco_res["agg_metrics"] = agg_metrics
            metrics = coco_res

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

@registry.register_task("rico_ground_vqa")
class RicoGroundTask(RicoVQATask):
    def valid_step(self, model, samples):
        assert len(samples["image_id"]) == 1 # single sample for eval since we have to go over all elements
        
        # print(samples["target_question_input"])
        # print(samples["other_obj_question_input"])
        target_sample = samples
        target_sample["text_input"] = target_sample["target_question_input"]
        target_yes_score = model.predict_grounding_answers(
            samples=target_sample,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        
        other_obj_yes_scores = []
        eval_bs = 16
        # for i in range(0, len(target_sample["other_obj_question_input"])):
        #     curr_sample_scores = []
        for i in range(0, len(target_sample["other_obj_question_input"]), eval_bs):
            curr_question_batch = copy.deepcopy(target_sample)
            curr_question_batch["text_input"] = [y for x in curr_question_batch["other_obj_question_input"][i:i+eval_bs] for y in x]
            # print(curr_question_batch["image"].size())
            # print(min(eval_bs, len(curr_question_batch["text_input"])))
            curr_question_batch["image"] = curr_question_batch["image"].repeat(min(eval_bs, len(curr_question_batch["text_input"])), 1, 1, 1)
            # print(curr_question_batch["image"].size())
            # for key in curr_question_batch:
            #     print(key)
            #     print(curr_question_batch[key])
            #     curr_question_batch[key] = curr_question_batch[key][i:i+eval_bs]
            #     print(curr_question_batch[key])
            # print(curr_question_batch["text_input"])
            # other_samples = []
            # for q in curr_question_batch:
            #     curr_sample = samples
            #     curr_sample["text_input"] = q
            #     other_samples.append(curr_sample)
            
            other_yes_scores = model.predict_grounding_answers(
                samples=curr_question_batch,
                answer_list=self.answer_list,
                inference_method=self.inference_method,
                num_beams=self.num_beams,
                max_len=self.max_len,
                min_len=self.min_len,
                num_ans_candidates=self.num_ans_candidates,
                prompt=self.prompt,
            )
            other_obj_yes_scores += other_yes_scores

        question_id = samples["question_id"]
        gt_answer = samples["text_output"]
        img_id = samples["image_id"]
        if isinstance(gt_answer, list):
            assert len(gt_answer) == 1
            gt_answer = gt_answer[0]
        # print(target_yes_score)
        # print(other_obj_yes_scores)
        # print(max(other_obj_yes_scores))
        
        
        # for ques_id, gt_answer, img_id, tscore, oscore in zip(question_ids, gt_answers, img_ids, target_yes_scores, other_obj_yes_scores):
        ques_id = int(question_id.item())
        if target_yes_score > max(other_obj_yes_scores):
            answer = "yes"
        else:
            answer = "no"
        pred_qa_pairs = [{"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer, "image_id": int(img_id)}]

        return pred_qa_pairs

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        results = json.load(open(result_file, "r"))

        acc = []
        vqa_tool = VQAEval()

        for res in results:
            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]

            if self.inference_method == "generate":
                pred = vqa_tool.processPunctuation(pred)
                pred = vqa_tool.processDigitArticle(pred)

            vqa_acc = 1 if pred == gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "full_object_evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

@registry.register_task("gqa")
class GQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        
        for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        TODO: add other evaluation metrics for GQA
        """

        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]

            # if self.inference_method == "generate":
            pred = vqa_tool.processPunctuation(pred)
            pred = vqa_tool.processDigitArticle(pred)

            vqa_acc = 1 if pred == gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics
        

@registry.register_task("aok_vqa")
class AOKVQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
        )

        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["direct_answers"]

        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            pred_qa_pairs.append(
                {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer}
            )

        return pred_qa_pairs

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Implementing accuracy computation for AOKVQA, see
        https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py#L45 for details.
        """
        # TODO add evaluation for multi-choice

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            pred = res["pred_ans"]
            gt_ans = res["gt_ans"]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / 3.0)

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

    @dist_utils.main_process
    def _save_result_leaderboard(self, results):
        """
        Saving the results in the format required for leaderboard evaluation.

        [TODO] add support for multi-choice.
        """
        result_leaderboard = dict()
        for res in results:
            result_leaderboard[res["question_id"]] = {
                "direct_answer": res["pred_ans"],
                "multiple_choice": "",
            }

        result_file = registry.get_path("result_dir") + "_leaderboard.json"

        with open(result_file, "w") as f:
            json.dump(result_leaderboard, f)

        logging.info(f"Saved results for leaderboard evaluation at {result_file}")
