"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import glob
import random

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset
from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset

class PretrainVQADataset(ImageTextPairDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, use_prefix_lm=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.use_prefix_lm = use_prefix_lm
        print("Using prefix lm loss %s" % use_prefix_lm)
        self.img_ids = {}
        # self.question_ids = {}
        n = 0
        # qn = 0
        for ann in self.annotation:
            img_id = ann["image"]
            # question_id = ann["image_id"] + ann["dataset"] + ann["question"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        # if question_id not in self.question_ids.keys():
        #     self.question_ids[question_id] = qn
        #     qn += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        if "question" in ann:
            question = self.text_processor(ann["question"], "Question: {} Answer: ")
        else:
            question = ""
        # question_id_key = ann["image_id"] + ann["dataset"] + ann["question"]
        if self.use_prefix_lm:
            original_answer = self.text_processor(ann["caption"]).split(" ")
            
            num_words = len(original_answer)
            if num_words > 1:
                prefix_split = random.randint(1, len(original_answer) // 2)
                answer_prefix = " ".join(original_answer[:prefix_split]) + " "
                
                question += answer_prefix
                answer = " ".join(original_answer[prefix_split:])
            else:
                answer = self.text_processor(ann["caption"])
        else:
            answer = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
            "image_id": self.img_ids[ann["image"]],
            # "question_id": self.question_ids[question_id_key],
            "prompt": question,
        }

class PretrainVQAEvalDataset(PretrainVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        question = self.text_processor(ann["question"], "Question: {} Answer: ")
        # question_id_key = ann["image_id"] + ann["dataset"] + ann["question"]

        return {
            "image": image,
            "text_input": question,
            # "image_id": self.img_ids[ann["image_id"]],
            "text_output": "",
            # "question_id": self.question_ids[question_id_key],
            "prompt": question,
        }