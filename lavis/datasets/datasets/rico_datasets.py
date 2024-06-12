"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import glob

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

class RicoCapDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        prompt = self.text_processor("")
        caption = self.text_processor.pre_caption(ann["caption"])

        return {
            "image": image,
            "text_input": prompt,
            "text_output": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }

class RicoVQADataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"], "Question: {} Answer: ")
        question_id = ann["question_id"]
        # print(ann["caption"])
        answer = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
            "image_id": ann["image_id"], # self.img_ids[ann["image_id"]],
            "question_id": question_id,
            "prompt": question,
        }

class RicoVQAEvalDataset(RicoVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"], "Question: {} Answer: ")
        question_id = ann["question_id"]

        return {
            "image": image,
            "text_input": question,
            "image_id": ann["image_id"], # self.img_ids[ann["image_id"]],
            "text_output": "",
            "question_id": question_id,
            "prompt": question,
        }

class RicoGroundCaptionEvalDataset(RicoVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"], "Question: {} Answer: ")
        question_id = ann["question_id"]

        return {
            "image": image,
            "text_input": question,
            "image_id": ann["image"].split(".")[0], 
            "text_output": "",
            "question_id": question_id,
            "prompt": question,
            # "target_id": ann["target_id"],
            "object_id": ann["object_id"],
        }

class RicoGroundVQAEvalDataset(RicoVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        target_question = self.text_processor(ann["target_question_input"], "Question: {} Answer: ")
        other_questions = [self.text_processor(x, "Question: {} Answer: ") for x in ann["other_obj_question_input"]]
        question_id = ann["question_id"]

        answer = self.text_processor(ann["caption"]) # should always be yes

        return {
            "image": image,
            "target_question_input": target_question,
            "other_obj_question_input": other_questions,
            "image_id": ann["image_id"],
            "text_output": answer,
            "question_id": question_id,
        }