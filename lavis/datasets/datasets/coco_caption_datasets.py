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

COCOCapDataset = CaptionDataset

# just needed to reprocess image paths for 2017 images
# class COCOCapDataset(CaptionDataset):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         super().__init__(vis_processor, text_processor, vis_root, ann_paths)

#     def __getitem__(self, index):

#         # TODO this assumes image input, not general enough
#         ann = self.annotation[index]


#         image_path = glob.glob(os.path.join(self.vis_root, '*', ann["image"].split('_')[-1]))
#         assert len(image_path) == 1
#         image_path = image_path[0]
#         image = Image.open(image_path).convert("RGB")

#         image = self.vis_processor(image)
#         caption = self.text_processor(ann["caption"])

#         return {
#             "image": image,
#             "text_input": caption,
#             "image_id": self.img_ids[ann["image_id"]],
#         }

class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }


class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["img_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }
