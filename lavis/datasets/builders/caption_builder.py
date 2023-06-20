"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.datasets.datasets.rico_datasets import (
    RicoOPTCapDataset,
    RicoFlanT5CapDataset,
    RicoFlanT5VQADataset,
    RicoFlanT5VQAEvalDataset
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)

@registry.register_builder("tap_vqa")
class RicoVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = RicoVQADataset
    eval_dataset_cls = RicoVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rico/tappability_vqa.yaml",
    }

@registry.register_builder("screen_caption_opt")
class COCOCapBuilder(BaseDatasetBuilder):
    # def __init__(self, cfg=None):
    #     BaseDatasetBuilder.__init__(self, cfg)
    # model_type=self.config.build_info.model
    train_dataset_cls = RicoOPTCapDataset # .__init__()
    eval_dataset_cls = COCOCapEvalDataset
        
    DATASET_CONFIG_DICT = {
            "default": "configs/datasets/rico/screen_summarization_opt.yaml",
        }

@registry.register_builder("screen_caption_flant5")
class COCOCapBuilder(BaseDatasetBuilder):
    # def __init__(self, cfg=None):
    #     BaseDatasetBuilder.__init__(self, cfg)
    # model_type=self.config.build_info.model
    train_dataset_cls = RicoFlanT5CapDataset # .__init__()
    eval_dataset_cls = COCOCapEvalDataset
        
    DATASET_CONFIG_DICT = {
            "default": "configs/datasets/rico/screen_summarization_flant5.yaml",
        }

@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }
