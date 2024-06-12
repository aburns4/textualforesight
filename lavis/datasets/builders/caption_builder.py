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
    RicoCapDataset,
    RicoVQADataset,
    RicoVQAEvalDataset,
    RicoGroundVQAEvalDataset,
    RicoGroundCaptionEvalDataset
)

from lavis.datasets.datasets.app_pretrain_datasets import (
    PretrainVQADataset,
    PretrainVQAEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)

@registry.register_builder("tap_vqa")
class RicoVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = RicoVQADataset
    eval_dataset_cls = RicoVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rico/tappability_vqa.yaml",
        "twice": "configs/datasets/rico/tappability_2_vqa.yaml",
        "flipped": "configs/datasets/rico/tappability_flipped_vqa.yaml",
        "flipped_debug": "configs/datasets/rico/tappability_flipped_debug_vqa.yaml",
        "flipped_twice": "configs/datasets/rico/tappability_flipped_2_vqa.yaml",
        "caption": "configs/datasets/rico/tappability_caption_vqa.yaml",
        "caption_double": "configs/datasets/rico/tappability_caption_2_vqa.yaml",
        "caption_quad": "configs/datasets/rico/tappability_caption_4_vqa.yaml",
        "caption_ten": "configs/datasets/rico/tappability_caption_10_vqa.yaml",
        "caption_100": "configs/datasets/rico/tappability_caption_100_vqa.yaml",
        "desc_caption": "configs/datasets/rico/tappability_desc_caption_vqa.yaml",
        "desc_caption_double": "configs/datasets/rico/tappability_desc_caption_2_vqa.yaml",
        "desc_caption_quad": "configs/datasets/rico/tappability_desc_caption_4_vqa.yaml",
    }

@registry.register_builder("language_ground")
class RicoVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = RicoVQADataset
    eval_dataset_cls = RicoVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rico/language_grounding_1.yaml",
        "ratio_4": "configs/datasets/rico/language_grounding_4.yaml",
        "ratio_10": "configs/datasets/rico/language_grounding_10.yaml",
        "all": "configs/datasets/rico/language_grounding_all.yaml",
        "all_eval": "configs/datasets/rico/language_grounding_eval_og_all.yaml",
        "captions": "configs/datasets/rico/language_grounding_captions.yaml", # by default this is first instr from mug
        "captions_full": "configs/datasets/rico/language_grounding_captions_all_instr.yaml",
        "captions_last": "configs/datasets/rico/language_grounding_captions_last_instr.yaml",

    }

@registry.register_builder("language_ground_eval")
class RicoVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = RicoVQADataset
    eval_dataset_cls = RicoGroundVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rico/language_grounding_eval.yaml",
        "debug": "configs/datasets/rico/language_grounding_eval_debug.yaml",
    }

@registry.register_builder("language_ground_caption_eval")
class RicoVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = RicoVQADataset
    eval_dataset_cls = RicoGroundCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rico/language_grounding_captions_eval.yaml",
    }

# @registry.register_builder("language_ground_captions_eval")
# class RicoVQABuilder(BaseDatasetBuilder):
#     train_dataset_cls = RicoVQADataset
#     eval_dataset_cls = RicoGroundCaptionEvalDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/rico/language_grounding_captions_eval.yaml",
#     }

@registry.register_builder("widget_vqa")
class RicoVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = RicoVQADataset
    eval_dataset_cls = RicoVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rico/widget_vqa.yaml",
    }

@registry.register_builder("rico_pretrain")
class PretrainVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PretrainVQADataset
    eval_dataset_cls = PretrainVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "stage1": "configs/datasets/app_pretrain/rico_pretrain_stage1.yaml",
        "stage2": "configs/datasets/app_pretrain/rico_pretrain_stage2.yaml",
        "dummy": "configs/datasets/app_pretrain/rico_pretrain_stage1_dummy.yaml",
    }

@registry.register_builder("motif_pretrain")
class PretrainVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PretrainVQADataset
    eval_dataset_cls = PretrainVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "stage1": "configs/datasets/app_pretrain/motif_pretrain_stage1.yaml",
        "stage1_fortune": "configs/datasets/app_pretrain/motif_pretrain_stage1_fortune.yaml",
        "stage2": "configs/datasets/app_pretrain/motif_pretrain_stage2.yaml",
        "stage2_2m_imgs": "configs/datasets/app_pretrain/motif_pretrain_stage2_2m_imgs.yaml",
        "stage2_2m_samples": "configs/datasets/app_pretrain/motif_pretrain_stage2_2m_samples.yaml",
        "stage2_fortune": "configs/datasets/app_pretrain/motif_pretrain_stage2_fortune.yaml",
        "dummy": "configs/datasets/app_pretrain/motif_pretrain_stage1_dummy.yaml",

    }

@registry.register_builder("longitudinal_pretrain")
class PretrainVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PretrainVQADataset
    eval_dataset_cls = PretrainVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "stage1": "configs/datasets/app_pretrain/longitudinal_pretrain_stage1.yaml",
        "stage1_fortune": "configs/datasets/app_pretrain/longitudinal_pretrain_stage1_fortune.yaml",
        "stage2": "configs/datasets/app_pretrain/longitudinal_pretrain_stage2.yaml",
        "stage2_2m_imgs": "configs/datasets/app_pretrain/longitudinal_pretrain_stage2_2m_imgs.yaml",
        "stage2_2m_samples": "configs/datasets/app_pretrain/longitudinal_pretrain_stage2_2m_samples.yaml",
        "stage2_fortune": "configs/datasets/app_pretrain/longitudinal_pretrain_stage2_fortune.yaml"
    }

@registry.register_builder("aitw_pretrain")
class PretrainVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PretrainVQADataset
    eval_dataset_cls = PretrainVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "stage1": "configs/datasets/app_pretrain/aitw_pretrain_stage1.yaml",
        "stage1_fortune": "configs/datasets/app_pretrain/aitw_pretrain_stage1_fortune.yaml",
        "no_icon_stage1": "configs/datasets/app_pretrain/aitw_pretrain_no_icon_stage1.yaml",
        "stage2": "configs/datasets/app_pretrain/aitw_pretrain_stage2.yaml",
        "stage2_2m_imgs": "configs/datasets/app_pretrain/aitw_pretrain_stage2_2m_imgs.yaml",
        "stage2_2m_samples": "configs/datasets/app_pretrain/aitw_pretrain_stage2_2m_samples.yaml",
        "stage2_fortune": "configs/datasets/app_pretrain/aitw_pretrain_stage2_fortune.yaml",
    }

@registry.register_builder("screen_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    # def __init__(self, cfg=None):
    #     BaseDatasetBuilder.__init__(self, cfg)
    # model_type=self.config.build_info.model
    train_dataset_cls = RicoCapDataset # .__init__()
    eval_dataset_cls = COCOCapEvalDataset
        
    DATASET_CONFIG_DICT = {
            "default": "configs/datasets/rico/screen_summarization.yaml",
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
