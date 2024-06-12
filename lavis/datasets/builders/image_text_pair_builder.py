"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset
from lavis.datasets.datasets.laion_dataset import LaionDataset
from lavis.datasets.datasets.app_pretrain_datasets import PretrainVQADataset

@registry.register_builder("aitw_spotlight_caption")
class SpotlightCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/app_pretrain/pretrain_spotlight_aitw_list_caps.yaml",
                           "list_2m": "configs/datasets/app_pretrain/pretrain_spotlight_aitw_list_caps_2m.yaml",
                            "gpt": "configs/datasets/app_pretrain/pretrain_spotlight_aitw_gpt_caps.yaml"}

@registry.register_builder("motif_spotlight_caption")
class SpotlightCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/app_pretrain/pretrain_spotlight_motif_list_caps.yaml",
                            "final": "configs/datasets/app_pretrain/pretrain_spotlight_motif_list_caps_final.yaml",
                            "list_2m": "configs/datasets/app_pretrain/pretrain_spotlight_motif_list_caps_final_2m.yaml",
                            "gpt": "configs/datasets/app_pretrain/pretrain_spotlight_motif_gpt_caps.yaml"}

@registry.register_builder("longitudinal_spotlight_caption")
class SpotlightCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/app_pretrain/pretrain_spotlight_longitudinal_list_caps.yaml",
                            "final": "configs/datasets/app_pretrain/pretrain_spotlight_longitudinal_list_caps_final.yaml",
                            "list_2m": "configs/datasets/app_pretrain/pretrain_spotlight_longitudinal_list_caps_final_2m.yaml",
                            "gpt": "configs/datasets/app_pretrain/pretrain_spotlight_longitudinal_gpt_caps.yaml"}

@registry.register_builder("aitw_spotlight_stage2_caption")
class SpotlightCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = PretrainVQADataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/app_pretrain/pretrain_spotlight_2_aitw_list_caps.yaml",
                           "list_2m": "configs/datasets/app_pretrain/pretrain_spotlight_2_aitw_list_caps_2m.yaml",
                           "gpt": "configs/datasets/app_pretrain/pretrain_spotlight_2_aitw_gpt_caps.yaml",
                           "gpt_2m": "configs/datasets/app_pretrain/pretrain_spotlight_2_aitw_gpt_caps_2m.yaml"}

@registry.register_builder("motif_spotlight_stage2_caption")
class SpotlightCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = PretrainVQADataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/app_pretrain/pretrain_spotlight_2_motif_list_caps.yaml",
                            "final": "configs/datasets/app_pretrain/pretrain_spotlight_2_motif_list_caps_final.yaml",
                            "list_2m": "configs/datasets/app_pretrain/pretrain_spotlight_2_motif_list_caps_final_2m.yaml",
                            "gpt": "configs/datasets/app_pretrain/pretrain_spotlight_2_motif_gpt_caps.yaml",
                            "gpt_2m": "configs/datasets/app_pretrain/pretrain_spotlight_2_motif_gpt_caps_2m.yaml"}

@registry.register_builder("longitudinal_spotlight_stage2_caption")
class SpotlightCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = PretrainVQADataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/app_pretrain/pretrain_spotlight_2_longitudinal_list_caps.yaml",
                            "final": "configs/datasets/app_pretrain/pretrain_spotlight_2_longitudinal_list_caps_final.yaml",
                            "list_2m": "configs/datasets/app_pretrain/pretrain_spotlight_2_longitudinal_list_caps_final_2m.yaml",
                            "gpt": "configs/datasets/app_pretrain/pretrain_spotlight_2_longitudinal_gpt_caps.yaml",
                            "gpt_2m": "configs/datasets/app_pretrain/pretrain_spotlight_2_longitudinal_gpt_caps_2m.yaml"}

@registry.register_builder("conceptual_caption_3m")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m.yaml"
    }


@registry.register_builder("conceptual_caption_12m")
class ConceptualCaption12MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_12m.yaml"
    }


@registry.register_builder("sbu_caption")
class SBUCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sbu_caption/defaults.yaml"}


@registry.register_builder("vg_caption")
class VGCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption.yaml"}


@registry.register_builder("laion2B_multi")
class Laion2BMultiBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults_2B_multi.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
