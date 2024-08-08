# Data Setup and Storage

Here we provide information on how to set up the data needs for pretraining with Textual Foresight or other baselines as well as the data set up needed for finetuning in our framework. We release the processed data files for both for direct use. You also can download and reprocess the raw data if you are interested in modifying the steps we take to clean and curate the final data files.

All `yamls` under `LAVIS/lavis/configs/datasets/` contain fields for where the annotation files and images are stored. **PLEASE UPDATE THESE PATHS** to reflect where you download the data, it need not follow the same naming convention.

Pretraining data configs are under `LAVIS/lavis/configs/datasets/app_pretrain` and finetuning data configs for our downstream tasks are under `LAVIS/lavis/configs/datasets/rico`. These are referenced in the dataset builders located at `lavis/datasets/builders/caption_builder.py` and `lavis/datasets/builders/image_text_pair_builder.py` which allows for different training variants to be used for the same dataset.

Our example scripts provided under `run_scripts` include the appropriate yaml config for each experiment in our paper and set the data fields as needed.

## Pretraining Data
### (Quick Download) Download Annotation Files
<details close>
<summary>Expand for Instructions</summary>
<br>

In our [released data](https://drive.google.com/drive/folders/1JmSfh6AP0dpSMrNv-5koZgnQXrajxqS7?usp=sharing), we provide a folder for `processed_pretraining_data` which consists of json files with image caption pairs. The captions and additional fields vary by pretraining objective (element vs. element list vs. screen caption vs. textual foresight). These should be unzipped within this `pretrain_folder`. You can confirm their folder is correct by cross checking their annotation paths in their corresponding `yaml` - we provide a table below to help guide you to the correct yaml for comparison. Of course, you will likely need to update the yaml annotation storage paths to reflect your root directory.

This is what the final annotation folder structure should look like after unzipping the processed files.
```
pretrain_stuff/
    gpt_jsons/
        aitw/
            fortune_captions/
            gpt_captions/
        longitudinal/
            fortune_captions/
            gpt_captions/
        motif/
            fortune_captions/
            gpt_captions/
    spotlight_jsons/
        aitw/
            elem_list_captions_no_icon
            subsampled/
                stage1_post/
                stage2_post/
        longitudinal/
            elem_list_captions_final/
            subsampled/
                stage1/
                stage2/
        motif/
            elem_list_captions_final/
            subsampled/
                stage1/
                stage2/
```

`fortune_captions` refers to Textual Foresight pretraining data. At some point we considered naming the method Fortune (like fortune telling), which is why there's some outdated naming throughout. Feel free to rename if it causes confusion. `gpt_captions` refers to the screen captioning data generated with GPT 3.5 Turbo. under `spotlight_captions` we store element and element list captions, with the former being broken into stage 1 and stage 2 annotations under `subsampled`.
</details>

### (Quick Download) Download Pretraining Images
<details close>
<summary>Expand for Instructions</summary>
<br>

To pretrain with our provided annotations, we also have to store the images used for each sample. We provide the raw data from the Longitudinal and MoTIF prior work which contain the images. Simply download and unzip/untar the [raw pretrain data](https://drive.google.com/drive/folders/1rMdsgSDLlQvhechvicFfY3E2Yfw9YIOh?usp=drive_link) under `pretrain_stuff`. The following file structure should result:

```
pretrain_stuff/
    longitudinal/
        app.package.name
        ...
    motif/
        traces_02_14_21/
            app.package.name
            ...
        traces_03_17_21/
            app.package.name
            ...
        traces_05_11_2022/
            app.package.name
            ...
```

We do not store the full raw data for AITW and instead only store their images. We provide a script to fetch the AITW images and store them in a desired output folder: please run `get_aitw_images.sh`. We have to run this script multiple times due to the size of the dataset, so we provide arguments for which dataset subset and subset of files to process ("general", "google_apps", "install", "web_shopping" and then you can specify the number of files you want to process per split to break up the job).

Again, when finished storing the AITW images, **make sure to update** the `images` `storage` field in the corresponding yaml to ensure it is pointed to correctly.

| Pretraining Data Type          | Annotation File Path                                     | Dataset yaml <br> (under `lavis/configs/datasets/app_pretrain/` )    | Dataset Builder <br> (under `lavis/datasets/builders/`)               |
| ------------------------------ | -------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Element Captioning / Spotlight | `pretrain_stuff/spotlight_jsons/*/subsampled/*/*`        | `*_pretrain_stage2.yaml`| `caption_builder.py` <br> * `longitudinal_pretrain` (`stage2`)  <br> *  `aitw_pretrain` (`stage2`)  <br> * `motif_pretrain` (`stage2`)|
| Element List Captioning        | `pretrain_stuff/spotlight_jsons/*/elem_list_captions_*/*`| `pretrain_spotlight_2_*_list_caps_final.yaml`<br> `pretrain_spotlight_2_aitw_list_caps.yaml` | `image_text_pair_builder.py` <br> * `longitudinal_spotlight_stage2_caption` (`final`)  <br> * `aitw_spotlight_stage2_caption` (`default`)  <br> *  `motif_spotlight_stage2_caption` (`final`) |
| Screen Captioning              | `pretrain_stuff/gpt_jsons/*/gpt_captions/*`              | `pretrain_spotlight_2_*_gpt_caps.yaml`| `image_text_pair_builder.py` <br> * `longitudinal_spotlight_stage2_caption` (`gpt`)<br> * `aitw_spotlight_stage2_caption` (`gpt`) <br> * `motif_spotlight_stage2_caption` (`gpt`)|
`motif_spotlight_stage2_caption` (`final`) |
| Textual Foresight              | `pretrain_stuff/gpt_jsons/*/fortune_captions/*`          | `*_pretrain_stage2_fortune.yaml` | `caption_builder.py` <br> * `longitudinal_pretrain` (`stage2_fortune`) <br> * `aitw_pretrain` (`stage2_fortune`)  <br> * `motif_pretrain` (`stage2_fortune`) |
</details>

### From Scratch
<details close>
<summary>Expand for Instructions</summary>
<br>

If you are interested in reprocessing the pretraining datasets, we also include the code used to process, format, and annotate the files for pretraining. We note that this is a complicated multistep process per dataset due to make filters and reformattings being required. Keep in mind that for all steps, you likely need to change the input arguments to reflect your root path/directory on your own machine. Follow the below steps:

#### Spotlight / Element Captioning Pretraining Data
1. MoTIF and Longitudinal (follow the same steps for each with different dataset arguments)
    * Run `pretrain_stuff/get_spotlight_data.py` with `process_from_raw` set to `True`.
        * This results in the initial `raw/` output of element captions.
    * Rerun `pretrain_stuff/get_spotlight_data.py` with `process_from_raw` set to `False`.
        * This performs the final merging and thresholding (a Spotlight filter per their paper) which requires each element caption to occur at least 5 times. <br> This results in a `thresholded/` intermediate output.
    * Run `pretrain_stuff/merge_all_dataset_jsons.py`
        * This will result in the final `subsampled/stage1` and `subsampled/stage2` folders under the respective dataset folder within `pretrain_stuff`.
2. AITW (this is a complicated process due to the size of AITW and different formatting)
    * Run `pretrain_stuff/preprocess_aitw.py` with `process_from_raw` set to `True`.
        * This results in the initial `raw_aitw_by_sub_dataset/` output of element captions, as well as `counts` metadata concerning the frequency of each caption.
    * The counts output from the prior step was saved per dataset subset, so merge them by running `aitw_counts/join_counts.py`.
        * There should now be an `all_counts.json` file in the `aitw_counts` folder.
    * Rerun `pretrain_stuff/preprocess_aitw.py` with `process_from_raw` set to `False`.
        * This results in the next intermediate output `aitw_by_app_thresholded/`
    * Run `pretrain_stuff/spotlight_jsons/aitw/join_all_apps.py`
        * This reformats the intermediate jsons and saves the output in a `grouped_apps` folder.
    * Run `pretrain_stuff/merge_all_dataset_jsons.py` to obtain `subsampled/stage1/` and `subsampled/stage2/` folders.
    * Finally, we manually (apologies) split each file in half within their respect `subsampled/stage1/` and `subsampled/stage2/` folders. <br> Store these in `subsampled/stage1_post/` and `subsampled/stage2_post/` folders.
        * We do this because of how large the files are.
        * By manually, we mean on the command line - it should just be a couple lines of code.



#### Element List Captioning Pretraining Data
1. MoTIF and Longitudinal (follow the same steps for each with different dataset arguments)
    * Run `pretrain_stuff/get_elems_for_gpt.py`.
        * This results in the intermediate `elements_final/` output of element captions.
2. AITW
    * Run `pretrain_stuff/process_aitw_gpt.py`. This should result in the intermediate `aitw/elements_no_icon/` dataset folder.
3. Run `pretrain_stuff/make_caption_from_elems.py` (this handles the final formatting for all datasets)
    * It should result in the `aitw/elem_list_captions_no_icon`, `motif/elem_list_captions_final`, <br> and `longitudinal/elem_list_captions_final` data folders.

#### GPT / Screen Captioning Pretraining Data

1. Rerun `pretrain_stuff/process_aitw_gpt.py` with the output `folder` set to `elements_raw` and `include_icons` set to False.
    * This should result in the intermediate `aitw/elements_raw/` dataset folder.
    * We choose to use different element processing for GPT queries than that outlined by Spotlight, <br> which was used for element 
        and element list processing.
2. Run `gpt_jsons/elem_stats.py` which provides the samples to query GPT with (before the prompt formatting).
3. Run `gpt_jsons/gpt3_5_turbo_async.py` to query GPT and get pseudo caption outputs.
    * An example script is provided in `gpt_jsons/gpt_3.5_scc.sh`
    * These queries will store intermediate text outputs in a `gpt_jsons/gpt3_5_captions` folder
4. Finally, format the text files with `gpt_jsons/make_blip_caption_files.py` to get the final json files needed for training.
    * This must be done for each dataset (see the file's input arguments for reference)

#### Textual Foresight Pretraining Data
Note that to run the below steps from scratch you must first have gone through the screen <br>
captioning steps outlined above. Textual Foresight uses a subset of the screen captioning data
curated with GPT 3.5 Turbo.

1. Run `pretrain_stuff/get_state_action_triplets.py` for each dataset (AITW, Longitudinal, MoTIF).
    * This will result in `triplets.txt` files under each respective dataset folder in `spotlight_jsons`.
2. Run `gpt_jsons/get_diff_st_s1_pairs.py`  for each dataset (AITW, Longitudinal, MoTIF). <br> This provides an extra level of cleaning to ensure valid state, action, next state triplets.
    * This will result in `triplets_clean.txt` files under each respective dataset folder in `spotlight_jsons`.
3. Run `gpt_jsons/get_fortune_samples_to_be_captioned.py`.
    * This will result in a txt file `fortune_set_samples.txt` later to be used in finding the subset of GPT captions <br> that
    can be used for Textual Foresight pretraining.
4. Finally, run `gpt_jsons/make_fortune_caption_files.py` to obtain the json files under each dataset.
</details>

## Finetuning Data
###  Quick Download
<details close>
<summary>Expand for Instructions</summary>
<br>

We evaluate on four downstream tasks: screen summarization, element captioning, tappability prediction, and language command grounding. We provide the already processed/formatted annotation files for you to download [here](https://drive.google.com/drive/folders/1-gl4qjixf8uSMJXb8yQnxWiHwxUN3vwp?usp=drive_link), and then all that is left to do is (1) download the images associated with these downstream tasks and (2) update the dataset yamls to reflect where you choose to store the annotation and image files.

All downstream tasks are annotated on top of the `Rico` dataset. Download the [raw data](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz) which contains the images needed for finetuning and make sure to **UPDATE** the `images` `storage` field to reflect the root path of where you store the images. The rico data is zipped in a folder named `combined/`. Again, the annotations paths should also be updated to reflect the folders you store the data in.

| Finetuning Task                 | Annotation File Path                                     | Dataset yaml <br> (under `lavis/configs/datasets/rico/`)  | Dataset Builder <br> (under `lavis/datasets/builders/`)   |
| ------------------------------- | -------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| Screen Summarization/Captioning | * `screen2words/train.json` <br> * `screen2words/val.json`  <br> * `screen2words/test.json` | `screen_summarization.yaml`  | `caption_builder.py` <br> * `screen_caption` (`default`)  |
| Element/Widget Captioning       | * `widget-caption/train.json` <br> * `widget-caption/dev.json`<br> *  `widget-caption/test.json` | `widget_vqa.yaml`                                         | `caption_builder.py` <br> * `widget_vqa` (`default`)      |
| Tappability Prediction          | * `taperception/train_4_tap_caption.json` <br> * `taperception/eval_4_tap_caption.json` <br> * `taperception/test_tap_caption.json` | `tappability_caption_4_vqa.yaml`  | `caption_builder.py` <br> * `tap_vqa` (`caption_quad`)    |
| Language Grounding              |  * `mug/mug_captions_full_instr_train.json` <br> * `mug/mug_captions_full_instr_eval.json` <br> * `mug/mug_captions_2_test.json`    | `language_grounding_captions_all_instr.yaml` <br >`language_grounding_captions_eval.yaml` | `caption_builder.py` <br> * `language_ground` (`captions_full`) <br> * `language_ground_caption_eval` (`default`) |

Finally, we note that there are additional val/test files used during training that require a different format to be compatible with MSCOCO eval and metric packages. We note these files below which should also lie under the respective finetuning dataset folder (they are included in the files for download, we just want to point them out). In the existing BLIP2 codebase, these filenames are hardcoded within task files (see `LAVIS/lavis/tasks/vqa.py`).

Below we provide the filenames for reference and where they are hardcoded.
| Finetuning Task                 | COCO Formatted Annotation Files                          | Where they are hardcoded (under `LAVIS/lavis/tasks/`) |
| ------------------------------- | -------------------------------------------------------- | ------------------------ |
| Screen Summarization/Captioning | `eval_val.json` <br> `eval_test.json` | `captioning.py` |
| Element/Widget Captioning       | `eval_dev.json` <br> `eval_test.json` | `vqa.py` |
| Tappability Prediction          | `tap_captions_eval_coco.json` <br> `tap_captions_test_coco.json` | `vqa.py` |
| Language Grounding              | `mug_captions_full_instr_eval_coco.json` | `vqa.py` |
</details>

### From Scratch

<details close>
<summary>Expand for Instructions</summary>
<br>

If you are interested in reprocessing the original downstream task datasets, we also include the code used to format the files for finetuning. Follow the below steps and inspect each file to see how to set input arguments or set file names etc. (and feel free to ask any clarifying questions here on the repo):

1. Run `LAVIS/modify_task_annotations.py` to format the raw data from each respective dataset.
2. Then, follow the below steps below
    * Screen Summarization
        * Run `screen2words/eval_anns_format.py`
    * Widget Captioning
        * Run `widget-caption/make_split_json.py`
        * Then, run `widget-caption/eval_anns_format.py`
    * Tappability Prediction
        * Run `taperception/make_split_json.py`
        * \[Optional\] Run `taperception/make_higher_ratio.py` to upsample the not-tappable class samples. <br> We upsampled by a factor of 4 due to the class imbalance and use the resulting files for finetuning as mentioned in the paper.
        * Then, run `taperception/eval_anns_format.py`
    * Language Grounding
        * Run `mug/mug_to_caption_annotations.py`
        * Then, run `mug/eval_anns_format.py`
</details>


## Evaluation Metrics

<details close>
<summary>Expand for Instructions</summary>
<br>
Screen summarization and element captioning tasks have built in captioning metrics which are included in the COCO eval setup during training, but if you wish to additionally report more recent metrics like we do (e.g., BERTScore and BLEURT), you will need to run the final eval script below to obtain those additional metrics. Additionally, tappability and language grounding are evaluating as a captioning task during training (please find more details in the paper and Appendix), but are evaluated with metrics like accuracy and F1 score at test time.

<br>

Below we provide pointers for the metric files and scripts to run for each downstream task to obtain the full suite of metrics. Again, note that the flags should be updated to reflect your file output paths and which task you're evaluating.

| Finetuning Task                 | Metric Computation File (under `LAVIS/lavis/`)    | Example Script to Run Metric File (under `LAVIS/lavis/`) |
| ------------------------------- | ------------------------------------------------ | ------------------------ |
| Screen Summarization/Captioning | `output/BLIP2/fresh_metric.py`                   | `output/BLIP2/gpu_metric.sh` |
| Element/Widget Captioning       | `output/BLIP2/fresh_metric.py`                   | `output/BLIP2/gpu_metric.sh` |
| Tappability Prediction          | `output/BLIP2/fresh_metric.py`                   | `output/BLIP2/gpu_metric.sh` |
| Language Grounding              | `output/language_ground_caption_eval.py`         | `output/lang_ground_gpu_metric.sh` |
</details>