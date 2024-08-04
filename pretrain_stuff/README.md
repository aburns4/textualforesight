# Data Setup and Storage

Here we provide information on how to set up the data needs for pretraining with Textual Foresight or other baselines as well as the data set up needed for finetuning in our framework. We release the processed data files for both for direct use. You also can download and reprocess the raw data if you are interested in modifying the steps we take to clean and curate the final data files.

All `yamls` under `LAVIS/lavis/configs/datasets/` contain fields for where the annotation files and images are stored. **PLEASE UPDATE THESE PATHS** to reflect where you download the data, it need not follow the same naming convention.

Pretraining data configs are under `LAVIS/lavis/configs/datasets/app_pretrain` and finetuning data configs for our downstream tasks are under `LAVIS/lavis/configs/datasets/rico`. These are referenced in the dataset builders located at `lavis/datasets/builders/caption_builder.py` and `lavis/datasets/builders/image_text_pair_builder.py` which allows for different training variants to be used for the same dataset.

Our example scripts provided under `run_scripts` include the appropriate yaml config for each experiment in our paper and set the data fields as needed.

## Pretraining Data
### Annotation Files
In our [released data](), we provide a folder for `processed_pretraining_data` which consists of json files with image caption pairs. The captions and additional fields vary by pretraining objective (element vs. element list vs. screen caption vs. textual foresight). These should be unzipped within this `pretrain_folder`. You can confirm their folder is correct by cross checking their annotation paths in their corresponding `yaml` - we provide a table below to help guide you to the correct yaml for comparison. Of course, you will likely need to update the yaml annotation storage paths to reflect your root directory.

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

### Pretraining Images
To pretrain with our provided annotations, we also have to store the images used for each sample. We provide the raw data from the Longitudinal and MoTIF prior work which contain the images. Simply download and unzip/untar the [pretrain raw data]() under `pretrain_stuff`. The following file structure should result:

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

### From Scratch

## Finetuning Data
We evaluate on four 

### Quick Download


