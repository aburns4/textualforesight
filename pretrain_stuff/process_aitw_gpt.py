# -*- coding: utf-8 -*-
"""process_aitw.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SIE7DCq8EMvO5csxR5XycxGbc16744Vu

Copyright 2023 The Google Research Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
sys.path.append('./google-research')

from android_in_the_wild import visualization_utils
import argparse
import tensorflow as tf
import random
import json
import glob
import numpy as np
import time
import os
import random
import dask.bag as db
from dask.distributed import Client, LocalCluster, progress
from collections import defaultdict, Counter
from typing import List
from multiprocessing import freeze_support

GENERIC_WORDS = ['action', 'bar', 'menu', 'title', 'and', 'ans', 'app', 'icon', 'name',
                 'arg', 'background', 'element', 'btn', 'but', 'bottom', 'button', 'content',
                 'desc', 'text', 'item', 'empty', 'fab', 'image', 'grid', 'header', 'img',
                 'imgfile', 'lbutton', 'label', 'letter', 'list', 'view', 'pic', 'placeholder',
                 'random', 'row', 'single', 'raw', 'small', 'large', 'sub', 'template', 'navbar', 
                 'banner', 'test', 'textinput', 'error', 'texto', 'todo', 'toolbar', 'tool', 'track',
                 'txt', 'unknown', 'stub', 'web', 'left', 'right', 'tlb', 'nan', 'page', 'feature',
                 'menugrid', 'picture', 'tabs', 'number', 'node', 'iconimage', 'entity', 'webview',
                 'heading', 'logo', 'tbl', 'tab', 'primary', 'footer']

dataset_directories = {
    'general': 'gs://gresearch/android-in-the-wild/general/*',
    'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
    'install': 'gs://gresearch/android-in-the-wild/install/*',
    'single': 'gs://gresearch/android-in-the-wild/single/*',
    'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
}

parser = argparse.ArgumentParser('Create pretrain annotations for Spotlight objective')
parser.add_argument('--dataset',
                    type=str,
                    default="aitw",
                    help='specify which dataset to process')
parser.add_argument('--folder',
                    type=str,
                    default="elements_no_icon",
                    help='specify output_folder')
parser.add_argument('--include_icons',
                    type=bool,
                    default=False,
                    help='specify whether icon text should be included')
parser.add_argument('--dataset_subset',
                    type=str,
                    choices=["general", "google_apps", "install", "web_shopping"],
                    help='specify which dataset subset to process')
parser.add_argument('--start_range',
                    type=int,
                    default=0,
                    help='specify start of range to process')
parser.add_argument('--end_range',
                    type=int,
                    default=5000,
                    help='specify end of range to process')
parser.add_argument('--index',
                    type=int,
                    default=0,
                    help='specify split index of current dataset to process')

def is_good_ocr(field):
  toks = field.split(' ')
  only_gen = (len(set(toks).difference(set(GENERIC_WORDS))) == 0)
  single_or_empty_char = (len(field) <= 1)
  is_url = (len(toks) == 1 and 'http' in field)
  transformed_field = field.encode('unicode-escape').decode('ascii')
  is_alpha = all(x.isalpha() or x.isspace() for x in transformed_field) # TODO: I think I fixed this, not sure if I want to allow numbers though
  if (not only_gen) and (not single_or_empty_char) and (not is_url) and is_alpha:
      return True
  return False

def process_aitw_sample(exs, use_icons=False):
  samples = defaultdict(lambda: defaultdict(list))

  for ex in exs:
    app_id = ex.features.feature['current_activity'].bytes_list.value[0].decode('utf-8')
    ui_text = ex.features.feature['image/ui_annotations_text'].bytes_list.value
    ui_types = ex.features.feature['image/ui_annotations_ui_types'].bytes_list.value

    for ocr, ui_type in zip(ui_text, ui_types):
      if not ocr:
        if not use_icons:
          continue
        ui_type = ui_type.decode('utf-8')
        if "ICON" in ui_type:
          ocr = " ".join(ui_type.split("_")[1:]).lower()
        else:
          print(ui_type)
      else:
        ocr = ocr.decode('utf-8')

      if ocr: # and is_good_ocr(ocr):
        img_id = "_".join(
            [ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8'),
            str(ex.features.feature['step_id'].int64_list.value[0])])

        # print(ocr, ui_type)
        samples[app_id][img_id].extend([ocr])
  return samples

def main():
  start_time = time.time()

  global args
  args = parser.parse_args()

  dataset_name = args.dataset_subset
  filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
  assert args.start_range >= 0 and args.start_range <= args.end_range
  
  filenames = filenames[args.start_range : args.end_range]
  print(args.start_range, args.end_range, len(filenames))
  raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

  i = 0

  exs = []
  for d in raw_dataset:
    if i % 1000 == 0:
      print(i)
    
    ex = tf.train.Example()
    ex.ParseFromString(d)
    exs.append(ex)
    i+=1

  print('including icons %s' % args.include_icons)
  processed = process_aitw_sample(exs, args.include_icons)

  output_intermediate_dir = os.path.join('gpt_jsons', args.dataset, args.folder)
  if not os.path.exists(output_intermediate_dir):
      os.makedirs(output_intermediate_dir)
  
  fn = "_".join([dataset_name, str(args.index), 'pretrain.json'])
  with open(os.path.join(output_intermediate_dir, fn), 'w') as f:
      json.dump(processed, f)
  
  end_time = time.time()
  seconds = end_time - start_time
  minutes = seconds / 60
  hours = minutes / 60
  print('Time to process elements of %s %s samples: %.2f seconds / %.2f minutes / %.2f / hours' % (i, args.dataset, seconds, minutes, hours))

if __name__ == "__main__":
  freeze_support()
  main()