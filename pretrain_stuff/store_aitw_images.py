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
import dask.bag as db
from dask.distributed import Client, LocalCluster, progress
from collections import defaultdict, Counter
from typing import List
import io
import PIL.Image as Image

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
parser.add_argument('--dataset_subset',
                    type=str,
                    choices=["general", "google_apps", "install", "web_shopping"],
                    help='specify which dataset subset to process')
parser.add_argument('--image_output_path',
                    type=str,
                    default="/projectnb/ivc-ml/aburns4/aitw_images",
                    help='where to store image byte files')
parser.add_argument('--start_range',
                    type=int,
                    default=0,
                    help='specify start of file range to process')
parser.add_argument('--end_range',
                    type=int,
                    default=1100,
                    help='specify end of file range to process')

def convert_bytes(b, savepath, im_h, im_w, nchannels):
    x = tf.io.decode_raw(b, out_type=tf.uint8)
    x = tf.reshape(x, (im_h, im_w, nchannels)).numpy()
    img = Image.fromarray(x, 'RGB')
    img.save(savepath)

def __main__():
    global args
    args = parser.parse_args()
    
    dataset_name = args.dataset_subset
    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
    assert args.start_range >= 0 and args.start_range <= args.end_range
    
    filenames = filenames[args.start_range : args.end_range]
    print(args.start_range, args.end_range, len(filenames))
    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

    i = 0
    for d in raw_dataset:
        if i % 1000 == 0:
            print(i)
        i += 1
        
        ex = tf.train.Example()
        ex.ParseFromString(d)
      
        file_map = "_".join([ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8'),
                            str(ex.features.feature['step_id'].int64_list.value[0])])

        if not os.path.exists(args.image_output_path):
            os.makedirs(args.image_output_path)
      
        save_path = os.path.join(args.image_output_path, file_map + ".jpg")
        if os.path.exists(save_path):
            continue

        # Write image to file
        im_bytes = ex.features.feature['image/encoded'].bytes_list.value[0]
        im_w = ex.features.feature['image/width'].int64_list.value[0]
        im_h = ex.features.feature['image/height'].int64_list.value[0]
        im_channels = ex.features.feature['image/channels'].int64_list.value[0]
        convert_bytes(im_bytes, save_path, im_h, im_w, im_channels)


__main__()