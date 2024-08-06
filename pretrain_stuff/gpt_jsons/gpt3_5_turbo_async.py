import argparse
import os
import pandas as pd
import re
import openai
import signal
import torch

from contextlib import contextmanager
from functools import partial
from torch.multiprocessing import Pool, Process, set_start_method
from retry import retry

parser = argparse.ArgumentParser('')
parser.add_argument('--input_path',
                    type=str,
                    default="/projectnb2/ivc-ml/aburns4/LAVIS/pretrain_stuff/gpt_jsons/set_samples_all_final.txt",
                    help='path to read processed elements from')
parser.add_argument('--output_path',
                    type=str,
                    default="gpt3_5_outputs.txt",
                    help='path to write gpt captions to')
parser.add_argument('--output_folder',
                    type=str,
                    default="gpt3_5_captions",
                    help='path to write gpt captions to')
parser.add_argument('--increment',
                    type=int,
                    default=650,
                    help='specify number of samples to process per batch')
parser.add_argument('--increment_fraction',
                    type=int,
                    default=4,
                    help='specify fraction of batch to process when exception occurs')
parser.add_argument('--start_range',
                    type=int,
                    default=0,
                    help='specify start of range to get captions for. can be used to resume part way through')
parser.add_argument('--end_range',
                    type=int,
                    default=10000,
                    help='specify end of range to get captions for. can be used to resume part way through')
parser.add_argument('--num_workers',
                    type=int,
                    default=12,
                    help='number of workers, which should be set to num cpu cores requestes')

DEFAULT_PROMPT = "The following is an Android mobile app package name: %s.\nThere is a screen for this app with the following elements: %s.\nWhat does the screen show? Describe the app screen based on the functionality of the elements and use the mobile app category as context. Write a single sentence with fewer than 15 words. Remain objective, be specific, and do not repeat the app package name. Fill in the blank, the screen shows ____"

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Ensure the 'spawn' method is used for creating processes (required on some platforms)
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# Define a worker function that uses PyTorch and additional arguments
@retry(tries=5, delay=1, backoff=2)
def worker(sample_elems, prompt=DEFAULT_PROMPT):
    app, elems = sample_elems
    # try:
    # NOTE: some requests might fail, so it's better to add a try-except block to handle failures
    # with time_limit(5):
    messages=[{"role": "user", "content": prompt % (app, elems)}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=60,
    )

    caption = response['choices'][0]['message']['content'].strip()
    gpt_output = (app, elems, caption)
    # except:
    #     gpt_output = (dataset, app, elems, "TIME OUT EXCEPTION")
    
    return gpt_output

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    # Create a list of tasks
    with open(args.input_path) as f: 
        all_sample_elems = f.readlines()
        total_samples = len(all_sample_elems)
        all_sample_elems = [x.strip().split("[*]") for x in all_sample_elems][args.start_range:args.end_range]
    print("Global start to end range of samples %d : %d" % (args.start_range, min(args.end_range, total_samples)))
    
    write_path = os.path.join(args.output_folder, "_".join([str(args.start_range), str(args.end_range), args.output_path]))
    print("Writing to... %s" % write_path)
    for i in range(0, len(all_sample_elems), args.increment):
        start = i
        end = i + args.increment 
        curr_sample_elems = all_sample_elems[start : end]

        # Create a multiprocessing pool with a specified number of processes
        num_processes = args.num_workers
        try:
            with Pool(processes=num_processes) as pool:
                results = pool.map(worker, curr_sample_elems)
            pool.close()
            pool.join()

            # Now, 'results' contains the results of processing each item using PyTorch with additional arguments
            done = ["[*]".join(x) for x in results]
            to_write = "\n".join(done)
            if os.path.exists(write_path):
                to_write = "\n" + to_write
            
            with open(write_path, "a") as f:
                f.write(to_write)
            print("Completed batch %d : %d" % (args.start_range + start, min(args.end_range, args.start_range + end)))
        except:
            print("Entered exception.... splitting batch into %d parts" % args.increment_fraction)
            for j in range(args.increment_fraction):
                with Pool(processes=num_processes) as pool:
                    results = pool.map(worker,
                        curr_sample_elems[j * int(args.increment/args.increment_fraction) : (j+1) * int(args.increment/args.increment_fraction)])
                pool.close()
                pool.join()

                # Now, 'results' contains the results of processing each item using PyTorch with additional arguments
                done = ["[*]".join(x) for x in results]
                to_write = "\n".join(done)
                if os.path.exists(write_path):
                    to_write = "\n" + to_write
                
                with open(write_path, "a") as f:
                    f.write(to_write)
            print("Completed batch %d : %d" % (args.start_range + start, min(args.end_range, args.start_range + end)))
