import json
import os
import argparse

import multiprocessing
from tqdm import tqdm
import boto3

dataset_json = "../dinov3mridataset.json"
mini_dataset_json = "../dinov3mridataset_mini.json"

s3_bucket = "dinov2preprocessedmridataset"

use_file_obj = True

def download_file(job_arg):
    bucket, file_name, out_dir = job_arg
    s3_client = boto3.client('s3')
    out_pth = os.path.join(out_dir, file_name)
    if use_file_obj:
        with open(out_pth, "wb") as f:
            s3_client.download_fileobj(bucket,
                                       file_name,
                                       f)
    else:
        s3_client.download_file(bucket, file_name, os.path.join(out_dir,
                                                                   file_name,
                                                                   out_pth))

def main(args):
    if args.mini:
        dataset_pth = mini_dataset_json
    else:
        dataset_pth = dataset_json

    with open(dataset_pth, "r") as f:
        dataset = json.load(f)


    if args.num_workers < 0:
        for item in tqdm(dataset):
            file_pth = item["image"]
            try:
                download_file((s3_bucket, file_pth, args.out_dir))
            except Exception as e:
                print(f'subject {file_pth} failed with error: {e}')
    if args.num_workers >= 0:

        jobs = [(s3_bucket, subject['image'], args.out_dir) for subject in dataset]

        num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else args.num_workers
        pool = multiprocessing.Pool(num_workers)
        pool.map(download_file, jobs)
        pool.close()
        pool.join()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mini", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)