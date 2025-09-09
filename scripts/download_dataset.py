import json
import os
import argparse

from tqdm import tqdm
import boto3

dataset_json = "./dinov3mridataset.json"
mini_dataset_json = "./dinov3mridataset_mini.json"

s3_bucket = "dinov2preprocessedmridataset"
s3_client = boto3.client('s3')

def download_file(file_name, out_dir):
    s3_client.download_file(s3_bucket, file_name, os.path.join(out_dir, file_name))

def main(args):
    if args.mini:
        dataset_pth = mini_dataset_json
    else:
        dataset_pth = dataset_json

    with open(dataset_pth, "r") as f:
        dataset = json.load(f)

    for item in dataset:
        file_pth = item["image"]
        try:
            download_file(file_pth, args.out_dir)
        except Exception as e:
            print(f'subject {file_pth} failed with error: {e}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mini", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)