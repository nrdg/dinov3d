import os
import nibabel as nib
import json
import argparse
import boto3
import tempfile
from tqdm import tqdm
import ray


@ray.remote
def extract_metadata_remote(key_path_pair, bucket_name):
    """Remote function for extracting metadata from a single file"""
    key = key_path_pair['key']
    local_path = key_path_pair['local_path']

    s3 = boto3.client('s3')
    s3.download_file(bucket_name, key, local_path)

    nib.load(local_path)
    header = nib.load(local_path).header
    spacing = [float(x) for x in header.get_zooms()]
    shape = [int(x) for x in header.get_data_shape()]
    base_name = os.path.basename(key)

    os.remove(local_path)

    return {
        'image': base_name,
        'spacing': spacing,
        'shape': shape,
    }


def extract_metadata(key_path_pair, bucket_name):
    key = key_path_pair['key']
    local_path = key_path_pair['local_path']

    s3 = boto3.client('s3')
    s3.download_file(bucket_name, key, local_path)

    nib.load(local_path)
    header = nib.load(local_path).header
    spacing = [float(x) for x in header.get_zooms()]
    shape = [int(x) for x in header.get_data_shape()]
    base_name = os.path.basename(key)

    os.remove(local_path)

    return {
        'image': base_name,
        'spacing': spacing,
        'shape': shape,
    }


def main(args):
    ray.init(num_cpus=args.num_cpus)

    temp_dir = tempfile.mkdtemp()

    s3_client = boto3.client('s3')
    s3_resource = boto3.resource('s3')

    bucket = s3_resource.Bucket(args.bucket)

    paginator = s3_client.get_paginator('list_objects_v2')

    key_path_pairs = []

    for page in tqdm(paginator.paginate(Bucket=args.bucket), desc="Listing files"):
        for obj in page.get('Contents', []):
            key = obj['Key']
            local_path = os.path.join(temp_dir, key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            key_path_pairs.append({'key': key, 'local_path': local_path})

    # Limit to 1000 samples if test mode is enabled
    if args.test:
        key_path_pairs = key_path_pairs[:1000]
        print(f"Test mode: Processing only {len(key_path_pairs)} files...")
    else:
        print(f"Processing {len(key_path_pairs)} files in parallel...")

    # Option 1: Use ray.wait() with progress tracking
    futures = [extract_metadata_remote.remote(pair, args.bucket) for pair in key_path_pairs]

    results = []
    with tqdm(total=len(futures), desc="Extracting metadata") as pbar:
        while futures:
            done_id, futures = ray.wait(futures, num_returns=1)
            result = ray.get(done_id[0])
            results.append(result)
            pbar.update(1)

    import shutil
    shutil.rmtree(temp_dir)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate monai format .json from s3 bucket"
    )

    parser.add_argument('--bucket')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--output', default=None)
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--test', action='store_true', help='Run in test mode with only 1000 samples')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.bucket + "_dataset.json"

    main(args)