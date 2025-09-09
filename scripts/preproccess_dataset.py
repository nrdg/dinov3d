import os
import argparse
from monai.data import PersistentDataset
import nibabel as nib
import boto3
import tempfile
import json
import ray


from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    SignalFillEmptyd,
    ToNumpyd,
    ToNumpy
)

def make_dataset(base_dir, dataset_json, transform):
    with open(dataset_json, "r") as f:
        dataset = json.load(f)

    for subject in dataset:
        subject["image"] = os.path.join(base_dir, subject["image"])

    dataset = PersistentDataset(dataset,cache_dir=None, transform=transform)
    return dataset


@ray.remote
class ScanProcessor:
    def __init__(self, s3_source_bucket, s3_target_bucket, dataset_json):

        with open(dataset_json, "r") as f:
            dataset_json = json.load(f)

        self.dataset = PersistentDataset(dataset_json, cache_dir=None, transform=None)
        self.s3_source_bucket = s3_source_bucket
        self.s3_target_bucket = s3_target_bucket
        self.s3 = boto3.client('s3')

        self.transform = Compose([
            LoadImaged(keys=["image"]),
            SignalFillEmptyd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToNumpyd(keys=["image"])
        ])


    def __call__(self, subject_index):
        subject = self.dataset[subject_index]

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                md5_filename = subject['image']
                local_filename = os.path.join(tmpdir, f"{subject['image']}.nii")
                self.s3.download_file(self.s3_source_bucket, subject['image'], local_filename)

                subject['image'] = local_filename
                transformed_arr = self.transform(subject)['image']
                orig_img = nib.load(subject["image"])

                out_nifti_img = nib.Nifti1Image(transformed_arr,
                                                affine=orig_img.affine,
                                                header=orig_img.header)

                local_preproc_filename = os.path.join(tmpdir, f'preproc_{md5_filename}')
                nib.save(out_nifti_img, local_preproc_filename)
                self.s3.upload_file(local_preproc_filename, self.s3_target_bucket, md5_filename)

                return True
        except Exception as e:
            print(f'error {e}')

            return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_s3', type=str)
    parser.add_argument('--output_s3', type=str)
    parser.add_argument('--dataset_json', type=str)
    parser.add_argument('--num_workers', type=int, default=10)

    args = parser.parse_args()
    return args

def main(args):

    with open(args.dataset_json, "r") as f:
        dataset_read_for_len = json.load(f)

    num_subjects = len(dataset_read_for_len)

    items = [i for i in range(num_subjects)]

    actors = [ScanProcessor.remote(args.input_s3,
                                   args.output_s3,
                                   args.dataset_json)
              for _ in range(args.num_workers)]

    futures = []
    for i, item in enumerate(items):
        actor = actors[i % args.num_workers]
        futures.append(actor.__call__.remote(item))

    results = ray.get(futures)

if __name__ == "__main__":
    args = parse_args()
    main(args)