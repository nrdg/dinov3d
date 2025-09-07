# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import random
import torch
import torch.nn.functional as F
from dinov3.data import RandomResizedCrop3d

from monai.transforms import (
    Compose,
    Resize,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandFlip,
    RandRotate90d,
    SignalFillEmptyd,
    RandGaussianSmooth,
    RandGaussianNoise,
    RandBiasField,
    RandHistogramShift,
    RandGibbsNoise,
    ToTensor,
)



logger = logging.getLogger("dinov3")


class DataAugmentation3d(object):
    def __init__(
        self,
        keys: list[str],
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=96,
        local_crops_size=48,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        horizontal_flips=True,
        mean=0,
        std=1,
    ):
        '''
        This class is a wrapper around the monai transforms to apply data augmentation to a 3D MRI image.

        make sure the global and local transforms are non-dictonary transforms.
        '''

        if len(keys) != 1:
            raise ValueError(f"DataAugmentation3d only supports a single"
                             "key, recived: {keys}")
        else:
            self.key = keys[0]

        if local_crops_subset_of_global_crops:
            # this argument is not used in any of the training stages configs here, so we will skip it
            raise NotImplementedError("Local crops subset of global crops not implemented for 3d")

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        if local_crops_subset_of_global_crops:
            raise NotImplementedError("local crop subset of global crops not "
                                      "implement for 3d, as it is not used in training")

        logger.info("###################################")
        logger.info("Using 3d data augmentation parameters:")
        logger.info(f"keys: {keys}")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # here we separate cropping and other transforms to keep track of spacing
        self.global_cropper = RandomResizedCrop3d(
            crop_size=global_crop_max_size,
            scale=global_crops_scale
        )

        self.geometric_augmentation_global = Compose(
            [
                RandFlip(prob=0.5 if horizontal_flips else 0.0)
            ]
        )


        resize_global = nn.Identity()
        self.resize_global_post_transf = (
            nn.Identity()
        )

        self.resize_gram_teacher = None
        if gram_teacher_crops_size is not None:
            if gram_teacher_no_distortions:
                resize_global = Resize(
                    global_crops_size
                )
            else:
                self.resize_global_post_transf = Resize(
                    gram_teacher_crops_size, #maybe set interpolation method?
                )

            self.resize_gram_teacher = Resize(
                gram_teacher_crops_size #maybe set interpolation method
            )

        self.local_cropper = RandomResizedCrop3d(
            crop_size=local_crops_size,
            scale=local_crops_scale
        )

        self.geometric_augmentation_local = Compose(
            [
                RandGaussianSmooth(prob=0.1),
                RandGaussianNoise(prob=0.1),
                RandBiasField(prob=0.1),
                RandHistogramShift(prob=0.1),
                ToTensor(),
                RandFlip(prob=0.5 if horizontal_flips else 0.0)
            ]
        )

        global_transfo1_extra = Copose([
            RandGaussianSmooth(p=1),
            RandGaussianNoise(p=0.1),
            RandGibbsNoise(p=0.2)
        ])

        global_transfo2_extra = Compose([
            RandGaussianSmooth(p=0.1),
            RandGaussianNoise(p=0.2),
            RandGibbsNoise(p=0.1)
        ])

        local_transfo_extra = RandGaussianSmooth(p=0.5)

        self.normalize = ToTensor()

        self.global_transfo1 = Compose([
            resize_global, global_transfo1_extra, self.normalize
        ])
        self.global_transfo2 = Compose([
            resize_global, global_transfo2_extra, self.normalize
        ])
        self.local_transfo = Compose([
            local_transfo_extra, self.normalize
        ])

    def __call__(self, image_dict):
        output = {}
        output["weak_flag"] = True # not sure what this does

        spacing = image_dict['spacing']
        image = image_dict[self.key]

        im1_base = self.global_cropper(image, spacing)
        im1_base = self.geometric_augmentation_global(im1_base) # maybe needs spacing arg
        global_crop_1_trasf = self.global_transfo1(img1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_trasf)

        im2_base = self.global_cropper(image, spacing)
        im2_base = self.geometric_augmentation_global(im1_base) # maybe needs spacing arg
        global_crop_2_trasf = self.global_transfo2(img2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_trasf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        output["global_crops_teacher"] = [img1_base, img2_base] # already normalized, not sure tho

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normlaize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_trasf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_trasf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(self.local_cropper(image, spacing))) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()