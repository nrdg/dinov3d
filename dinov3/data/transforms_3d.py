import monai
import torch
from monai.transforms import RandomizableTransform
import numpy as np
from typing import Union

class RandomResizedCrop3d(RandomizableTransform):
    def __init__(self,
                 prob: float = 1,
                 scale: tuple[float, float] = [0.5,1],
                 crop_size: int = 48,
                 antialias = False
                 ):
        self.prob = prob
        self.crop_size = crop_size
        self.scale = scale
        self.antialias = antialias

    def __call__(self,
                 img: Union[np.ndarray, torch.Tensor],
                 spacing: tuple,
                 ):
        '''
        Takes as argument a img, and spacing

        returns cropped volume from that img
        '''
        _, *spatial_dims = tensor.shape

        # get random value between crop size range
        scale = (self.scale[1] - self.scale[0]) * torch.rand() + self.scale[0]

        spacing = tuple(filter(lambda x: x != 0, spacing))
        assert len(spatial_dims) == 3, f"spatial_dims must be 3D {spatial_dims}"
        assert len(spacing) == 3, f"spacing must be 3D, got {spacing}"

        spacing_t = torch.tensor(spacing)
        voxel_count = torch.prod(torch.tensor(spatial_dims))
        physical_volume = voxel_count * torch.prod(spacing_t)

        target_phys_size = (physical_volume * scale) ** (1/3)

        min_vox_size = min(min(spatial_dims), 10)

        for axis in range(3):
            max_axis_size = spatial_dims[axis] * spacing_t[axis]
            if target_phys_size / spacing_t[axis] > spatial_dims[axis]:
                target_phys_size = max_axis_size

        voxel_crops = (target_phys_size / spacing_t).clamp(min=min_vox_size).int()
        max_starts = [d - c for d, c in zip(spatial_dims, voxel_crop)]
        starts = [
            torch.randint(0, ms, (1,)).item() if ms > 0 else 0
            for ms in max_starts
        ]

        cropped = tensor[
            :,
            starts[0]:starts[0] + voxel_crop[0],
            starts[1]:starts[1] + voxel_crop[1],
            starts[2]:starts[2] + voxel_crop[2]
        ]

        try:
            result = F.interpolate(
                cropped.unsqueeze(0),
                size=(self.crop_size, self.crop_size, self.crop_size),
                mode=interpolation,
                align_corners=True,
                antialias=self.antialias
            ).squeeze(0)
        except Exception as e:
            raise RuntimeError(f'failed to interpolate, spatial_dims: {spatial_dims}, spacing: {spacing}, min_vox_size: {min_vox_size}, passing error : {e}')

        # prob not needed cleanup here
        del spacing_t
        del voxel_count
        del physical_volume
        del target_phys_size
        del voxel_crop
        del max_starts
        del starts

        return result