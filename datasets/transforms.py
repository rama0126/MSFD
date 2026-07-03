"""Some extra transforms for video"""

import bisect
from dataclasses import InitVar
import imp
import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import math

import torch.nn as nn
from tqdm import tqdm
import pickle
# datasets/transforms.py

import torch
import random
import numpy as np
from PIL import Image


import torch
import torch.nn as nn

class GridShuffle(nn.Module):
    """
    입력 텐서(이미지 또는 이미지 배치)를 그리드로 나눈 뒤, 그리드를 무작위로 섞습니다.
    이 변환은 배치의 모든 이미지에 동일한 셔플 패턴을 적용합니다.

    Args:
        grid_size (int): 이미지를 나눌 정사각형 그리드의 한 변의 크기 (픽셀 단위).
    """
    def __init__(self, grid_size: int):
        super().__init__()
        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError("grid_size must be a positive integer.")
        self.grid_size = grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 셔플을 적용할 입력 텐서. 
                              예상 모양: (B, C, H, W) 또는 (C, H, W)
                              B: 배치 크기, C: 채널, H: 높이, W: 너비

        Returns:
            torch.Tensor: 그리드가 셔플된 텐서. 입력과 동일한 모양을 가집니다.
        """
        # 입력이 단일 이미지(C, H, W)인 경우, 배치 차원(B=1)을 추가해 처리
        is_single_image = len(x.shape) == 3
        if is_single_image:
            x = x.unsqueeze(0)

        # 1. 입력 텐서의 크기 확인
        batch_size, channels, height, width = x.shape
        
        # 높이와 너비가 grid_size로 나누어 떨어지는지 확인
        if height % self.grid_size != 0 or width % self.grid_size != 0:
            raise ValueError(f"Image dimensions ({height}, {width}) must be divisible by grid_size ({self.grid_size}).")

        # 2. 텐서를 그리드 단위로 재구성
        # (B, C, H, W) -> (B, C, num_grid_h, grid_size, num_grid_w, grid_size)
        num_grid_h = height // self.grid_size
        num_grid_w = width // self.grid_size
        
        # view를 통해 텐서를 그리드 구조로 분리
        x_reshaped = x.view(batch_size, channels, num_grid_h, self.grid_size, num_grid_w, self.grid_size)

        # 3. 그리드 차원을 앞으로 모으기 위해 차원 순서 변경
        # (B, C, num_grid_h, grid_size, num_grid_w, grid_size) -> (B, C, num_grid_h, num_grid_w, grid_size, grid_size)
        # contiguous()는 permute 이후 메모리 구조를 연속적으로 만들어 다음 view 연산을 가능하게 함
        x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

        # 4. 셔플을 위해 모든 그리드를 하나의 차원으로 펼치기
        # (B, C, num_grid_h, num_grid_w, grid_size, grid_size) -> (B, C, num_grids, grid_size, grid_size)
        num_grids = num_grid_h * num_grid_w
        flat_grids = x_permuted.view(batch_size, channels, num_grids, self.grid_size, self.grid_size)

        # 5. 그리드 인덱스를 무작위로 섞기
        # 0부터 num_grids-1 까지의 인덱스를 섞어 새로운 순서 생성
        # 이 순서는 배치의 모든 이미지에 동일하게 적용됨
        permuted_indices = torch.randperm(num_grids, device=x.device)
        
        # 생성된 순서에 따라 그리드 재정렬
        shuffled_flat_grids = flat_grids[:, :, permuted_indices, :, :]

        # 6. 셔플된 그리드를 다시 이미지 형태로 복원 (재구성의 역순)
        # (B, C, num_grids, grid_size, grid_size) -> (B, C, num_grid_h, num_grid_w, grid_size, grid_size)
        shuffled_grids = shuffled_flat_grids.view(batch_size, channels, num_grid_h, num_grid_w, self.grid_size, self.grid_size)

        # (B, C, num_grid_h, num_grid_w, grid_size, grid_size) -> (B, C, num_grid_h, grid_size, num_grid_w, grid_size)
        shuffled_permuted = shuffled_grids.permute(0, 1, 2, 4, 3, 5).contiguous()

        # (B, C, num_grid_h, grid_size, num_grid_w, grid_size) -> (B, C, H, W)
        shuffled_tensor = shuffled_permuted.view(batch_size, channels, height, width)
        
        # 입력이 단일 이미지였다면, 배치 차원을 다시 제거
        if is_single_image:
            shuffled_tensor = shuffled_tensor.squeeze(0)

        return shuffled_tensor
def get_mask(mask_num):
    if mask_num == 0:
        mask = np.ones((224, 224), np.float32)
        return mask
    n_holes = random.randint(1, mask_num)
    
    while 1:
        mask = np.ones((224, 224), np.float32)
        for n in range(n_holes):
            length = np.random.randint(1, 224)
            width = np.random.randint(1, 224)
            y = np.random.randint(224)
            x = np.random.randint(224)
            y1 = np.clip(y - length // 2, 0, 224)
            y2 = np.clip(y + length // 2, 0, 224)
            x1 = np.clip(x - width // 2, 0, 224)
            x2 = np.clip(x + width // 2, 0, 224)
            mask[y1: y2, x1: x2] = 0.
        s = np.sum(mask == 0)
        if s > 0.3 * 224 * 224 and s < 0.7 * 224 * 224:
            break
    return mask

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()
class FTVideoRandomHorizontalFlip(object):
    """ Horizontal flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.
        
        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([3])

        return video
class RandomScale(object):

    
    def __call__(self, pil_img, selected_size):
        """
        Args:
            pil_img: PIL image to resize.
            selected_size: size to resize.
        
        Returns:
            PIL image: Randomly resized image.
        """
        pil_img = pil_img.resize((selected_size, selected_size))
        return pil_img
class RandomCrop(object):
    """ randomly crop the given PIL image."""
    """ Args:
        size (int): size of the crop.
        
    """ 
    def __call__(self, pil_img, crop_center, crop_size):
        """
        Args:
            pil_img: PIL image to crop.
            crop_center: center of the crop.
            crop_size: size of the crop.
        
        Returns:
            PIL image: Randomly cropped image.
        """
        
        pil_img = pil_img.crop((crop_center-crop_size//2, crop_center-crop_size//2, crop_center+crop_size//2, crop_center+crop_size//2))

        return pil_img     

def to_tensor(clip):
    """
    Cast numpy type to float, then permute dimensions from TxHxWxC to CxTxHxW, and finally divide by 255

    Parameters
    ----------
    clip : torch.tensor
        video clip
    """
    return torch.from_numpy(clip.float().permute(3, 0, 1, 2) / 255.0)


def normalize(clip, mean, std):
    """
    Normalise clip by subtracting mean and dividing by standard deviation

    Parameters
    ----------
    clip : torch.tensor
        video clip
    mean : tuple
        Tuple of mean values for each channel
    std : tuple
        Tuple of standard deviation values for each channel
    """
    clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        return normalize(clip, self.mean, self.std)


class ToTensorVideo:
    def __init__(self):
        pass

    def __call__(self, clip):
        return to_tensor(clip)
