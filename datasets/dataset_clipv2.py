import os
import random
import hashlib
import pickle
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

from datasets.transforms import gkern, FTVideoRandomHorizontalFlip, RandomScale, RandomCrop, get_mask
ROOT = "/workspace/datasets" # placeholder for dataset root path
CACHE_DIR = f"{ROOT}/.clip_cache" 
class VideoClips(Dataset):
    """Dataset class that yields fixed-length clips of frames from videos,
    supports train/test modes, memory buffering, dual sampling (task vs memory),
    and returns video indices for video-level metrics."""
    def __init__(
        self,
        paths,
        frames_per_clip=32,
        grayscale=False,
        transform=Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        max_frames_per_video=270,
        is_train=True,
        memory_list=None,
        memory_dual=False,
        frame_size=224,
        domain=0,
    ):
        self.paths = paths  # list of (video_dir, label_str)
        self.frames_per_clip = frames_per_clip
        self.grayscale = grayscale
        self.transform = transform
        self.is_train = is_train
        self.frame_size = frame_size
        self.memory_list = memory_list or []
        self.memory_dual = memory_dual
        self.memory_len = len(self.memory_list)
        self.domain = domain

        # Precompute augmentation transforms if training
        if self.is_train:
            self.horizon_flip = FTVideoRandomHorizontalFlip()
            self.random_scale = RandomScale()
            self.random_crop = RandomCrop()
            # mask parameter controls number of holes
            self.mask_param = 3

        # Build clip metadata (with disk cache to avoid repeated os.listdir)
        self.clip_meta, self.clips_per_video = self._build_clip_meta(
            self.paths, max_frames_per_video, frames_per_clip, domain
        )

        # Append memory entries as separate clips if not dual sampling
        if not self.memory_dual and self.memory_list:
            for mem_meta in self.memory_list:
                entry = mem_meta.copy()
                entry['is_memory'] = True
                if 'domain' not in entry:
                    entry['domain'] = None
                self.clip_meta.append(entry)

    def _build_clip_meta(self, paths, max_frames_per_video, frames_per_clip, domain):
        # cache key: hash of (paths, max_frames_per_video, frames_per_clip)
        key_src = str(paths) + str(max_frames_per_video) + str(frames_per_clip)
        cache_key = hashlib.md5(key_src.encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        os.makedirs(CACHE_DIR, exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                clip_meta, clips_per_video = pickle.load(f)
            # domain 필드는 캐시 후에도 현재 domain으로 덮어씀
            for m in clip_meta:
                if not m.get('is_memory', False):
                    m['domain'] = 2 * domain + m['label']
            return clip_meta, clips_per_video

        clip_meta = []
        clips_per_video = []
        for vid_idx, (vpath, vlabel) in enumerate(paths):
            vpath = vpath.format(ROOT=ROOT)
            frames = sorted(os.listdir(vpath))[:max_frames_per_video]
            num_frames = len(frames)
            if num_frames < frames_per_clip:
                print(f"Skipping {vpath}: only {num_frames} frames (<{frames_per_clip})")
                clips_per_video.append(0)
                continue
            n_clips = num_frames // frames_per_clip
            clips_per_video.append(n_clips)
            for c in range(n_clips):
                start = c * frames_per_clip
                clip_files = frames[start:start + frames_per_clip]
                clip_meta.append({
                    'video_idx': vid_idx,
                    'path': vpath,
                    'label': vlabel,
                    'frame_files': clip_files,
                    'start_idx': start,
                    'is_memory': False,
                    'domain': 2 * domain + vlabel
                })

        with open(cache_path, 'wb') as f:
            pickle.dump((clip_meta, clips_per_video), f)
        return clip_meta, clips_per_video

    def __len__(self):
        return len(self.clip_meta)
    def get_video_level_clips(self, video_idx):
        """Return indices of all clips belonging to a video."""
        return [i for i, m in enumerate(self.clip_meta) if m['video_idx'] == video_idx]
    
    # =====> 이 메서드를 추가하세요 <=====
    def get_meta_list(self):
        """Returns the list of all clip metadata dictionaries."""
        return self.clip_meta

    def get_memory(self):
        """Return list of memory clip metadata entries."""
        return self.memory_list
    def _load_clip_from_meta(self, meta):
        """Load and return clip tensor, label, video_idx from metadata."""
        clips = []
        # Pre-generate mask and augmentation settings if training
        if self.is_train:
            mask = get_mask(self.mask_param)
            mask = torch.from_numpy(mask)
            # random scale/crop params
            sel_size = random.randint(256, 320)
            crop_size = random.randint(self.frame_size, sel_size)
            center = random.randint(crop_size // 2, sel_size - crop_size // 2)
        for fname in meta['frame_files']:
            bgr = cv2.imread(os.path.join(meta['path'], fname))
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if self.is_train:
                img = self.random_scale(img, sel_size)
                img = self.random_crop(img, center, crop_size)
            img = img.resize((self.frame_size, self.frame_size))
            if self.transform:
                img = self.transform(img)
                if self.is_train:
                    mask_rgb = mask.unsqueeze(0).expand_as(img)
                    img = img * mask_rgb
            clips.append(img)
        clip_tensor = torch.stack(clips, dim=1)  # shape: [C, T, H, W]
        if self.is_train:
            clip_tensor = self.horizon_flip(clip_tensor)
        label = torch.tensor(meta['label'], dtype=torch.long)
        return clip_tensor, label, meta['video_idx']

    def __getitem__(self, idx):
        meta = self.clip_meta[idx]
        clip_tensor, label, vid = self._load_clip_from_meta(meta)
        sample = {
            'clip': clip_tensor,
            'label': label,
            'video_idx': vid,
            'is_memory': meta.get('is_memory', False),
            'meta': meta,
            'domain': meta.get('domain', self.domain)
        }
        # Dual sampling: add a random memory clip
        if self.memory_dual and self.memory_list:
            mem_meta = random.choice(self.memory_list)
            mem_clip, mem_label, mem_vid = self._load_clip_from_meta(mem_meta)
            if 'domain' not in mem_meta:
                assert False, "domain not in mem_meta"
            sample.update({
                'memory_clip': mem_clip,
                'memory_label': mem_label,
                'memory_video_idx': mem_vid,
                'memory_is_memory': True,
                'memory_meta': mem_meta,
                'memory_domain': mem_meta.get('domain')
            })
        return sample

    def get_video_level_clips(self, video_idx):
        """Return indices of all clips belonging to a video."""
        return [i for i, m in enumerate(self.clip_meta) if m['video_idx'] == video_idx]

    def get_memory(self):
        """Return list of memory clip metadata entries."""
        return self.memory_list

    def get_clip_meta(self, idx):
        """Return metadata dict for a given clip index."""
        return self.clip_meta[idx]

    def print_clip_meta(self, idx):
        """Pretty-print metadata for a given clip index."""
        from pprint import pprint
        pprint(self.clip_meta[idx])
