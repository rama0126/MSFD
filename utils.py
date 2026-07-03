import pandas as pd

import math
import os
from operator import index
import sys
from typing import Iterable, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy
# from losses import DistillationLoss # (?)
import utils

from sklearn import metrics
from collections import defaultdict
import torch.nn.functional as F
def get_files_from_split(split):
    """ "
    Get filenames for real and fake samples

    Parameters
    ----------
    split : pandas.DataFrame
        DataFrame containing filenames
    """
    files_1 = split[0].astype(str).str.cat(split[1].astype(str), sep="_")
    files_2 = split[1].astype(str).str.cat(split[0].astype(str), sep="_")
    files_real = pd.concat([split[0].astype(str), split[1].astype(str)]).to_list()
    files_fake = pd.concat([files_1, files_2]).to_list()
    return files_real, files_fake


import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
# import mmcv

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

# Official FTCN 에서 가중치를 사용하기 위해 인자 추가 #
# 참고: https://tutorials.pytorch.kr/beginner/saving_loading_models.html
def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs, _use_new_zipfile_serialization=False)

############################################################
### 파이토치(pytorch) Distributed Data Parallel(DDP) 사용 ###
############################################################
# 참고 link
# https://velog.io/@doooli/Pytorch-Multi-GPU
# https://swprog.tistory.com/entry/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98Pytorch-Distributed-Data-Parallel-DDP%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0
# https://developer0hye.tistory.com/167

def init_distributed_mode(args):
    # 1번: 초기화 단계
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ: 
        args.rank = int(os.environ["RANK"])                 # os.environ["RANK"]: 각 프로세스의 우선순위
        args.world_size = int(os.environ['WORLD_SIZE'])     # os.environ["WORLD_SIZE"]: 전체 프로세스 개수 = 전체 GPU의 개수
        args.gpu = int(os.environ['LOCAL_RANK'])            # os.environ["LOCAL_RANK"]: 특정 프로세스가 사용하는 GPU의 번호, 프로그램이 실행되면 변수가 자동으로 설정
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])         # (?)
        args.gpu = args.rank % torch.cuda.device_count()    # (?)
    else:
        print('Not using distributed mode')                 # 분산 병렬처리 진행하지 않는 경우
        args.distributed = False
        return

    args.distributed = True

    # 2번: 그룹을 초기화 하고, 프로세스가 사용할 GPU 번호를 실제로 설정하는 과정
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()     # 여기까지 프로세스가 수행된 후, 더 진행하기 전에 다른 프로세스들이 여기까지 수행하기를 기다리도록 함.
    
    # 3번: 메인프로세스만 출력하도록 하는 방법을 진행
    # -> 여러 프로세스들을 수행하기 때문에 1개의 print문 일지라도 각 프로세스가 진행되면 그 개수만큼 출력
    # -> 이를 해결하기 위해 메인 프로세스만 출력되도록 한다. 
    # 이부분 만져서 GPU 할당 관련 문제 해결
    setup_for_distributed(args.rank == 0)
    
    
""" Random Erasing (Cutout)
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng
Hacked together by / Copyright 2020 Ross Wightman

참고: https://timm.fast.ai/RandomErase
참고: https://github.com/zhoudaquan/Refiner_ViT/blob/master/timm/data/random_erasing.py

"""
import random
import math
import torch

threshold = 0.5
def compute_auc(scores, targets):
    """Compute ROC AUC from score/target arrays."""
    import numpy as np
    scores_np = np.asarray(scores).flatten()
    targets_np = np.asarray(targets).flatten()
    fpr, tpr, _ = metrics.roc_curve(targets_np, scores_np)
    return metrics.auc(fpr, tpr)

def compute_video_level_auc(video_to_logits, video_to_labels):
    """ "
    Compute video-level area under ROC curve. Averages the logits across the video for non-overlapping clips.

    Parameters
    ----------
    video_to_logits : dict
        Maps video ids to list of logit values
    video_to_labels : dict
        Maps video ids to label
    """
    import numpy as np

    output_batch = torch.stack(
        [torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False)
         for video_id in video_to_logits.keys()]
    )

    output_labels = torch.stack([video_to_labels[video_id] for video_id in video_to_logits.keys()])


    # print(output_batch, output_labels, f"All {len(output_labels.cpu().numpy())} videos done!")

    fpr, tpr, _ = metrics.roc_curve(output_labels.cpu().numpy(), output_batch.cpu().numpy())
    
    # roc_auc_score 랑 동일하게 작동함
    return metrics.auc(fpr, tpr)


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


### class for Cutout ###
class RandomErasing_Clip:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device
        

    def _erase_1(self, img, chan, img_h, img_w, dtype):
        arg_dict = {}

        pro = random.random() # !
        arg_dict['pro'] = pro
        if pro > self.probability:
            return arg_dict
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count) # !

        arg_dict['count'] = count

        for i in range(count):
            arg_dict[f"{i}"] = []

        for i in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio))) # !
                w = int(round(math.sqrt(target_area / aspect_ratio))) # !
                
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h) # !
                    left = random.randint(0, img_w - w) # !
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    
                    arg_dict[f'{i}'].append(h)
                    arg_dict[f'{i}'].append(w)
                    arg_dict[f'{i}'].append(top)
                    arg_dict[f'{i}'].append(left)
                    break
        return arg_dict

    def _erase_a(self, img, chan, img_h, img_w, dtype, arg_dict):
        if arg_dict['pro'] > self.probability:
            return
        count = arg_dict['count']

        for i in range(count):
            arg_list = arg_dict[f"{i}"]
            # 10번의 attempt 에도 안된경우 예외처리
            if len(arg_list) == 0:
                continue
            
            h = arg_list[0]
            w = arg_list[1]
            top = arg_list[2]
            left = arg_list[3]

            img[:, top:top + h, left:left + w] = _get_pixels(self.per_pixel,
                                                             self.rand_color,
                                                             (chan, h, w),
                                                             dtype=dtype,
                                                             device=self.device)
    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, num_frames, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            input = torch.permute(input, (0, 2, 1, 3, 4))
            for i in range(batch_start, batch_size): 
                for j in range(num_frames):
                    if j == 0:
                        arg_dict = self._erase_1(input[i][j], chan, img_h, img_w, input.dtype)
                    else:
                        self._erase_a(input[i][j], chan, img_h, img_w, input.dtype, arg_dict)
            input = torch.permute(input, (0, 2, 1, 3, 4))
        return input
    
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def get_GPU_USAGE():
    # Get GPU Memory Occupy
    
    def log_gpu_usage():
        if not torch.cuda.is_available():
            return

        from pynvml.smi import nvidia_smi

        
        nvsmi = nvidia_smi.getInstance()
        res_memory_list = nvsmi.DeviceQuery("memory.free, memory.total, memory.used")["gpu"]
        res_util_list = nvsmi.DeviceQuery("utilization.gpu, memory.free, memory.total, memory.used")["gpu"]


        total = np.average([each["fb_memory_usage"]["total"] for each in res_memory_list])
        used = np.average([each["fb_memory_usage"]["used"] for each in res_memory_list])
        free = np.average([each["fb_memory_usage"]["free"] for each in res_memory_list])
        percentage = 100*used/total
            
        utilization = np.average([each["utilization"]["gpu_util"] for each in res_util_list])
        
        print(
            f'GPU Usage. Util: {utilization} Used: {used} Total: {total} ({percentage}% used). Free: {free}'
        )
    
    def log_loop():
        import time
        log_gpu_usage()
        time.sleep(15)
    log_loop()
