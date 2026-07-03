

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import os
import bisect
import numpy as np
from datasets.transforms import FTVideoRandomHorizontalFlip, RandomScale, RandomCrop, get_mask
from torch.autograd import Variable
from models.init_model import init_model_architecture__
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
import torch.nn as nn
import utils
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')       

class ContinualModel(nn.Module):
    def __init__(self, args, architecture='torchvision_r3d_18', pretrained = True, separate_classifier=True,separate_feature_extractor=False):
        super(ContinualModel, self).__init__()
        """
        ### architecture_list
            - torchvision_r3d_18
            - torchvision_r2plus1d_18
            - torch_hub_slow_r50
            - torch_hub_slowfast_r50
            - torch_hub_i3d_r50
            - torch_hub_slowfast_r101
            - vit_torch_ViViT
            - FTCN_TT
            - FTCN_only
        """
        self.args = args
        if separate_classifier:
            backbone, classifier = init_model_architecture__(architecture, pretrained, separate_classifier, separate_feature_extractor)
            self.classifier = classifier
        else:
            backbone = init_model_architecture__(architecture, pretrained, separate_classifier, separate_feature_extractor)
            self.classifier = nn.Identity()
        self.backbone = backbone
        self.separate_classifier = separate_classifier
        self.compute_means = False
        self.memory_dual = False
        if architecture == 'torchvision_r3d_18':
            self.feature_dim = 512
        elif architecture == 'torchvision_r2plus1d_18':
            self.feature_dim = 512
        elif architecture == 'torch_hub_slow_r50':
            self.feature_dim = 2048
        elif architecture == 'torch_hub_slowfast_r50':
            self.feature_dim = 2048
        elif architecture == 'torch_hub_i3d_r50':
            self.feature_dim = 1024
        elif architecture == 'torch_hub_slowfast_r101':
            self.feature_dim = 2048
        elif architecture == 'vit_torch_ViViT':
            self.feature_dim = 768
        elif architecture == 'FTCN_TT':
            self.feature_dim = 512
        elif architecture == 'FTCN_only':
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
    def feature_encoder(self,x):
        return self.backbone(x)
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    def classify(self, x):
        x = self.backbone(x)
        real_features = []
        fake_features = []
        if self.compute_means :
            with torch.no_grad():
                memory_dataset = self.get_memory_dataset(Train=False)
                memory_loader = DataLoader(memory_dataset, batch_size=args.memory_batch_size, 
                                                shuffle=False, num_workers=args.memory_num_workers)
                for inputs, labels, _, _ in memory_loader:
                    inputs = inputs.cuda(non_blocking=True)
                    features = self.backbone(inputs)
                    real_features.append(features[labels == 0])
                    fake_features.append(features[labels == 1])
                real_features = torch.cat(real_features)
                fake_features = torch.cat(fake_features)
            self.real_means = torch.mean(real_features, dim=0)
            self.fake_means = torch.mean(fake_features, dim=0)
            self.compute_means = False
        
        real_dist = torch.norm(x - self.real_means, dim=1)
        fake_dist = torch.norm(x - self.fake_means, dim=1)
        preds = real_dist < fake_dist
        return preds  
    def before_task(self, task_id ):
        pass
    def after_task(self, task_id, current_dataset, device):
        pass
    def save_memory(self,task_id, save_root):
        pass
    def del_memory(self,remain_memory_size_per_domain_class, task_num):
        pass
    def update_memory(self, cur_dataset, device):
        # Update memroay data 
        pass
    def load_memory(self, task_id, save_root):
        pass
        
    def after_train(self,task_id,current_dataset,device):
        pass
        
    @staticmethod
    def train_one_epoch(model: nn.Module,
                        data_loader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        device: torch.device,
                        epoch: int,
                        threshold: float=0.5):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Epoch: [{epoch}]"
        iters = 0

        for batch in metric_logger.log_every(data_loader, print_freq=200, header=header):
            # 1) Load inputs
            samples = batch['clip'].to(device, non_blocking=True)  # [B, C, T, H, W]
            targets = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)  # [B,1]
            video_indices = batch['video_idx']  # CPU tensor [B]

            # 2) Forward
            outputs = model(samples)  # [B,1]
            ce_criterion = nn.BCEWithLogitsLoss()
            ce_loss = ce_criterion(outputs, targets)

            # 3) Accuracy
            preds = (torch.sigmoid(outputs) > threshold).long()
            acc = (preds == targets.long()).float().mean()

            loss = ce_loss

            # 5) Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            # 6) Logging
            metric_logger.update(loss=loss.item(),
                                 ce_loss=ce_loss.item(),
                                 acc=acc.item(),
                                 lr=optimizer.param_groups[0]['lr'])
            iters += 1
            metric_logger.update(iters=iters)

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @staticmethod
    @torch.no_grad()
    def evaluate(data_loader: DataLoader,
                    model: nn.Module,
                    device: torch.device,
                    threshold: float=0.5):
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        video_to_logits = defaultdict(list)
        video_to_labels = {}
        
        # For clip-level metrics
        all_clip_scores = []
        all_clip_labels = []
        
        sigmoid = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss()
        final_loss = 0
        print_freq = 200

        for batch in metric_logger.log_every(data_loader, print_freq, header):
            samples = batch['clip'].to(device, non_blocking=True)
            targets = batch['label'].to(device, non_blocking=True).float()
            video_indices = batch['video_idx']

            outputs = model(samples)
            loss = criterion(outputs.view(-1,1), targets.view(-1,1))
            scores = sigmoid(outputs.view(-1,1))
            loss_value = loss.item()

            # Collect clip-level predictions
            all_clip_scores.extend(scores.cpu().numpy().flatten())
            all_clip_labels.extend(targets.cpu().numpy().flatten())

            for i, vid in enumerate(video_indices):
                video_id = int(vid.item())
                video_to_logits[video_id].append(scores[i])
                video_to_labels[video_id] = targets.view(-1,1)[i]

            metric_logger.update(loss=loss_value)
            final_loss += loss_value

        # Clip-level metrics
        all_clip_scores = np.array(all_clip_scores)
        all_clip_labels = np.array(all_clip_labels)
        
        clip_preds = (all_clip_scores > threshold).astype(int)
        acc_clip = metrics.accuracy_score(all_clip_labels, clip_preds)
        auc_clip = metrics.roc_auc_score(all_clip_labels, all_clip_scores)
        
        # Aggregate video-level predictions and labels
        video_scores = []
        video_labels = []
        for video_id in sorted(video_to_logits.keys()):
            avg_score = torch.stack(video_to_logits[video_id]).mean().item()
            video_scores.append(avg_score)
            video_labels.append(video_to_labels[video_id].item())
        
        video_scores = np.array(video_scores)
        video_labels = np.array(video_labels)
        
        # Compute video-level AUC
        auc_video = metrics.roc_auc_score(video_labels, video_scores)
        
        # Compute video-level Accuracy
        video_preds = (video_scores > threshold).astype(int)
        acc_video = metrics.accuracy_score(video_labels, video_preds)
        
        # Compute video-level EER
        fpr, tpr, thresholds = metrics.roc_curve(video_labels, video_scores, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        print(f"=====>Clip-Level-AUC: {auc_clip:.4f}")
        print(f"=====>Clip-Level-Accuracy: {acc_clip:.4f}")
        print(f"=====>Video-Level-AUC: {auc_video:.4f}")
        print(f"=====>Video-Level-Accuracy: {acc_video:.4f}")
        print(f"=====>Video-Level-EER: {eer:.4f} (threshold: {eer_threshold:.4f})")

        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats['auc_clip'] = auc_clip
        stats['accuracy_clip'] = acc_clip
        stats['auc_video'] = auc_video
        stats['accuracy_video'] = acc_video
        stats['eer'] = eer
        
        return auc_video, stats  # @staticmethod  # @torch.no_grad()
    # def get_features_in_loader(data_loader: DataLoader,
    #              model: nn.Module,
    #              device: torch.device,
    #              length: Optional[int]=None):
    #     model.eval()
    #     metric_logger = utils.MetricLogger(delimiter="  ")
    #     header = 'Test:'
    #     video_to_logits = defaultdict(list)
    #     video_to_labels = {}
    #     sigmoid = nn.Sigmoid()
    #     criterion = nn.BCEWithLogitsLoss()
    #     final_loss = 0
    #     print_freq = 200
        
    #     for batch in metric_logger.log_every(data_loader, print_freq, header):
    #         samples = batch['clip'].to(device, non_blocking=True)
    #         targets = batch['label'].to(device, non_blocking=True).float()
    #         video_indices = batch['video_idx']

    #         outputs = model(samples)
    #         for i, vid in enumerate(video_indices):
    #             video_id = int(vid.item())
    #             video_to_logits[video_id].append(outputs[i])
    #             video_to_labels[video_id] = targets.view(-1,1)[i]

    #     return video_to_logits, video_to_labels
    @staticmethod
    @torch.no_grad()
    def get_features_in_loader(data_loader: DataLoader,
                 model: nn.Module,
                 device: torch.device,
                 length: Optional[int]=None):
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        video_to_features = []
        video_to_labels = []
        sigmoid = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss()
        final_loss = 0
        print_freq = 200
        
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            samples = batch['clip'].to(device, non_blocking=True)
            targets = batch['label'].to(device, non_blocking=True).float()
            video_indices = batch['video_idx']

            outputs = model.backbone(samples)
            for i, vid in enumerate(video_indices):
                video_id = int(vid.item())
                video_to_features.append(outputs[i])
                video_to_labels.append(targets[i])
        video_to_features = torch.stack(video_to_features)
        video_to_labels = torch.stack(video_to_labels)
        return video_to_features, video_to_labels
            

            