# models/msfd.py

import os
import json
import random
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
from sklearn.metrics import roc_curve

from .init_continual_model import ContinualModel
import utils

# gradient scaling

def batch_to_meta_list(metas: dict):
    """
    Convert a batch-shaped meta-dict into a list of per-sample meta-dicts.
    """
    new_metas = []
    batch_size = len(metas['path'])
    for i in range(batch_size):
        meta = {}
        for key in metas.keys():
            data = metas[key]
            if torch.is_tensor(data):
                # 1-1) 1차원 Tensor: scalar
                if data.dim() == 1:
                    sample = data[i].item()
                # 1-2) 그 외 차원: Tensor 그대로
                else:
                    sample = data[i].tolist()
            elif isinstance(data, list) or isinstance(data, tuple):
                sample = data[i]
                if torch.is_tensor(sample):
                    if sample.dim() == 1:
                        sample = sample.item()
                    else:
                        assert False, f"{sample}, new case"
                else:
                    if isinstance(sample, list) or isinstance(sample, tuple):
                        sample = []
                        for di in range(len(data)):
                            sample.append(data[di][i])
            else:
                sample = data
            meta[key] = sample
        new_metas.append(meta)
    return new_metas
            
class TempMemoryDataset(Dataset):
    """A temporary dataset wrapper for memory items during mean computation or herding."""
    def __init__(self, memory_meta_list, frames_per_clip=32, frame_size=224, transform=None, is_train=False):
        self.memory_meta = memory_meta_list
        self.frames_per_clip = frames_per_clip
        self.frame_size = frame_size
        self.transform = transform if transform else Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.is_train = is_train

    def __len__(self):
        return len(self.memory_meta)

    def _load_clip_from_meta(self, meta):
        """Simplified clip loading for feature extraction."""
        clips = []
        for fname in meta['frame_files']:
            try:
                img = Image.open(os.path.join(meta['path'], fname)).convert('RGB')
                img = img.resize((self.frame_size, self.frame_size))
                if self.transform:
                    img = self.transform(img)
                clips.append(img)
            except Exception as e:
                print(f"Error loading frame {os.path.join(meta['path'], fname)}: {e}")
                return None, None, None

        if not clips:
            return None, None, None

        if len(clips) < self.frames_per_clip:
            print(f"Warning: Loaded {len(clips)} frames, expected {self.frames_per_clip} for {meta['path']}")
            last_frame = clips[-1]
            clips.extend([last_frame] * (self.frames_per_clip - len(clips)))

        clip_tensor = torch.stack(clips, dim=1)  # shape: [C, T, H, W]
        label = torch.tensor(meta['label'], dtype=torch.long)
        video_idx = meta['video_idx']
        return clip_tensor, label, video_idx

    def __getitem__(self, idx):
        meta = self.memory_meta[idx]
        clip_tensor, label, vid_idx = self._load_clip_from_meta(meta)

        if clip_tensor is None:
            dummy_clip = torch.zeros((3, self.frames_per_clip, self.frame_size, self.frame_size))
            dummy_label = torch.tensor(-1, dtype=torch.long)
            dummy_vid_idx = -1
            return {'clip': dummy_clip, 'label': dummy_label, 'video_idx': dummy_vid_idx, 'meta': meta, 'valid': False}

        return {'clip': clip_tensor, 'label': label, 'video_idx': vid_idx, 'meta': meta, 'valid': True}

def collate_skip_invalid(batch):
    batch = [item for item in batch if item['valid']]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# =============================================================================
# Main Model: FREQK
# =============================================================================
class MSFD(ContinualModel):
    """
    Intermediate Feature Distillation을 수행하는 모델.
    Intermediate distillation 3D Frequnecy MSE loss 진행
    """
    def __init__(self, args, architecture='torchvision_r3d_18', pretrained=True, separate_classifier=True):
        super().__init__(args, architecture, pretrained, separate_classifier)
        
        self.args = args
        self.memory_size = args.memory_size
        
        # --- [LOWA 핵심 파라미터] ---
        # lambda_kd: Knowledge Distillation (logit-level) 가중치
        # lambda_fd: Feature Distillation (low-level) 가중치
        self.lambda_fd = getattr(args, 'lowa_lambda_fd', 1.0)
        # distill_layer_name: 증류에 사용할 중간 레이어 이름
        if architecture in ['torchvision_r3d_18', 'torchvision_r2plus1d_18', 'ftcn_only']:
            distill_layer_names = getattr(args, 'lowa_feature_layer', 'layer1,layer2,layer3,layer4')
        elif architecture in ['torch_hub_i3d_r50']:
            distill_layer_names = getattr(args, 'lowa_feature_layer', 'layer1,layer2,layer3,layer4,layer5')

        self.distill_layer_names = distill_layer_names.split(',')  #
        self.lambda_pd = getattr(args, 'lowa_lambda_pd', 1.0)
        print(f"Initializing LOWA model. Distillation layer: '{self.distill_layer_names}'")
        print(f"Distillation weights: FD={self.lambda_fd}")
        self.gumbel_tau = getattr(args, 'lowa_gumbel_tau', 1.0)
        print(f"Gumbel-Softmax temperature: {self.gumbel_tau}")
        
        self.spatial_lambda = getattr(args, 'lowa_spatial_lambda', 1.0)
        self.temporal_lambda = getattr(args, 'lowa_temporal_lambda', 1.0)
        self.spatiotemporal_lambda = getattr(args, 'lowa_spatiotemporal_lambda', 0.5)
        self.consistency_lambda = getattr(args, 'lowa_orthogonal_lambda', 0.5)
        print(f"Frequency Distillation Weights: Spatial={self.spatial_lambda}, Temporal={self.temporal_lambda}, Spatiotemporal={self.spatiotemporal_lambda}, Orthogonal={self.consistency_lambda}")
        
        self.memory_list = []
        self.learnable_filters = getattr(self, 'learnable_filters', nn.ParameterDict())
        self.learnable_masks = getattr(self, 'learnable_masks', nn.ModuleDict())
        self.old_model = None
        self.current_task_id = 0
        self.class_means = None
        self.compute_means = True
        self.memory_dual = True
        self._initialize_learnable_components()
    def _initialize_learnable_components(self):
        """
        더미 입력을 사용하여 특징맵 크기에 따라 동적으로 `bins`를 설정하고,
        Gumbel-Softmax 기반 Radial 마스크들을 미리 생성합니다.
        """
        print("\n--- Initializing DYNAMIC Gumbel Radial Frequency Masks ---")
        dummy_input = torch.randn(1, 3, 32, 224, 224)
        self.eval()

        for layer_name in self.distill_layer_names:
            print(f"  Processing layer: '{layer_name}'")
            with torch.no_grad():
                feat = self.feature_extractor(dummy_input, layer_name=layer_name)
            
            shape = feat.shape
            T = shape[2] if feat.dim() == 5 else 1
            H, W = shape[-2], shape[-1]
            print(f"    - Inferred feature shape: C={shape[1]}, T={T}, H={H}, W={W}")


            t_bins = 4
            s_bins = 16
            st_bins = 32

            if T > 1:
                self.learnable_masks[f'temporal_mask_{layer_name}'] = GumbelRadialFreqMask(
                    (T,), bins=t_bins, gumbel_tau=self.gumbel_tau)
                print(f"    - Created 1D Gumbel Radial Mask (T={T}) with dynamic bins={t_bins}")
                self.learnable_masks[f"spatiotemporal_mask_{layer_name}"] = GumbelRadialFreqMask(
                    (T, H, W), bins=st_bins, gumbel_tau=self.gumbel_tau)

                
            self.learnable_masks[f'spatial_mask_{layer_name}'] = GumbelRadialFreqMask(
                (H, W), bins=s_bins, gumbel_tau=self.gumbel_tau)
            print(f"    - Created 2D Gumbel Radial Mask (H,W={H},{W}) with dynamic bins={s_bins}")

            # mask는 처음에는 다 통과로 초기화
            
            self._get_or_create_filter(f'spatial_filter_{layer_name}', shape=(H,W), in_channels=shape[1])
            if T > 1:
                self._get_or_create_filter(f'temporal_filter_{layer_name}',shape=(T,), in_channels=shape[1])
                self._get_or_create_filter(f'spatiotemporal_filter_{layer_name}', shape=(T,H,W), in_channels=shape[1])
                self.learnable_masks[f'spatial_proj_{layer_name}'] = nn.Linear((H*W), s_bins)
                self.learnable_masks[f'temporal_proj_{layer_name}'] = nn.Linear((T), t_bins)
                self.learnable_masks[f'spatiotemporal_proj_{layer_name}_spatial'] = nn.Linear((T*H*W), s_bins)
                self.learnable_masks[f'spatiotemporal_proj_{layer_name}_temporal'] = nn.Linear((T*H*W), t_bins)
            
        self.train()
        print("--- Initialization of dynamic Gumbel radial masks complete ---\n")
    def train(self, mode=True):
        super().train(mode)
        for mask in self.learnable_masks.values():
            mask.train(mode)
        for filt in self.learnable_filters.values():
            filt.requires_grad = True
        return self
    def eval(self):
        super().eval()
        for mask in self.learnable_masks.values():
            mask.eval()
        for filt in self.learnable_filters.values():
            filt.requires_grad = False
        return self
    def _get_or_create_filter(self, filter_name, shape = (224,224), in_channels=512):
        if filter_name not in self.learnable_filters:
            filt = nn.Parameter(torch.zeros((in_channels, *shape)), requires_grad=True)
            self.learnable_filters[filter_name] = filt

    def parameters(self):
        params = list(self.backbone.parameters()) + list(self.classifier.parameters())
        params.extend(list(self.learnable_filters.parameters()))
        params.extend(list(self.learnable_masks.parameters()))
        return params
    def feature_extractor(self, x: torch.Tensor, layer_name: str = 'final'):
        """
        다양한 비디오 백본 아키텍처에서 특정 레이어의 특징을 추출합니다.
        'final'은 최종 특징(분류기 입력 전)을 의미합니다.
        """
        # 1. 최종 특징 추출 (기본 동작)
        if layer_name == 'final':
            # SlowFast는 입력이 리스트 형태일 수 있으므로 그대로 전달
            if isinstance(x, list):
                return self.backbone(x)
            # SlowFast가 아닌 모델은 텐서 형태의 입력을 받음
            else:
                # PytorchVideo 모델들은 입력으로 텐서를 받음
                if 'facebookresearch/pytorchvideo' in getattr(self.backbone, '_get_name', lambda: '')():
                    return self.backbone(x)
                # ViViT는 입력으로 텐서를 받음
                elif 'ViT' in self.backbone.__class__.__name__:
                    return self.backbone(x)
                # Torchvision 모델들도 입력으로 텐서를 받음
                else:
                    return self.backbone(x)


        # --- 중간 레이어 특징 추출 ---
        
        # 2. Torchvision 비디오 모델 (r3d_18, r2plus1d_18)
        # torchvision.models.video.resnet.VideoResNet
        if 'VideoResNet' in self.backbone.__class__.__name__:
            b = self.backbone
            x = b.stem(x)
            if layer_name == 'all':
                layers_out= {}
                layers_out['stem'] = x
                x = b.layer1(x)
                layers_out['layer1'] = x
                x = b.layer2(x)
                layers_out['layer2'] = x
                x = b.layer3(x)
                layers_out['layer3'] = x
                x = b.layer4(x)
                layers_out['layer4'] = x
                x = b.avgpool(x)
                layers_out['pool'] = x.flatten(1)
                return layers_out
            else:   
                if layer_name == 'stem': return x
                x = b.layer1(x)
                if layer_name == 'layer1': return x
                x = b.layer2(x)
                if layer_name == 'layer2': return x
                x = b.layer3(x)
                if layer_name == 'layer3': return x
                x = b.layer4(x)
                if layer_name == 'layer4': return x
                
                # 'pool' layer for getting features before fc
                if layer_name == 'pool':
                    x = b.avgpool(x)
                    return x.flatten(1)
            
            raise ValueError(f"Unsupported layer_name '{layer_name}' for VideoResNet.")

        # 3. PyTorchVideo 모델 (Slow, SlowFast, I3D)
        # pytorchvideo.models.resnet.ResNet
        elif 'pytorchvideo' in str(type(self.backbone)):
            b = self.backbone
            
            # SlowFast 모델 처리
            if isinstance(x, list) and len(x) == 2:
                x_slow, x_fast = x[0], x[1]
                
                # stem
                x_slow = b.blocks[0](x_slow)
                x_fast = b.blocks[0](x_fast)
                
                if layer_name == 'stem_slow': return x_slow
                if layer_name == 'stem_fast': return x_fast
                
                # blocks (0은 stem, 1~4는 res_stages, 5는 head)
                x_slow, x_fast = b.blocks[1]((x_slow, x_fast))
                if layer_name == 'block1' or layer_name == 'layer1': return [x_slow, x_fast]
                
                x_slow, x_fast = b.blocks[2]((x_slow, x_fast))
                if layer_name == 'block2' or layer_name == 'layer2': return [x_slow, x_fast]
                
                x_slow, x_fast = b.blocks[3]((x_slow, x_fast))
                if layer_name == 'block3' or layer_name == 'layer3': return [x_slow, x_fast]
                
                x_slow, x_fast = b.blocks[4]((x_slow, x_fast))
                if layer_name == 'block4' or layer_name == 'layer4': return [x_slow, x_fast]
                
                raise ValueError(f"Unsupported layer_name '{layer_name}' for SlowFast.")

            # Slow, I3D 모델 처리
            else:

                x = b.blocks[0](x) # stem

                if layer_name=='all':
                    layers_out = {}
                    layers_out['stem'] = x
                    x = b.blocks[1](x)
                    layers_out['layer1'] = x
                    x = b.blocks[2](x)
                    layers_out['layer2'] = x
                    x = b.blocks[3](x)
                    layers_out['layer3'] = x
                    x = b.blocks[4](x)
                    layers_out['layer4'] = x
                    x = b.blocks[5](x)
                    layers_out['layer5'] = x
                    x = b.blocks[6](x)
                    layers_out['pool'] = x

                    return layers_out
                else:
                    if layer_name == 'stem': return x
                    x = b.blocks[1](x) # res_stage 1
                    if layer_name == 'block1' or layer_name == 'layer1': return x
                    x = b.blocks[2](x) # res_stage 2
                    if layer_name == 'block2' or layer_name == 'layer2': return x
                    x = b.blocks[3](x) # res_stage 3
                    if layer_name == 'block3' or layer_name == 'layer3': return x
                    x = b.blocks[4](x) # res_stage 4
                    if layer_name == 'block4' or layer_name == 'layer4': return x
                    x = b.blocks[5](x)
                    if layer_name == 'block5' or layer_name == 'layer5': return x
                    
                    raise ValueError(f"Unsupported layer_name '{layer_name}' for PyTorchVideo ResNet.")
                    
        # 4. ViViT (vit-pytorch)
        elif 'ViT' in self.backbone.__class__.__name__:
            b = self.backbone
            x = b.to_patch_embedding(x)
            b, c, f, h, w = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(b, f, h * w, c)

            # Spatial Transformer
            x = b.space_transformer(x)
            if layer_name == 'spatial_transformer': return x
            
            # Reshape for Temporal Transformer
            x = x.reshape(b, f, -1).permute(0, 2, 1).reshape(b * (h*w), f, c)
            
            # Temporal Transformer
            x = b.temporal_transformer(x)
            if layer_name == 'temporal_transformer': return x

            # Return pooled features if needed
            if layer_name == 'pool':
                x = x.mean(dim=1) # or x[:, 0] for CLS token based models
                return x.reshape(b, -1, c).mean(dim=1)

            raise ValueError(f"Unsupported layer_name '{layer_name}' for ViViT.")
            
        # 5. FTCN (가정: `model.resnet`에 접근 가능)
        elif hasattr(self.backbone, 'resnet'):
            # FTCN의 백본은 ResNet 기반이므로 ResNet 로직과 유사하게 처리
            b = self.backbone.resnet
            x = b.stem(x)
            if layer_name == 'stem': return x
            x = b.layer1(x)
            if layer_name == 'layer1': return x
            x = b.layer2(x)
            if layer_name == 'layer2': return x
            x = b.layer3(x)
            if layer_name == 'layer3': return x
            x = b.layer4(x)
            if layer_name == 'layer4': return x
            raise ValueError(f"Unsupported layer_name '{layer_name}' for FTCN.")

        # 6. 지원하지 않는 아키텍처
        raise NotImplementedError(f"Intermediate feature_extractor for {type(self.backbone).__name__} is not implemented.")

    def forward(self, x, return_features=False):
        """
        표준 forward pass. 최종 특징과 로짓을 반환할 수 있습니다.
        """
        final_features = self.feature_extractor(x, layer_name='final')
        logits = self.classifier(final_features)
        if return_features:
            return logits, final_features
        return logits

    def train_one_epoch(self, model, data_loader, optimizer, device, epoch, threshold=0.5):
        self.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Epoch: [{epoch}] Task: [{self.current_task_id}]"
        
        ce_criterion = nn.BCEWithLogitsLoss()
        optimizer.zero_grad(set_to_none=True)
        start_tau = 1.0
        min_tau = 0.05
        anneal_epochs = 10 # 10 에폭에 걸쳐 tau를 점진적으로 감소
        current_tau = max(min_tau, start_tau - (start_tau - min_tau) * (epoch / anneal_epochs))
        for module in self.learnable_masks.values():
            module.gumbel_tau = current_tau
        accumulating_batches = 2 if self.args.batch_size == 4 else 1
        if epoch == 0:
            print(f"Gradient accumulation steps: {accumulating_batches}")
        accumulated_steps = 0
        for batch in metric_logger.log_every(data_loader, print_freq=200, header=header):
            # Get inputs and targets
            samples = batch['clip'].to(device, non_blocking=True)
            targets = batch['label'].to(device, non_blocking=True).float().unsqueeze(1)
            domains = batch['domain'].to(device, non_blocking=True)
            
            # Get memory samples if available
            memory_samples = batch.get('memory_clip', None)
            if memory_samples is not None:
                memory_samples = memory_samples.to(device, non_blocking=True)
                memory_targets = batch['memory_label'].to(device, non_blocking=True).float().unsqueeze(1)
                memory_domains = batch['memory_domain'].to(device, non_blocking=True)
                assert batch['memory_clip'] is not None, "memory_clip is None"

            # Forward pass
            total_samples = samples
            if memory_samples is not None:
                samples = torch.cat([samples, memory_samples], dim=0).to(device, non_blocking=True)
                targets = torch.cat([targets, memory_targets], dim=0).to(device, non_blocking=True)
            cur_low_level_features = self.feature_extractor(samples, layer_name='all')
            outputs = self.classifier(cur_low_level_features['pool'])
            
            # Classification loss
            ce_loss = ce_criterion(outputs, targets)
            
            # Knowledge distillation loss
            fd_distill_loss = torch.tensor(0.0, device=device)
            pd_distill_loss = torch.tensor(0.0, device=device)
            orthol_loss = torch.tensor(0.0, device=device)
            # Prediction- & feature-distillation exist only once a previous-task
            # model has been snapshotted (task_id > 0). At task 0 both old_model
            # and the replay buffer are None, so every distillation term stays 0
            # and the objective reduces to the classification loss.
            if self.old_model is not None and memory_samples is not None:
                with torch.no_grad():
                    old_probs = torch.sigmoid(self.old_model(memory_samples))
                cur_prediction = outputs[-len(memory_samples):]
                pd_distill_loss = F.binary_cross_entropy_with_logits(cur_prediction, old_probs)
                with torch.no_grad():
                    old_low_level_features = self.old_model.feature_extractor(memory_samples, layer_name='all')
                for layer_name in self.distill_layer_names:
                    cur_features = cur_low_level_features[layer_name][ -len(memory_samples):]
                    old_features = old_low_level_features[layer_name]
                    l_fd_distill_loss, l_othol_loss = self.spatiotemporal_freq_kd_loss(cur_features, old_features, layer_name, device=device)
                    fd_distill_loss += l_fd_distill_loss
                    orthol_loss += l_othol_loss
                fd_distill_loss = fd_distill_loss / len(self.distill_layer_names)
                orthol_loss = orthol_loss / len(self.distill_layer_names)
                
                
            loss = ce_loss + self.lambda_fd * fd_distill_loss + self.lambda_pd * pd_distill_loss + orthol_loss * self.consistency_lambda
            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > threshold).long()
            acc = (preds == targets.long()).float().mean()
            
            # Backpropagation
            (loss / accumulating_batches).backward()
            accumulated_steps += 1
            if accumulated_steps % accumulating_batches == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0
            
            
            # Update metrics
            metric_logger.update(
                loss=loss.item(),
                ce_loss=ce_loss.item(),
                fd_dist_loss=fd_distill_loss.item(),
                pd_dist_loss=pd_distill_loss.item(),
                orthol_loss=orthol_loss.item(),
                acc=acc.item(),
                lr=optimizer.param_groups[0]['lr'],
                tau=current_tau,
            )

        if accumulated_steps > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Print and return stats
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def before_task(self, task_id):
        print(f"\n--- Preparing for Task {task_id} (LOWA) ---")
        self.current_task_id = task_id
        if task_id > 0:
            print("Loading old model and memory for distillation...")
            self.old_model = copy.deepcopy(self)
            self.old_model.eval()
            for param in self.old_model.parameters(): param.requires_grad = False
            self.load_memory(task_id, self.args.save_root)
        else:
            self.old_model = None
        self.compute_means = True

    def after_task(self, task_id, current_dataset, device):
        print(f"\n--- Finalizing Task {task_id} (LOWA) ---")
        self.update_memory(current_dataset, device)
        self.save_memory(task_id, self.args.save_root)
        self.compute_class_means(device)
        print(f"--- Task {task_id} Finalized ---")

    @torch.no_grad()
    def update_memory(self, cur_dataset, device):
        self.eval()
        exemplars_per_class = self.memory_size // 2
        print(f"Updating memory: Target {exemplars_per_class} exemplars per class.")

        candidate_features, candidate_meta = {0: [], 1: []}, {0: [], 1: []}

        # 1. 기존 메모리에서 특징 추출
        if self.memory_list:
            memory_dataset = TempMemoryDataset(self.memory_list)
            memory_loader = DataLoader(memory_dataset, batch_size=self.args.batch_size * 2,
                num_workers=self.args.num_workers,collate_fn=collate_skip_invalid)
            for batch in memory_loader:
                if batch is None: continue
                features = self.feature_extractor(batch['clip'].to(device), 'final').cpu()
                features = F.normalize(features, p=2, dim=1)
                for i, label in enumerate(batch['label'].tolist()):
                    candidate_features[label].append(features[i])
                    candidate_meta[label].append(batch_to_meta_list(batch['meta'])[i])
        
        # 2. 현재 데이터셋에서 특징 추출
        cur_loader = DataLoader(cur_dataset, batch_size=self.args.batch_size * 2,
                num_workers=self.args.num_workers,)
        for batch in cur_loader:
            features = self.feature_extractor(batch['clip'].to(device), 'final').cpu()
            features = F.normalize(features, p=2, dim=1)
            for i, label in enumerate(batch['label'].tolist()):
                candidate_features[label].append(features[i])
                candidate_meta[label].append(batch_to_meta_list(batch['meta'])[i])

        # 3. Herding으로 최종 메모리 선택
        final_memory_list = []
        for class_idx in [0, 1]:
            features, metas = candidate_features[class_idx], candidate_meta[class_idx]
            if not features: continue
            num_to_select = min(exemplars_per_class, len(features))
            
            features_tensor = torch.stack(features)
            class_mean = torch.mean(features_tensor, dim=0)
            
            selected_indices, current_sum = [], torch.zeros_like(class_mean)
            for k in range(num_to_select):
                target = (k + 1) * class_mean - current_sum
                distances = torch.norm(features_tensor - target.unsqueeze(0), dim=1)
                candidates = [i for i in range(len(features_tensor)) if i not in selected_indices]
                if not candidates: break
                best_candidate_idx = candidates[torch.argmin(distances[candidates]).item()]
                selected_indices.append(best_candidate_idx)
                current_sum += features_tensor[best_candidate_idx]
            
            for idx in selected_indices: final_memory_list.append(metas[idx])
        
        self.memory_list = final_memory_list
        random.shuffle(self.memory_list)
        print(f"Updated memory size: {len(self.memory_list)}")

    def save_memory(self, task_id, save_root):
        if utils.is_main_process():
            mem_dir = os.path.join(save_root, 'memories')
            os.makedirs(mem_dir, exist_ok=True)
            path = os.path.join(mem_dir, f"{task_id}_memory.json")
            with open(path, 'w') as f: json.dump(self.memory_list, f, indent=2)
            print(f"LOWA Memory saved to {path}")
    def load_memory(self, task_id, save_root):
        mem_dir = os.path.join(save_root, 'memories')
        path = os.path.join(mem_dir, f"{task_id - 1}_memory.json")
        if utils.is_main_process():
            if os.path.exists(path):
                with open(path, 'r') as f: 
                    self.memory_list = json.load(f)
                print(f"LOWA Memory loaded from {path}, size: {len(self.memory_list)}")
            else: 
                print(f"LOWA Memory file not found: {path}")
        if utils.is_dist_avail_and_initialized():
            torch.distributed.barrier()
            # Broadcast memory_list from rank 0 to all processes
            memory_list_obj = [self.memory_list] if utils.is_main_process() else [None]
            torch.distributed.broadcast_object_list(memory_list_obj, src=0)
            self.memory_list = memory_list_obj[0]

    @torch.no_grad()
    def compute_class_means(self, device):
        if not self.memory_list: self.class_means = None; return
        self.eval()
        memory_dataset = TempMemoryDataset(self.memory_list)
        memory_loader = DataLoader(memory_dataset, batch_size=self.args.batch_size * 2, 
                num_workers=self.args.num_workers, collate_fn=collate_skip_invalid)
        features_by_class = {0: [], 1: []}
        for batch in memory_loader:
            if batch is None: continue
            features = self.feature_extractor(batch['clip'].to(device), 'final').cpu()
            for c in [0, 1]:
                if torch.any(batch['label'] == c):
                    features_by_class[c].append(F.normalize(features[batch['label'] == c], p=2, dim=1))
        self.class_means = {}
        for c in [0, 1]:
            if features_by_class[c]:
                self.class_means[c] = F.normalize(torch.mean(torch.cat(features_by_class[c], dim=0), dim=0), p=2, dim=0).to(device)
        self.compute_means = False

    @torch.no_grad()
    def classify_ncm(self, x):
        if self.class_means is None: return torch.rand(x.shape[0], device=x.device)
        features = F.normalize(self.feature_extractor(x, 'final'), p=2, dim=1)
        scores = []
        for feat in features:
            dist0 = torch.norm(feat - self.class_means.get(0, torch.zeros_like(feat))) if self.class_means.get(0) is not None else float('inf')
            dist1 = torch.norm(feat - self.class_means.get(1, torch.zeros_like(feat))) if self.class_means.get(1) is not None else float('inf')
            scores.append(dist0 - dist1)
        return torch.tensor(scores, device=x.device)

    @torch.no_grad()
    def evaluate(self, data_loader, model, device, threshold=0.5):
        """Evaluate using Nearest Class Mean classifier and FC scores"""
        self.eval()
        
        # Ensure class means are computed if we have memory
        if self.memory_list and self.compute_means:
            self.compute_class_means(device)
        
        if self.memory_list and self.class_means is None:
            print("Recomputing class means as they were None but memory exists...")
            self.compute_class_means(device)
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f'Test Task: [{self.current_task_id}]:'
        
        # For video-level AUC calculation
        video_to_logits_ncm = defaultdict(list)
        video_to_logits_fc = defaultdict(list)
        video_to_labels = {}
        
        # For clip-level (frame-level) accuracy calculation
        all_preds_fc = []
        all_targets = []
        
        for batch in metric_logger.log_every(data_loader, 200, header):
            samples = batch['clip'].to(device, non_blocking=True)
            targets = batch['label'].to(device, non_blocking=True).float()
            video_indices = batch['video_idx']
            
            # Always get FC scores
            outputs_fc = self(samples)
            scores_fc = torch.sigmoid(outputs_fc.view(-1))
            
            # Clip-level predictions for accuracy
            preds_fc = (scores_fc > threshold).long()
            all_preds_fc.append(preds_fc.cpu())
            all_targets.append(targets.cpu().long())
            
            # Store scores for video-level AUC calculation
            for i, vid in enumerate(video_indices):
                video_id = int(vid.item())
                video_to_labels[video_id] = targets.view(-1, 1)[i].cpu()
                
                # Store FC scores
                video_to_logits_fc[video_id].append(scores_fc[i].cpu())
            
            # Update metrics (dummy loss since NCM doesn't have a loss)
            metric_logger.update(loss=0.0)
        
        # Calculate clip-level (frame-level) accuracy
        all_preds_fc = torch.cat(all_preds_fc)
        all_targets = torch.cat(all_targets)
        frame_accuracy = (all_preds_fc == all_targets).float().mean().item()
        print(f"=====>Frame-Level-Accuracy (FC): {frame_accuracy:.4f}")
        
        # Calculate video-level AUC for FC scores
        auc_video_fc = utils.compute_video_level_auc(video_to_logits_fc, video_to_labels)
        print(f"=====>Video-Level-AUC (FC): {auc_video_fc:.4f}")
        
        # Calculate video-level EER and Accuracy for FC scores
        
        # Prepare video-level predictions and labels
        video_ids = sorted(video_to_logits_fc.keys())
        video_scores = []
        video_labels = []
        video_preds = []
        
        for vid in video_ids:
            # Average scores for each video
            avg_score = torch.stack(video_to_logits_fc[vid]).mean().item()
            video_scores.append(avg_score)
            video_labels.append(video_to_labels[vid].item())
            # Video-level prediction based on averaged score
            video_preds.append(1 if avg_score > threshold else 0)
        
        # Calculate video-level accuracy
        video_preds = torch.tensor(video_preds)
        video_labels_tensor = torch.tensor(video_labels)
        video_accuracy = (video_preds == video_labels_tensor).float().mean().item()
        print(f"=====>Video-Level-Accuracy (FC): {video_accuracy:.4f}")
        
        # Calculate EER
        fpr, tpr, thresholds = roc_curve(video_labels, video_scores)
        fnr = 1 - tpr
        eer_threshold = thresholds[torch.argmin(torch.tensor(abs(fpr - fnr)))]
        eer = fpr[torch.argmin(torch.tensor(abs(fpr - fnr)))]
        print(f"=====>Video-Level-EER (FC): {eer:.4f} (threshold: {eer_threshold:.4f})")
        
        auc_video = auc_video_fc
        
        # Return stats
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats['frame_accuracy'] = frame_accuracy
        stats['video_accuracy'] = video_accuracy
        stats['auc_fc'] = auc_video_fc
        stats['eer'] = eer
        return auc_video, stats

    def spatiotemporal_freq_kd_loss(self, student_feat: torch.Tensor,
                                    teacher_feat: torch.Tensor,
                                    layer_name: str,
                                    log_amp: bool = True,
                                    device: str = 'cuda') -> dict:
        """
        학생/교사 특징의 '시공간 주파수 표현'만 비교하는 MSE.
        """
        
        # shape 맞추기(필요시)
        if student_feat.shape != teacher_feat.shape:
            # 5D면 trilinear로 (T,H,W) 동시 보정, 4D면 bilinear로 (H,W)만
            if teacher_feat.dim() == 5 and student_feat.dim() == 5:
                student_feat = F.interpolate(student_feat, size=teacher_feat.shape[-3:], mode='trilinear', align_corners=False)
            elif teacher_feat.dim() == 4 and student_feat.dim() == 4:
                student_feat = F.interpolate(student_feat, size=teacher_feat.shape[-2:], mode='bilinear', align_corners=False)
            else:
                raise ValueError("student/teacher dims mismatch")
        
        # 시공간 고주파 표현 추출
        s_mag_t = temporalfreq_representation(student_feat, log_amp=log_amp) 
        with torch.no_grad():
            t_mag_t = temporalfreq_representation(teacher_feat, log_amp=log_amp)
        
        s_mag_s = spatialfreq_representation(student_feat, log_amp=log_amp)
        with torch.no_grad():
            t_mag_s = spatialfreq_representation(teacher_feat, log_amp=log_amp)
            
        st_mag_s = spatiotemporalfreq_representation(student_feat, log_amp=log_amp)
        with torch.no_grad():
            st_mag_t = spatiotemporalfreq_representation(teacher_feat, log_amp=log_amp)
        # Temporal Filter 로 filtering 적용 ( element-wise 곱)
        filt_t = self.learnable_filters[f'temporal_filter_{layer_name}'].to(device) # (C,T) 
        filt_t = filt_t.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # (1,C,T,1,1)
        s_mag_t = s_mag_t * filt_t  + s_mag_t
        
        # Spatial Filter 로 filtering 적용 ( element-wise 곱)
        filt_s = self.learnable_filters[f'spatial_filter_{layer_name}'].to(device) # (C,H,W) 
        filt_s = filt_s.unsqueeze(0).unsqueeze(2) # (1,C,1,H,W)
        s_mag_s = s_mag_s * filt_s  + s_mag_s
        
        
        filt_st = self.learnable_filters[f'spatiotemporal_filter_{layer_name}'].to(device) # (C,T,H,W)
        filt_st = filt_st.unsqueeze(0) # (1,C,T,H,W)
        st_mag_s = st_mag_s * filt_st + st_mag_s
        
        # spatiotemporal filtering 된 정보는 spatial과 temporal 정보의 중복이므로, spatiotemporal 정보와 spatial & temporal 정보와는 orthogonal 하도록 유도
        
        
        temporal_mask_module = self.learnable_masks[f"temporal_mask_{layer_name}"]
        temporal_mask = temporal_mask_module().to(device)  # (T,)
        mask_shape = (1, 1, *temporal_mask.shape, 1, 1)
        masked_student_t_freq = s_mag_t * temporal_mask.view(mask_shape)
        masked_teacher_t_freq = t_mag_t * temporal_mask.view(mask_shape)
        
        spatial_mask_module = self.learnable_masks[f"spatial_mask_{layer_name}"]
        spatial_mask = spatial_mask_module().to(device)  # (H,W)
        mask_shape = (1, 1, 1, *spatial_mask.shape)
        masked_student_s_freq = s_mag_s * spatial_mask.view(mask_shape)
        masked_teacher_s_freq = t_mag_s * spatial_mask.view(mask_shape)
        
        spatiotemporal_mask_module = self.learnable_masks[f"spatiotemporal_mask_{layer_name}"]
        spatiotemporal_mask = spatiotemporal_mask_module().to(device)  # (T,H,W)
        mask_shape = (1, 1, *spatiotemporal_mask.shape) # (1,1,T,H,W)
        spatiotemporal_mask = spatiotemporal_mask.view(mask_shape) 
        masked_student_st_freq = st_mag_s * spatiotemporal_mask.unsqueeze(1) # (N,C,T,H,W)
        masked_teacher_st_freq = st_mag_t * spatiotemporal_mask.unsqueeze(1) # (N,C,T,H,W)
        
        ## covariance alignment loss
        # --- projection modules (init에서 to(device) 완료해두는 걸 권장) ---
        proj_s     = self.learnable_masks[f'spatial_proj_{layer_name}']                 # S 전용
        proj_t     = self.learnable_masks[f'temporal_proj_{layer_name}']                # T 전용
        proj_st_s  = self.learnable_masks[f'spatiotemporal_proj_{layer_name}_spatial']  # ST→S
        proj_st_t  = self.learnable_masks[f'spatiotemporal_proj_{layer_name}_temporal'] # ST→T

        # --- ST 경로(학습 O): 항상 'student의 ST'를 투영 ---
        st_s = proj_st_s(st_mag_s.view(st_mag_s.size(0), st_mag_s.size(1), -1))  # (N,C,D)
        st_t = proj_st_t(st_mag_t.view(st_mag_t.size(0), st_mag_t.size(1), -1))  # (N,C,D)

        # --- S/T 참조(백본만 고정): 입력 detach → S/T projection 모듈은 학습 O ---
        # s : Spatial feature map → spatial projection (N,C,T,H,W) → (N,C, H,W) → flatten(2) → (N,C,D)
        s_ref = proj_s(s_mag_s.detach().mean(2).view(s_mag_s.size(0), s_mag_s.size(1), -1))  # (N,C,D)
        # t : Temporal feature map → temporal projection (N,C,T,H,W) → (N,C,T) → flatten(2) → (N,C,D)
        t_ref = proj_t(t_mag_t.detach().mean((-2, -1)).view(t_mag_t.size(0), t_mag_t.size(1), -1))  # (N,C,D)

        # --- 정렬(consistency) 손실 ---
        consis_st_s = self._Covariance_align_loss(st_s, s_ref)
        consis_st_t = self._Covariance_align_loss(st_t, t_ref)
        consistency_loss = consis_st_s + consis_st_t

        # --- KD(MSE) 손실 ---
        mse_t  = F.mse_loss(masked_student_t_freq,  masked_teacher_t_freq)  * self.temporal_lambda
        mse_s  = F.mse_loss(masked_student_s_freq,  masked_teacher_s_freq)  * self.spatial_lambda
        mse_st = F.mse_loss(masked_student_st_freq, masked_teacher_st_freq) * self.spatiotemporal_lambda

        # --- 최종 스칼라 반환 ---
        total_loss = mse_t + mse_s + mse_st
        return total_loss, consistency_loss
    def _Covariance_align_loss(self, feat_map1, feat_map2, eps: float = 1e-6):
        """
        ST projection(=feat_map1)을 S 또는 T 참조(=feat_map2)에 '정렬'시키는 손실.
        채널별로 표준화 후 token(=T/HW) 축 평균 상관이 없도록 유도.
        """
        assert feat_map1.shape == feat_map2.shape, "shape mismatch"

        B, C = feat_map1.shape[:2]
        x = feat_map1.reshape(B, C, -1)
        y = feat_map2.reshape(B, C, -1)

        # 채널별(z-score) 정규화: 스케일·오프셋 제거
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + eps)
        y = (y - y.mean(-1, keepdim=True)) / (y.std(-1, keepdim=True) + eps)

        # 채널별 상관 (토큰축 평균)
        corr = (x * y).mean(-1)          # (B, C)
        loss = torch.sqrt(torch.mean(corr.pow(2)))
        return loss
        
        



# ---- 2D-FFT magnitude on last two (H, W) dims; supports 4D (N,C,H,W) and 5D (N,C,T,H,W)
def _fft2_mag_last2(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N,C,H,W) 또는 (N,C,T,H,W)
    반환: |FFT2| (입력과 동일한 차원 수)
    """
    if x.dim() == 4:
        # (N,C,H,W)
        if hasattr(torch, "fft") and hasattr(torch.fft, "fft2"):
            spec = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
            mag = spec.abs()
        else:
            spec = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False)  # (N,C,H,W,2)
            mag = torch.norm(spec, dim=-1)  # (N,C,H,W)
        return mag

    elif x.dim() == 5:
        # (N,C,T,H,W) : 마지막 두 축(H,W)만 2D-FFT
        if hasattr(torch, "fft") and hasattr(torch.fft, "fft2"):
            spec = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
            mag = spec.abs()
        else:
            spec = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False)  # (N,C,T,H,W,2)
            mag = torch.norm(spec, dim=-1)  # (N,C,T,H,W)
        return mag
    else:
        raise ValueError(f"Unsupported tensor dim {x.dim()} (expected 4 or 5).")


def _normalize_per_map_2d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    2D 지도 단위 표준화.
    - 4D: (N,C,H,W) → (H,W) 축 기준
    - 5D: (N,C,T,H,W) → 각 (N,C,T) 단위로 (H,W) 축 기준
    """
    if x.dim() == 4:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std  = x.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (x - mean) / std
    elif x.dim() == 5:
        mean = x.mean(dim=(-2, -1), keepdim=True)  # (N,C,T,1,1)
        std  = x.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (x - mean) / std
    else:
        raise ValueError

def spatialfreq_representation(x: torch.Tensor,
                            log_amp: bool = True,
                            eps: float = 1e-6) -> torch.Tensor:
    """
    (N,C,T,H,W) 또는 (N,C,H,W) 입력에서 '공간(H,W) 주파수 성분'만 추출.
    - 시간축 T는 유지, FFT는 (H,W)만.
    - log_amp: log1p(|FFT|)로 동적범위 압축.
    - per-map 표준화로 채널/프레임별 스케일 차 완화.
    반환: 입력과 동일 차원.
    """
    if x.dim() not in (4, 5):
        raise ValueError("x must be 4D or 5D (N,C,H,W) or (N,C,T,H,W).")

    # 2D-FFT magnitude on spatial dims
    mag = _fft2_mag_last2(x)

    if log_amp:
        mag = torch.log1p(mag)

    # 고역 마스크 생성 (H,W 기준)
    H = x.shape[-2]
    W = x.shape[-1]
    mag = _normalize_per_map_2d(mag, eps=eps)
    return mag

def _normalize_per_map_1d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    1D 지도 단위 표준화.
    - 5D: (N,C,T,H,W) → 각 (N,C,T,H,W) 단위로 (T) 축 기준
    - 4D: (N,C,H,W) → (N,C,1,H,W)로 확장 후 (T) 축 기준
    """
    if x.dim() == 5:
        mean = x.mean(dim=2, keepdim=True)  # (N,C,1,H,W)
        std  = x.std(dim=2, keepdim=True).clamp_min(eps)
        return (x - mean) / std
    elif x.dim() == 4:
        x = x.unsqueeze(2)  # (N,C,1,H,W)
        mean = x.mean(dim=2, keepdim=True)
        std  = x.std(dim=2, keepdim=True).clamp_min(eps)
        return (x - mean) / std
    else:
        raise ValueError

def _fft1_mag_dim2(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N,C,T,H,W) 또는 (N,C,1,H,W)
    반환: |FFT1| (입력과 동일한 차원 수)
    """
    if x.dim() == 5:
        # (N,C,T,H,W)
        if hasattr(torch, "fft") and hasattr(torch.fft, "fft"):
            spec = torch.fft.fft(x, dim=2, norm="ortho")
            mag = spec.abs()
        else:
            spec = torch.rfft(x, signal_ndim=1, normalized=False, onesided=False)  # (N,C,T,H,W,2)
            mag = torch.norm(spec, dim=-1)  # (N,C,T,H,W)
        return mag
    else:
        raise ValueError(f"Unsupported tensor dim {x.dim()} (expected 5).")

def temporalfreq_representation(x: torch.Tensor,
                                log_amp: bool = True,
                                eps: float = 1e-6) -> torch.Tensor:
    """
    (N,C,T,H,W) 또는 (N,C,H,W) 입력에서 '시간(T) 주파수 성분'만 추출.
    - 공간축 (H,W)는 유지, FFT는 (T)만.
    - log_amp: log1p(|FFT|)로 동적범위 압축.
    - per-map 표준화로 채널/프레임별 스케일 차 완화.
    반환: 입력과 동일 차원.
    """
    if x.dim() not in (4, 5):
        raise ValueError("x must be 4D or 5D (N,C,H,W) or (N,C,T,H,W).")
    if x.dim() == 4:
        # (N,C,H,W) → (N,C,1,H,W)
        x = x.unsqueeze(2)
    # (N,C,T,H,W)
    N, C, T, H, W = x.shape
    if T < 2:
        raise ValueError("T must be >= 2 for temporal FFT.")
    # 1D-FFT magnitude on temporal dim
    mag = _fft1_mag_dim2(x)  # (N,C,T,H,W)
    if log_amp:
        mag = torch.log1p(mag)
    mag = _normalize_per_map_1d(mag, eps=eps)
    if N == 1 and C == 1:
        mag = mag.squeeze(0).squeeze(0)  # (T,H,W)
    return mag

def _normalize_per_map_3d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    3D 지도 단위 표준화.
    - 5D: (N,C,T,H,W) → 각 (N,C) 단위로 (T,H,W) 축 기준
    - 4D: (N,C,H,W) → (N,C,1,H,W)로 확장 후 (T,H,W) 축 기준
    """
    if x.dim() == 5:
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)  # (N,C,1,1,1)
        std  = x.std(dim=(-3, -2, -1), keepdim=True).clamp_min(eps)
        return (x - mean) / std
    elif x.dim() == 4:
        x = x.unsqueeze(2)  # (N,C,1,H,W)
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        std  = x.std(dim=(-3, -2, -1), keepdim=True).clamp_min(eps)
        return (x - mean) / std
    else:
        raise ValueError

def _fft3_mag_last3(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N,C,T,H,W) 또는 (N,C,1,H,W)
    반환: |FFT3| (입력과 동일한 차원 수)
    """
    if x.dim() == 5:
        # (N,C,T,H,W)
        if hasattr(torch, "fft") and hasattr(torch.fft, "fftn"):
            spec = torch.fft.fftn(x, dim=(-3, -2, -1), norm="ortho")
            mag = spec.abs()
        else:
            spec = torch.rfft(x, signal_ndim=3, normalized=False, onesided=False)  # (N,C,T,H,W,2)
            mag = torch.norm(spec, dim=-1)  # (N,C,T,H,W)
        return mag
    else:
        raise ValueError(f"Unsupported tensor dim {x.dim()} (expected 5).")

def spatiotemporalfreq_representation(x: torch.Tensor,
                                      log_amp: bool = True,
                                      eps: float = 1e-6) -> torch.Tensor:
    """
    (N,C,T,H,W) 또는 (N,C,H,W) 입력에서 '시공간 주파수 성분'만 추출.
    - log_amp: log1p(|FFT|)로 동적범위 압축.
    - per-map 표준화로 채널/프레임별 스케일 차 완화.
    반환: 입력과 동일 차원.
    """
    if x.dim() not in (4, 5):
        raise ValueError("x must be 4D or 5D (N,C,H,W) or (N,C,T,H,W).")
    if x.dim() == 4:
        # (N,C,H,W) → (N,C,1,H,W)
        x = x.unsqueeze(2)
    # (N,C,T,H,W)
    N, C, T, H, W = x.shape 
    if T < 2:
        raise ValueError("T must be >= 2 for temporal FFT.")
    # 3D-FFT magnitude on (T,H,W)
    mag = _fft3_mag_last3(x)  # (N,C,T,H,W)
    if log_amp:
        mag = torch.log1p(mag)
    mag = _normalize_per_map_3d(mag, eps=eps)
    if N == 1 and C == 1:
        mag = mag.squeeze(0).squeeze(0)  # (T,H,W)
    return mag
class GumbelRadialFreqMask(nn.Module):
    """
    Gumbel-Softmax를 사용하여 방사형 기저 이진 주파수 마스크를 생성하는 기본 클래스.
    # 처음에는 모든 주파수가 통과하도록 설정됨.
    """
    def __init__(self, shape, bins, gumbel_tau=0.05):
        super().__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.shape = shape
        self.bins = bins
        self.gumbel_tau = gumbel_tau

        self.radial_logits = nn.Parameter(torch.randn(bins, 2))

        freq_coords = [torch.fft.fftfreq(s) for s in shape]
        grids = torch.meshgrid(*freq_coords, indexing='ij')
        
        R_squared = torch.zeros_like(grids[0])
        for grid in grids:
            R_squared += grid**2
        R = torch.sqrt(R_squared)
        self.register_buffer("radius", R / R.max() if R.max() > 0 else R)
        bin_indices = (self.radius * (self.bins - 1)).long()
        self.register_buffer("bin_indices", bin_indices)
       
        
        
    def forward(self):
        bin_decisions_one_hot = F.gumbel_softmax(self.radial_logits, tau=self.gumbel_tau, hard=True, dim=-1)
        bin_pass_values = bin_decisions_one_hot[:, 1]
        mask = bin_pass_values[self.bin_indices]
        return mask
