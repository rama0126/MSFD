import os
import json
import argparse
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import random
import numpy as np
import utils
from models.init_optimizer import init_optimizer
import models
from datasets.dataset_clipv2 import VideoClips
from datasets.sampler import ConsecutiveClipSampler

# 고정 상수
TASK_LIST = ['FF', 'DFD', 'DFDCP', 'CDF', 'FFIW', 'KoDF']
TEST_LIST = ['FF', 'DFD', 'DFDCP', 'CDF','FFIW', 'KoDF']
        
def str2bool(v):
    return v.lower() in ('true', '1', 'yes', 'y')
def get_args_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 데이터
    parser.add_argument('--txt_root',       type=str, default='./protocol1_dataset_txt')
    parser.add_argument('--TASK_LIST',      type=str, default='FF,DFD,CDF,DFDCP,FFIW,KoDF')
    parser.add_argument('--TEST_LIST',      type=str, default='FF,DFD,CDF,DFDCP,FFIW,KoDF')
    parser.add_argument('--save_root',      type=str, default='./outputs')
    parser.add_argument('--seed', type=int, default=42)
    # 모델
    parser.add_argument('--model_name',     type=str, default='base')
    parser.add_argument('--architecture',   type=str, default='torchvision_r3d_18')
    parser.add_argument('--pretrained',     action='store_false')
    parser.add_argument('--separate_classifier', action='store_false')
    # 학습 하이퍼파라미터
    parser.add_argument('--batch_size',     '-b', type=int,   default=16)
    parser.add_argument('--initial_lr',     '-l', type=float, default=0.01)
    parser.add_argument('--continual_lr',   '-L', type=float, default=0.0001)
    parser.add_argument('--initial_epochs', '-e', type=int,   default=20)
    parser.add_argument('--continual_epochs','-E', type=int,   default=20)
    parser.add_argument('--weight_decay',   '-w', type=float, default=1e-4)
    parser.add_argument('--optimizer',      type=str,   default='sgd-momentum')
    parser.add_argument('--lr_scheduler',   type=str,   default='step')
    parser.add_argument('--lr_sche_step_size', type=int, default=5)
    parser.add_argument('--lr_sche_gamma', type=float , default=0.5)
    
    parser.add_argument('--test_step_size', type=int,   default=20)
    parser.add_argument('--eval_threshold', type=float, default=0.5)
    
    parser.add_argument('--memory_size', type=int, default= 200)
    
    parser.add_argument('--lambda_distill', type=float, default=0.5)
    parser.add_argument('--kd_ft_weight', type=float, default=0.5)
    


    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--restart_epoch', type=int, default=0)
    parser.add_argument('--restart_task', type=int, default= 0)
    parser.add_argument('--restart_test',action='store_true',)
    ### About distributed training ###
    parser.add_argument('--distributed', type=str2bool, default=True)
    parser.add_argument('--local_rank', type=int, default=0) 
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes') # fix: 1 -> 4 (학습에 사용되는 gpu 개수)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training') # (?)
    parser.add_argument('--num_workers', default=4, type=int) # fix: 10 -> 16 -> 12(num_workers = 4 * num_of_gpu)

    return parser

def load_paths(txt_root, prefix, phase):
    """{prefix}_train.txt 또는 {prefix}_val.txt에서 (path,label) 튜플 리스트를 반환."""
    fp = os.path.join(txt_root, f"{prefix}_{phase}.txt")
    with open(fp) as f:
        lines = [l.strip().split(',') for l in f]
    return [(p, int(l)) for l, p in lines]

def make_test_loaders(args):
    loaders = {}
    print("======== LOAD TEST Dataloader ==========")
    for name in TEST_LIST:
        paths = load_paths(args.txt_root, name, 'val')
        ds = VideoClips(paths, is_train=False,
            max_frames_per_video = 110,)
        sampler = ConsecutiveClipSampler(ds.clips_per_video)
        loaders[name] = DataLoader(
            ds,
            batch_size=args.batch_size*2,
            num_workers=args.num_workers,
            sampler=sampler
        )
        print(f"=====>Totally {len(ds)} {name} test video clips...")
        print(f"has {len(ds.paths)} Videos")
        print(f"has {sum(sampler.clips_per_video)} Clips")
    print("========================================")
    return loaders
def load_memory(task_id, save_root):
    memory_root = os.path.join(save_root, 'memories')
    memory_paths = os.path.join(memory_root,f"{(task_id-1)}_memory.json")
    if not os.path.exists(memory_paths):
        return None
    try:
        with open(memory_paths, 'r') as f:
            memory_data = json.load(f)
        print(f"Successfully loaded memory from {memory_paths}")
        return memory_data
    except json.JSONDecodeError as e:
        print(f"Error loading memory file {memory_paths}: {e}")
        print("Corrupted memory file detected. Starting with empty memory.")
        return None
    except Exception as e:
        print(f"Unexpected error loading memory file {memory_paths}: {e}")
        return None
def main(args):

    global TASK_LIST, TEST_LIST
    TASK_LIST = list(args.TASK_LIST.split(','))
    TEST_LIST = list(args.TEST_LIST.split(','))
    utils.init_distributed_mode(args)
    # Call dataset using args.dataset_sequences
    device = torch.device(args.gpu)
    test_loaders = make_test_loaders(args)
    
    # 모델, DDP
    ModelClass = getattr(models, args.model_name.lower())
    model = ModelClass(
        args=args,
        architecture=args.architecture,
        pretrained=args.pretrained,
        separate_classifier=args.separate_classifier
    ).to(device)
    if args.distributed:
        # find_unused_parameters=True: at task 0 (and for buffer-free steps) the
        # distillation masks/filters receive no gradient, so DDP must tolerate
        # parameters that don't participate in a given backward pass.
        model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_core = model.module
    else:
        model_core = model
    print("======== LOAD Model Desc. ==========")
    print(f'Contnual Method: {args.model_name}')
    print(f"BACKBONE: {args.architecture}")
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {num_params:.2f}M")
    os.makedirs(args.save_root, exist_ok=True)
    if args.distributed :
        if model.module.memory_dual:
            args.batch_size = args.batch_size //2
    else:
        if model.memory_dual:
            args.batch_size = args.batch_size // 2
            
            
    print(f"Save on {args.save_root}")
    # Task 루프
    for task_id, prefix in enumerate(TASK_LIST):
        if args.resume is not None:
            if task_id < args.restart_task:
                continue
        train_paths = load_paths(args.txt_root, prefix, 'train')
        
        if args.distributed :
            model.module.before_task(task_id) 
        else:
            model.before_task(task_id)
        if task_id == 0: 
            memory = None
        else:
            memory = load_memory(task_id, args.save_root)
            if args.distributed :
                model.module.memory_list = memory
            else:
                model.memory_list = memory

        epochs = args.initial_epochs if task_id == 0 else args.continual_epochs
        lr = args.initial_lr if task_id == 0 else args.continual_lr

        optimizer, scheduler = init_optimizer(
            model, lr, args.optimizer, args.lr_scheduler, epochs, args
        )
        print("------"*4)
        print('epochs : ', epochs, )
        print('scheduler : ', args.lr_scheduler)
        print('optimizer : ', args.optimizer, 'lr : ', lr, 'weight_decay : ', args.weight_decay)
        print('batch_size : ', args.batch_size)
        print("------"*4)
        print(f"\n=== Training Task {task_id}: {prefix} ===")
        ds = VideoClips(
                train_paths,
                is_train=True,
                memory_list=memory,
                memory_dual=model_core.memory_dual,
                domain = task_id,
            )
        sampler = DistributedSampler(ds) if args.distributed else None
        train_loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                # num_workers=1,
                sampler=sampler,
                pin_memory = True,
                persistent_workers=True,
                prefetch_factor=4,
            )
        start_epoch = 0
        if args.resume is not None and task_id == args.restart_task:
            print(f"LOADED MODEL!!")
            print(f"model from : {args.resume}")
            weights = torch.load(args.resume)
            if args.distributed:
                if  hasattr(model.module, 'old_model'):
                    if model.module.old_model is not None:
                        print("=> Adjusting state_dict keys for old_model...")
                        new_dict = model.module.state_dict()
                        for k in new_dict.keys():
                            if k in weights.keys():
                                new_dict[k] = weights[k]
                        weights = new_dict
                try:
                    model.module.load_state_dict(weights)
                except RuntimeError as e:
                    print(f"RuntimeError during state_dict loading: {e}")
                    print("Attempting to load state_dict with strict=False")
                    model.module.load_state_dict(weights, strict=False)
                    print("State_dict loaded with strict=False. Please verify model integrity.")
                    # 필요시 추가 조치 (예: 누락된 키 출력 등)
                    print("Missing keys:", set(weights.keys()) - set(model.module.state_dict().keys()))
                    print("Unexpected keys:", set(model.module.state_dict().keys()) - set(weights.keys()))
            else:
                
                if  hasattr(model.module, 'old_model'):
                    if model.old_model is not None:
                        print("=> Adjusting state_dict keys for old_model...")
                        new_dict = model.state_dict()
                        for k in new_dict.keys():
                            if k in weights.keys():
                                new_dict[k] = weights[k]
                        weights = new_dict
                    model.load_state_dict(weights)
            if args.distributed:
                model.module.before_task(task_id) 
            else:
                model.before_task(task_id)
            start_epoch = args.restart_epoch
        if start_epoch == epochs:
            print('-'*10)
            print(f"Skipping current task {task_id}")
            print('-'*10)
            if args.restart_test:
                print(f"Restarting test for task {task_id}...")
                # 평가
                eval_fn = model.module.evaluate if args.distributed else model.evaluate
                logs = OrderedDict(epoch=epochs, n_parameters=f"{num_params:.2f}M")
                for name, loader in test_loaders.items():
                    auc, es = eval_fn(loader, model, device, threshold=args.eval_threshold)
                    print(f"{name} AUC={auc:.4f}")
                    logs[f"{name}_AUC"] = f"{auc:.5f}"
                    for m, val in es.items():
                        logs[f"{name}_{m}"] = f"{val:.7f}"
                # 로그 저장
                task_dir = os.path.join(args.save_root, f"{task_id}_{prefix}")
                os.makedirs(task_dir, exist_ok=True)
                with open(os.path.join(task_dir, "log.json"), 'a') as f:
                    json.dump(logs, f, indent=2)
                    
                # 평가
                # eval_fn = model.module.evaluate if args.distributed else model.evaluate
                # logs = OrderedDict(
                #     **{f"train_{k}": f"{v:.7f}" for k, v in stats.items()},
                #     epoch=epoch,
                #     n_parameters=f"{num_params:.2f}M"
                # )
                # for name, loader in test_loaders.items():
                #     auc, es = eval_fn(loader, model, device, threshold=args.eval_threshold)
                #     print(f"{name} AUC={auc:.4f}")
                #     logs[f"{name}_AUC"] = f"{auc:.5f}"
                #     for m, val in es.items():
                #         logs[f"{name}_{m}"] = f"{val:.7f}"

                # # 로그 저장
                # with open(os.path.join(task_dir, "log.json"), 'w') as f:
                #     json.dump(logs, f, indent=2)
            pass
        else:
            if args.resume is not None and task_id == args.restart_task:
                if args.restart_epoch != 0:
                    print(f"LOADED Optimizer!!")
                    print(f"optimzier from : {args.resume.replace('model','optim')}")
                    optimizer.load_state_dict(torch.load(args.resume.replace('model','optim')))
                    if scheduler:
                        scheduler.load_state_dict(torch.load(args.resume.replace('model','sched')))
            for epoch in range(start_epoch,epochs):
                # 데이터로더
                if args.distributed:
                    train_loader.sampler.set_epoch(epoch)
                print(f"=====>Totally {len(ds)} video clips...")
                print(f"has {len(ds.paths)} Videos")
                if task_id > 0:
                    print(f"has memory data {ds.memory_len} clips")
                
                train_one_epoch = model.module.train_one_epoch if args.distributed else model.train_one_epoch
                # 학습
                stats = train_one_epoch(
                    model, train_loader, optimizer, device, epoch,
                    threshold=args.eval_threshold
                )
                if scheduler:
                    scheduler.step()

                # 체크포인트 & 평가
                if epoch != 0:
                    if epoch % args.test_step_size == 0 or epoch == epochs-1:
                        task_dir = os.path.join(args.save_root, f"{task_id}_{prefix}")
                        os.makedirs(task_dir, exist_ok=True)
                        core = model_core
                        torch.save(core.state_dict(), os.path.join(task_dir, f"model_{epoch+1}.pth"))
                        torch.save(optimizer.state_dict(), os.path.join(task_dir, f"optim_{epoch+1}.pth"))
                        if scheduler:
                            torch.save(scheduler.state_dict(), os.path.join(task_dir, f"sched_{epoch+1}.pth"))
                    
                        # 평가
                        eval_fn = model.module.evaluate if args.distributed else model.evaluate
                        logs = OrderedDict(
                            **{f"train_{k}": f"{v:.7f}" for k, v in stats.items()},
                            epoch=epoch,
                            n_parameters=f"{num_params:.2f}M"
                        )
                        for name, loader in test_loaders.items():
                            auc, es = eval_fn(loader, model, device, threshold=args.eval_threshold)
                            print(f"{name} AUC={auc:.4f}")
                            logs[f"{name}_AUC"] = f"{auc:.5f}"
                            for m, val in es.items():
                                logs[f"{name}_{m}"] = f"{val:.7f}"

                        # 로그 저장
                        with open(os.path.join(task_dir, "log.json"), 'w') as f:
                            json.dump(logs, f, indent=2)
        ds = VideoClips(
                train_paths,
                is_train=False,
            )
        model_core.after_task(task_id, ds, device)
        
    print("=== Training Complete ===")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # GPU 메모리 비우기
    torch.cuda.empty_cache()
    if args.distributed:
        # Seed Setting
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        main(args)
