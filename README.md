# Preserving Knowledge across Space and Time for Continual Video Deepfake Detection

Official repository of **"Preserving Knowledge across Space and Time for Continual Video Deepfake Detection"**, ECCV 2026.


🌐 [[Project Page]](https://rama0126.github.io/MSFD/)

<img width="1001" height="378" alt="image" src="https://github.com/user-attachments/assets/ab0f29aa-cdb5-4dbd-8144-91a2c6168918" />

## TL;DR
MSFD helps a video deepfake detector learn new fake videos without forgetting old ones. It keeps spatial, temporal, and spatiotemporal cues separately.

---

## Contents
- [Installation](#installation)
- [Datasets](#datasets)
- [Preprocessing ](#preprocessing)
- [Data layout & list format](#data-layout--list-format)
- [Continual protocols](#continual-protocols)
- [Training](#training)

---

## Installation

Python **3.9**, PyTorch **2.8** (CUDA), 2× GPUs recommended (training always runs under DDP).

```bash
conda create -n cdvd python=3.9 -y
conda activate cdvd
# install the torch/torchvision wheels that match your CUDA runtime first:
pip install torch==2.8.0 torchvision==0.23.0
pip install -r requirements.txt
```


---

## Datasets

Download each source dataset from its official page and preprocess it as described below.

| Prefix (in lists) | Dataset | Link |
|---|---|---|
| `FF`    | FaceForensics++ | https://github.com/ondyari/FaceForensics |
| `DF40`* | DF40 | https://github.com/YZY-stack/DF40 |
| `DFDCP` | DFDC (Preview) | https://ai.meta.com/datasets/dfdc/ |
| `CDF`   | Celeb-DF v2 | https://github.com/yuezunli/celeb-deepfakeforensics |
| `FFIW`  | FFIW | https://github.com/tfzhou/FFIW |
| `KoDF`  | KoDF | https://deepbrainai-research.github.io/kodf/ |
| `AIGVDBench`* | AIGVDBench | https://huggingface.co/datasets/AIGVDBench/AIGVDBench |

\* `DF40` supplies the extra manipulation-type tasks (`BF, FD, HR, MS, MCNET, ST`).
\* `AIGVDBench` supplies the AI-generated-video tasks (`T2V, I2V, V2V, SORA`). Use 'EasyAnimate', 'PyramidFlow', 'LTX', 'SORA'. 

---

## Preprocessing

We follow the face-processing pipeline of **FTCN** (*Exploring Temporal Coherence for More General Video Face Forgery Detection*, ICCV 2021 — https://github.com/yinglinzheng/FTCN). For every video we detect and **track** faces across frames, then **crop-and-align** each tracked face and dump the aligned face frames into a per-video folder. The FTCN tooling lives under [`preprocessing/`](preprocessing/) (`test_tools/`), and the driver is [`preprocessing/preprocess.py`](preprocessing/preprocess.py) (`FasterCropAlignXRay(256)` → aligned faces, saved as sequential frame images per video).

```bash
python preprocessing/preprocess.py \
    --video_root  /path/to/<Dataset>/videos \
    --crop_root   /workspace/datasets/<Dataset>/.../crop_images
```

This produces, per source video, a frame folder:

```
<Dataset>/.../crop_images/<video_id>/00000.png, 00001.png, ...
```

At training/eval time [`datasets/dataset_clipv2.py`](datasets/dataset_clipv2.py) slices each folder into non-overlapping **32-frame clips** at **224×224** with ImageNet normalization (videos with `< 32` frames are skipped).

**We have not preprocessed the AIGDVBench dataset.**
 
---
## Data layout & list format

The dataset root is `/workspace/datasets` (`ROOT` in [`datasets/dataset_clipv2.py`](datasets/dataset_clipv2.py)); the `{ROOT}` token in the lists is substituted with it at load time. Point `ROOT` to your own location if needed.

Each split is a text file `protocolX_dataset_txt/<PREFIX>_{train,val}.txt`, **one clip-source folder per line** as `label,path` (`0` = real, `1` = fake):

```
1,{ROOT}/FaceForensics/c23/Deepfakes/crop_images/071_054
0,{ROOT}/FaceForensics/c23/original/crop_images/033
```

---

## Continual protocols

Pick a protocol with `--txt_root` and set the task/eval order with `--TASK_LIST` / `--TEST_LIST`. After each task, the model is evaluated on **every** task in `TEST_LIST` (video-level AUC / EER) to measure retention.

| Protocol | `--txt_root` | Tasks (`TASK_LIST` = `TEST_LIST`)  | Purpose |
|---|---|---|---|
| **P1** | `protocol1_dataset_txt` | `FF, DFD, CDF, DFDCP, FFIW, KoDF` | 6-dataset domain-incremental benchmark |
| **P2** | `protocol2_dataset_txt` | `T2V, I2V, V2V, SORA`  | Fake Incremental (AI-generated-video) stream (AIGVDBench) |
| **P3** | `protocol3_dataset_txt` | `FF, DFDCP, DFD, CDF`  | Few Shot 4-dataset benchmark |


---

## Training

Training is always distributed (DDP), launched via `torch.distributed.launch`. The provided [`run.sh`](run.sh) runs the default P1 order with the R3D18 backbone:

```bash
bash run.sh
```

Or explicitly:

```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=10003 --use_env train.py \
    --world_size=2 \
    --model_name=msfd \
    --architecture=torchvision_r3d_18 \
    --txt_root=./protocol1_dataset_txt \
    --TASK_LIST=FF,DFD,CDF,DFDCP,FFIW,KoDF \
    --TEST_LIST=FF,DFD,CDF,DFDCP,FFIW,KoDF \
    --batch_size=8 --num_workers=8 --seed=42 \
    --save_root=./outputs/msfd_p1
```

Key arguments (see `get_args_parser` in [`train.py`](train.py)):

- `--model_name` — method; `msfd` for the proposed model, `base` for plain fine-tuning.
- `--architecture` — `torchvision_r3d_18` (default in `run.sh`).
- `--memory_size` — replay-buffer size (default 200).
- `--initial_epochs` / `--continual_epochs`, `--initial_lr` / `--continual_lr`, `--test_step_size`.

Per task, checkpoints and a per-test-set metrics log are written under `--save_root/<task_id>_<prefix>/`, and the replay buffer under `--save_root/memories/`.


