
GPU_NUM=2
NUM_WORKERS=8
WORLD_SIZE=${GPU_NUM}
MASTER_PORT=10003

MODEL="msfd"
ARCH="torch_hub_i3d_r50"
SEED=42

# 날짜 태그
DATE=$(date +%Y%m%d)
# 체크포인트 저장 디렉토리
SAVE_ROOT="./outputs/${DATE}_${MODEL}_${ARCH}"
# 로그 파일
LOG_FILE="train_${DATE}_${MODEL}.log"

BATCH_SIZE=8
# --------------------------
# 2) 실행 정보 출력
# --------------------------
echo ">>> GPU_NUM     = $GPU_NUM"
echo ">>> NUM_WORKERS = $NUM_WORKERS"
echo ">>> WORLD_SIZE  = $WORLD_SIZE"
echo ">>> MASTER_PORT = $MASTER_PORT"
echo ">>> MODEL       = $MODEL"
echo ">>> ARCH        = $ARCH"
echo ">>> SEED        = $SEED"
echo ">>> SAVE_ROOT   = $SAVE_ROOT"
echo ">>> DATA_PATH   = $DATA_PATH"
echo ">>> LOG_FILE    = $LOG_FILE"
echo ">>> BATCH_SIZE  = $BATCH_SIZE"
# 디렉토리 생성
mkdir -p "$SAVE_ROOT"

# --------------------------
# 3) 분산 학습 혹은 단일 GPU/CPU 실행
# --------------------------
nohup python -m torch.distributed.launch \
    --nproc_per_node="${GPU_NUM}" \
    --master_port="${MASTER_PORT}" \
    --use_env train.py \
    --world_size="${WORLD_SIZE}" \
    --batch_size="$BATCH_SIZE" \
    --model_name="$MODEL" \
    --architecture="$ARCH" \
    --num_workers="$NUM_WORKERS" \
    --seed="$SEED" \
    --save_root="$SAVE_ROOT" \
  > "$LOG_FILE" &
