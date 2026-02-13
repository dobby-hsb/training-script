export MODEL_NAME="gsdf/Counterfeit-V3.0"
export OUTPUT_DIR="wandb/sd15-counterfeit-wds"
export TRAIN_SHARDS_PATH_OR_URL="../datasets/dataset-{000000..000045}.tar"
export MAX_TRAIN_SAMPLES=45400

export HF_HOME="/workspace/huggingface/hf_home"

accelerate launch --mixed_precision="fp16"  training_scripts/train_sd15_wsd.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_shards_path_or_url="$TRAIN_SHARDS_PATH_OR_URL" \
  --max_train_samples=$MAX_TRAIN_SAMPLES \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=24 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="$OUTPUT_DIR"