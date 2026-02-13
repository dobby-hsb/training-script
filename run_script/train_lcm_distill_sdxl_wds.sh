export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="wandb/sdxl-lcm-distill-wds"

export HF_HOME="/workspace/huggingface/hf_home"


accelerate launch training_scripts/train_lcm_distill_sdxl_wds.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --learning_rate=1e-6 --loss_type="huber" --use_fix_crop_and_size --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_samples=45400 \
    --num_train_epochs=10 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="../datasets/dataset-{000000..000045}.tar" \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=5 \
    --train_batch_size=24 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --seed=453645634 \
    --cast_teacher_unet \
    --report_to="wandb" \