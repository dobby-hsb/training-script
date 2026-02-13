export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="wandb"
export CSV_PATH="meta_temp.csv"
export IMAGES_DIR="dataset_images"

accelerate launch training_scripts/train_lcm_distill_sdxl_csv.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --learning_rate=1e-6 --loss_type="huber" --use_fix_crop_and_size --ema_decay=0.95 --adam_weight_decay=0.0 \
    --num_train_epochs=10 \
    --dataloader_num_workers=2 \
    --use_csv_dataset \
    --csv_path=$CSV_PATH \
    --images_dir=$IMAGES_DIR \
    --validation_steps=50 \
    --checkpointing_steps=100 --checkpoints_total_limit=10 \
    --train_batch_size=1 \
    --vae_encode_batch_size=1 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --allow_tf32 \
    --seed=453645634

    # --report_to=wandb \