WORKSPACE=$HOME/workspace
DIR=$(dirname $0)

# export PATH=/usr/local/cuda-11.7/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

deepspeed --num_gpus=4 $DIR/train.py \
    --ddp_timeout=360000 \
    --output_dir "$WORKSPACE/models/flacuna-7b" \
    --deepspeed "llm/deepspeed_configs/bf16.json" \
    --bf16 True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --q_lora \
    --model_name_or_path "lmsys/vicuna-7b-v1.3" \
    --data_path "$WORKSPACE/datasets/flan_mini.json" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.0001 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
