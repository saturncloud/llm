WORKSPACE=$HOME/workspace
DIR=$(dirname $0)

deepspeed --num_gpus=4 $DIR/medcuna_sft.py \
    --deepspeed "llm/deepspeed_configs/bf16.json"

# torchrun --nproc_per_node=4 --master-addr localhost --master-port 8000 $DIR/medcuna_sft.py --deepspeed $DIR/deepspeed_configs/bf16.json
