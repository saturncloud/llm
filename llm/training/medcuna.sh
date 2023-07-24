WORKSPACE=$HOME/workspace
DIR=$(dirname $0)

torchrun --nproc_per_node=4 --master-addr localhost --master-port 8000 $DIR/medcuna_sft.py
