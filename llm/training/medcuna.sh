DIR=$(dirname $0)

NPROC_PER_NODE=${NPROC_PER_NODE}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-8000}
PASSTHROUGH=""
while [ "$1" ]; do
    case $1 in
        --nproc )
            shift
            NPROC_PER_NODE=$1
            ;;
        --master-addr )
            shift
            MASTER_ADDR=$1
            ;;
        --master-port )
            shift
            MASTER_PORT=$1
            ;;
        * )
            PASSTHROUGH="$PASSTHROUGH $1"
            ;;
    esac
    shift
done

if [ ! "$NPROC_PER_NODE" ]; then
    NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
fi

torchrun --nproc-per-node=$NPROC_PER_NODE --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $PASSTHROUGH $DIR/medcuna_sft.py
