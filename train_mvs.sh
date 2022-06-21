DATASET=$1
# DATASET=Lego
PARTITION=$2
NUM_GPU=$3

SRUN="srun -p $PARTITION -n1 --mpi=pmi2 --gres=gpu:$NUM_GPU --ntasks-per-node=1 --cpus-per-task=8 -x SH-IDC1-10-5-37-39 --job-name=train_generator --kill-on-bad-exit=0"
PYTHON="/mnt/lustre/fanrui/miniconda3/envs/kilonerf/bin/python -u "
# PYTHON="/mnt/lustre/share/spring/conda_envs/miniconda3/envs/s0.3.4/bin/python -u "

echo "[INFO] DATASET: $DATASET"
echo "[INFO] Partition: $PARTITION, Used GPU Num: $NUM_GPU. "
echo "[INFO] SRUN: $SRUN"
echo "[INFO] PYTHON: $PYTHON"

# SCRIPT1="train_kilonerf_new.py"
SCRIPT1="run_nerf.py"

# PYTHON_SCRIPT1="$PYTHON $SCRIPT1 --config ./configs/kilonerfsv3/$CONFIG --test_only"
PYTHON_SCRIPT1="$PYTHON $SCRIPT1 --config ./configs/kilonerfs/kilonerf_pretrain_BlendedMVS_base01.py --dataname $DATASET"
PYTHON_SCRIPT2="$PYTHON $SCRIPT1 --config ./configs/kilonerfs/kilonerf_distill_BlendedMVS_base01.py --dataname $DATASET"
PYTHON_SCRIPT3="$PYTHON $SCRIPT1 --config ./configs/kilonerfs/kilonerf_finetune_BlendedMVS_base01.py --dataname $DATASET"


echo "$PYTHON_SCRIPT1"
$PYTHON_SCRIPT1
echo "$PYTHON_SCRIPT2"
$PYTHON_SCRIPT2
echo "$PYTHON_SCRIPT3"
$PYTHON_SCRIPT3
