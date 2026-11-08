#!/bin/bash
# NYU Greene HPC job: train the main model.  Submit with:  sbatch run_hpc.sbatch
# ~200 epochs of a 5M-param ResNet at batch 128 takes roughly 2-3 h on one RTX8000 / A100.
#SBATCH --job-name=cifar5m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

module purge
singularity exec --nv \
  --overlay /scratch/$USER/overlay-15GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; cd $SLURM_SUBMIT_DIR; \
  python train.py --train_csv data/train_labels.csv --model ${MODEL:-resnet_se_343} --epochs ${EPOCHS:-200} --batch_size 128 --lr 0.1 \
                  --sched cosine --warmup_epochs 5 --label_smoothing 0.1 --mix_prob 0.5 \
                  --autoaugment trivial --random_erasing 0.25 --ema_decay 0.999 \
                  --workers 8 --seed ${SEED:-0} --out checkpoints/${MODEL:-resnet_se_343}_s${SEED:-0}"
