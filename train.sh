#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=train_qwen
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --gres=gpu:80gb:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G

module load anaconda/3
eval "$(conda shell.bash hook)"
conda activate ml

llamafactory-cli train ${SLURM_SUBMIT_DIR}/examples/train_lora/qwen2_5_lora_sft.yaml

exit 0
