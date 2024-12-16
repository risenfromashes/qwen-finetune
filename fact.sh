#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=fact_qwen
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --gres=gpu:80gb:4
#SBATCH --time=2:00:00
#SBATCH --mem=32G

module load anaconda/3
eval "$(conda shell.bash hook)"
conda activate ml

llamafactory-cli train ${SLURM_SUBMIT_DIR}/examples/train_lora/fact_lora_sft.yaml

exit 0
