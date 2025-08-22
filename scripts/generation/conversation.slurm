#!/bin/bash

#SBATCH --job-name=rollout
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --output=slurm/%x_%j.log
#SBATCH --error=slurm/%x_%j.log

cd /home/k_ishikawa/workspace/spalign/scripts/generation

deactivate
conda deactivate

export RESULTS_DIR=/nas/k_ishikawa/datasets/HappyRat/synthesis
export HF_HUB_ENABLE_HF_TRANSFER=0

uv run generate_conversation.py 250822 \
    -m /nas/c_arata/axolotl-output/models/24B-Kaji-HR-SFT-20250816 \
    -p metadata \
    --scenario Spiral-AI/Synthesized-Scenario-20250812 \
    --scenario_subset app \
    --max_concurrency 1