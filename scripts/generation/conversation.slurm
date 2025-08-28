#!/bin/bash

#SBATCH --job-name=rollout
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --output=/nas/k_ishikawa/logs/slurm/%x_%j.log
#SBATCH --error=/nas/k_ishikawa/logs/slurm/%x_%j.log

cd /home/k_ishikawa/workspace/spalign/scripts/generation

deactivate
conda deactivate

export RESULTS_DIR=/nas/k_ishikawa/datasets/HappyRat/synthesis
export HF_HUB_ENABLE_HF_TRANSFER=0
export GEMINI_API_KEY=${GEMINI_API_KEY}

uv run generate_conversation.py 250828-multilingual \
    -m /nas/c_arata/axolotl-output/models/24B-HR-SFT-Multilingual-20250826 \
    -p metadata \
    --scenario Spiral-AI/Synthesized-Scenario-20250812 \
    --scenario_subset app