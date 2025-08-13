#!/bin/bash

#SBATCH --job-name=rollout
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --output=slurm/%x_%j.log
#SBATCH --error=slurm/%x_%j.log

cd /home/k_ishikawa/workspace/spalign/scripts/generation

deactivate
conda deactivate

uv run conversation.py 250813-orpo \
    -m /nas/c_arata/axolotl-output/models/24B-HR-ORPO-20250811 \
    -p dataset \
    --scenario Spiral-AI/Synthesized-Scenario-20250812 \
    --scenario_subset default
