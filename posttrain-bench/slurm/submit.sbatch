#!/bin/bash
#SBATCH --job-name=ptb
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ptb_%j.out
#SBATCH --error=logs/ptb_%j.err

# Usage: sbatch submit.sbatch <eval_task> <agent> <model> <hours> <agent_config>
# Example: sbatch submit.sbatch gsm8k hf_agent Qwen/Qwen3-1.7B-Base 1 claude-opus-4-6

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_task.sh" "$@"
