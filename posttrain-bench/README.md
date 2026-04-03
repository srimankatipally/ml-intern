# PostTrainBench Integration

## SLURM + Docker/Pyxis (HF cluster)

### 1. Clone repos on cluster
```bash
cd /fsx/$USER
git clone https://github.com/aisa-group/PostTrainBench.git
git clone --branch posttrain-bench https://github.com/huggingface/hf_agent.git
```

### 2. Build Docker image
```bash
cd /fsx/$USER/hf_agent
bash posttrain-bench/slurm/build_and_push.sh
```

### 3. Set env vars
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export HF_TOKEN="hf_..."
```

### 4. Submit a test job (1 hour)
```bash
mkdir -p logs
sbatch posttrain-bench/slurm/submit.sbatch gsm8k hf_agent Qwen/Qwen3-1.7B-Base 1 claude-opus-4-6
```

### 5. Monitor
```bash
squeue -u $USER
tail -f logs/ptb_*.out
# After completion:
cat /fsx/$USER/PostTrainBench/results/hf_agent_claude-opus-4-6_1h/gsm8k_*/solve_out.txt
```

## HTCondor + Apptainer (original PostTrainBench setup)

Copy `solve.sh` to `PostTrainBench/agents/hf_agent/solve.sh` and use `commit.sh`.

## How solve.sh works

The solve script runs inside the container:
1. Clones this repo (branch `posttrain-bench`)
2. Creates Python 3.12 venv via `uv`
3. Installs agent deps
4. Runs `python -m agent.main --max-iterations -1 "$PROMPT"` headlessly

## Running locally (for testing)

```bash
python -m agent.main --max-iterations -1 "Your prompt here"
```
