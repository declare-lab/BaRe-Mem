#!/usr/bin/env bash
# Run an experiment of the BaRe-Mem pipeline. Everything is in the config (configs/base.yaml, configs/experiments/*.yaml,
# and the datasets / models / peer sets registered in configs/datasets, configs/models, configs/peers).
#   bash run.sh configs/experiments/main.yaml --dry-run        the jobs, and which are already done
#   bash run.sh configs/experiments/main.yaml --smoke          48 events, every step, into outputs/smoke/ (run this first)
#   bash run.sh configs/experiments/main.yaml                  the whole experiment (resumable: finished work is skipped)
#   bash run.sh configs/experiments/misleading.yaml --steps streams evaluate --set "datasets=[capability_supported_misleading_p050]"
# Long runs: nohup bash run.sh configs/experiments/<name>.yaml > logs/<name>.out 2>&1 &    (or inside tmux)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ENV=${BAREMEM_ENV:-bare-mem}   # the conda env with requirements_qwen3.txt
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV" ] && command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)" && conda activate "$ENV" || echo "could not activate conda env $ENV; using $(command -v python)"
fi
python -c "import vllm, transformers, torch, yaml; assert vllm.__version__ == '0.8.5', vllm.__version__" \
  || { echo "this needs vLLM 0.8.5 (the tilt's attention kernels are patched from its source), transformers 4.56.2, torch 2.6.0: pip install -r requirements_qwen3.txt"; exit 1; }
# FEEDBACK_CODE_EXEC_ALLOW: peers' and models' programs are executed to grade them (subprocess with limits, not a sandbox:
# use a disposable machine). VLLM_ENABLE_V1_MULTIPROCESSING=0: in-process engines (the tilt registers per-prompt biases).
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
python -m pipeline.run "$@"
