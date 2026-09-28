#!/bin/bash
# Sparse feedback: how much verified feedback the record needs before it beats reading without memory.
# Honest stream (the p000 rebuild), Qwen3-4B and Qwen3-8B, one feedback mask (seed 0), a fixed subsample of the stream
# (every Nth event, the same events in every cell). The record walks the whole stream: the ratio is the share of ALL
# events whose verified labels reach it; only the evaluated subsample is generated.
#
# Paths follow the main run: the record is fitted on the model's own +<tag> stream (7 answers, own in slot 6), while
# the central model is evaluated on the plain 6-peer stream.
set -u
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
R=${MODELS_ROOT:?set MODELS_ROOT to the directory holding the models (as paths.models_root in configs/base.yaml)}
OUT=outputs/sparse
RATIOS=${RATIOS:-"0 0.001 0.0025 0.005 0.01 1.00"}
GPU=${GPU:-0}
MODELS=${MODELS:-"q3_4b:Qwen3-4B qwen3_8b:Qwen3-8B"}
STREAMS=${STREAMS:-"capability_challenging_misleading_p000:12"}

for spec in $MODELS; do
  tag=${spec%%:*}; dir=${spec##*:}
  for st in $STREAMS; do
    stream=${st%%:*}; every=${st##*:}
    S=data/${stream}/test.jsonl                    # 6 peers: what the central model reads
    SOWN=data/${stream}+${tag}/test.jsonl          # 7 answers, the model's own in slot 6: what the record walks
    F=outputs/features/$tag/${stream}+own
    SOLO=$OUT/$tag/$stream/solo                    # the same rows in every cell: the record's order does not depend on feedback
    if [ ! -f "$SOLO/eval_metrics.json" ]; then
      echo "[sparse] $tag $stream solo (every $every) $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=$GPU python -m pipeline.evaluate --model $R/$dir --stream $S --every $every \
        --condition solo --mode solo --gamma 0.0 --engine vllm --max-new-tokens 768 --gpu-memory-utilization 0.85 \
        --output $SOLO 2>&1 | grep -E "^\[evaluate\] (solo|wrote)" | tail -2
    fi
    for r in $RATIOS; do
      cell=$OUT/$tag/$stream/r$r
      [ -f "$cell/combination/eval_metrics.json" ] && { echo "[sparse] skip $tag $stream r=$r"; continue; }
      REC=$cell/record.jsonl
      if [ ! -f "$REC" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python -m pipeline.record --stream $SOWN --features $F \
          --peers 7 --order shuffled0 --design qc --dim 256 --lam 100.0 --own-slot 6 \
          --feedback-ratio $r --feedback-seed 0 --out $REC 2>&1 | tail -1
      fi
      [ -f "$REC" ] || { echo "[sparse] FAILED record $tag $stream r=$r"; continue; }
      echo "[sparse] $tag $stream r=$r tilt $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=$GPU python -m pipeline.evaluate --model $R/$dir --record $REC --stream $S --every $every \
        --condition tilt --mode peers --gamma 3.0 --bias-form logratio --engine vllm --max-new-tokens 768 \
        --gpu-memory-utilization 0.85 --output $cell/tilt 2>&1 | grep -E "^\[evaluate\] (tilt|wrote)" | tail -1
      python -m pipeline.combination --record $REC --peers-memory $cell/tilt --question-alone $SOLO --skip-missing \
        --prior 0.5,0.0 --lam 1.0 --output $cell/combination 2>&1 | tail -1
    done
  done
done
echo "[sparse] SPARSE_DONE $(date +%H:%M)"
