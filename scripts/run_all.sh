#!/usr/bin/env bash
set -e

CONFIGS=(
  configs/baseline.yaml
  configs/exp_embedder_minilm.yaml
  configs/exp_embedder_e5.yaml
  configs/exp_embedder_rubert.yaml
  configs/exp_chunking_fixed_800.yaml
  configs/exp_chunking_fixed_1200.yaml
  configs/exp_chunking_fixed_1600.yaml
  configs/exp_chunking_recursive_800.yaml
  configs/exp_chunking_recursive_1200.yaml
  configs/exp_chunking_recursive_1600.yaml
  configs/exp_chunking_sentence_1200.yaml
  configs/exp_preprocessing_minimal.yaml
  configs/exp_preprocessing_aggressive.yaml
  configs/exp_retriever_bm25.yaml
  configs/exp_retriever_ensemble.yaml
  configs/exp_reranker_bge.yaml
  configs/exp_reranker_minilm.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "=========================================="
  echo "Running: $cfg"
  echo "=========================================="
  python -m scripts.run_retrieval --config "$cfg" || echo "FAILED: $cfg"
done

echo ""
echo "All experiments done. Results in results/retrieval/"
