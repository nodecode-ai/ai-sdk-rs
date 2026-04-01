#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <baseline-name> [bench...]" >&2
  exit 1
fi

baseline_name="$1"
shift

if [[ $# -eq 0 ]]; then
  set -- openai_responses streaming_pipeline core_hot_paths provider_matrix
fi

for bench_name in "$@"; do
  echo "Comparing bench '${bench_name}' against criterion baseline '${baseline_name}'"
  cargo bench --bench "${bench_name}" -- --baseline "${baseline_name}"
done
