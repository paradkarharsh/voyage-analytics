#!/usr/bin/env bash
# Re-execute all notebooks in place:  bash scripts/run_notebooks.sh
cd "$(dirname "$0")/../notebooks" || exit 1
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "$nb"; done
