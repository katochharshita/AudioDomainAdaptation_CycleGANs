#!/bin/bash
# Script to run evaluation of audio embeddings.
# This script can trigger different evaluation types based on arguments.
# For 1D-Triplet CNN evaluation (EER, TMR): python -m src.eval.eval_triplet_cnn
# For X-vector evaluation (EER, TMR): python -m src.eval.eval_xvector
# Usage: bash scripts/eval.sh [optional_args_for_eval_scripts]

# Example: Run 1D-Triplet CNN specific evaluation
# python -m src.eval.eval_triplet_cnn "$@"

# Example: Run X-vector speaker verification evaluation
# python -m src.eval.eval_xvector "$@" 