#!/bin/bash
# Script to run audio feature transformation using trained CycleGAN generators.
# Usage: bash scripts/transform.sh [optional_args_for_feature_transform.py]

python -m src.transform.feature_transform "$@" 