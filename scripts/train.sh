#!/bin/bash
# Script to run CycleGAN training.
# Usage: bash scripts/train.sh [optional_args_for_train_cyclegan.py]

python -m src.train.train_cyclegan "$@" 