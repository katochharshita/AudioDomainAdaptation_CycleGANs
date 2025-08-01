# Data Directory Structure for AudioDomainAdaptation_CycleGANs

This directory is intended to host your audio datasets, organized into training, validation,
and testing sets for both source (A) and target (B) domains.

## Expected Structure:

```
data/
├── <dataset_name>/                    # e.g., 'my_singing_speaking_data' or 'concat_ms'
│   ├── train/
│   │   ├── A/                         # Training audio files for Domain A (e.g., singing)
│   │   │   ├── audio1.wav
│   │   │   ├── audio2.wav
│   │   │   └── ...
│   │   └── B/                         # Training audio files for Domain B (e.g., speaking)
│   │       ├── audio1.wav
│   │       ├── audio2.wav
│   │       └── ...
│   ├── val/                           # Validation audio files
│   │   ├── A/
│   │   └── B/
│   └── test/                          # Test audio files
│       ├── A/
│       └── B/
│
├── <another_dataset_name>/            # Optional: for multiple datasets
│   ├── train/
│   │   ├── A/
│   │   └── B/
│   └── ...
│
└── features/                          # (Optional) Directory for pre-extracted features (e.g., .mat files for Triplet CNN)
    ├── MS_singing/
    │   ├── speaker1/
    │   │   ├── audio1.mat
    │   │   └── ...
    │   └── speaker2/
    │       └── ...
    └── MS_speaking_transformed/
        └── ...
```

## Guidelines:

*   **Audio Format**: All audio files should ideally be in `.wav` format. Ensure consistent sampling rates and channels as per your model's requirements.
*   **Dataset Name**: The `<dataset_name>` should correspond to the `dataset_name` variable or argument used in your training scripts (e.g., in `src/train/train_cyclegan.py`).
*   **Feature Files**: For evaluation with models like the 1D-Triplet CNN that expect pre-extracted features (e.g., `.mat` files), organize them under the `features/` subdirectory as shown above.

Adjust the paths in your scripts (e.g., `src/train/train_cyclegan.py`, `src/transform/feature_transform.py`, `src/eval/eval_triplet_cnn.py`, `src/eval/eval_triplet_cnn.py`) to correctly point to these data directories. 
