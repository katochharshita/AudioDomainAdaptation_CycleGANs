# CycleGAN Audio Domain Adaptation

This project implements a modular pipeline for audio domain adaptation using CycleGAN, feature transformation, and evaluation with a 1D-Triplet CNN. The code is organized for reproducibility and ease of use.

## Project Structure

```
AudioDomainAdaptation_CycleGANs/
│
├── README.md
├── requirements.txt
│
├── data/                          # Raw and processed audio data (see data/README.md)
│
├── src/
│   ├── models/
│   │   └── cyclegan.py            # CycleGAN model definitions (Generators, Discriminators)
│   │                            # Note: The 1D-Triplet CNN model used for evaluation (eval_triplet_cnn.py)
│   │                            # is externally provided or its checkpoint loaded.
│   │
│   ├── datasets/
│   │   └── audio.py               # Audio dataset loading and preprocessing classes
│   │
│   ├── train/
│   │   └── train_cyclegan.py      # Main script for training the CycleGAN model
│   │
│   ├── transform/
│   │   └── feature_transform.py   # Script for transforming audio features using trained CycleGAN
│   │
│   ├── eval/
│   │   ├── eval_triplet_cnn.py    # Script for 1D-Triplet CNN specific evaluation (EER, TMR),
│   │                            # which relies on an externally provided 1D-Triplet CNN model.
│   │   └── eval_xvector.py        # Script for X-vector speaker verification evaluation (EER, TMR).
│   │
│   └── utils/
│       └── utils.py               # Common utility functions (e.g., ReplayBuffer, LR schedulers)
│
├── scripts/
    ├── train.sh                   # Shell script to run CycleGAN training
    ├── transform.sh               # Shell script to run feature transformation
    └── eval.sh                    # Shell script to run evaluation (general or Triplet CNN)

```

## Workflow

1.  **Data Preparation**: Organize your audio datasets in the `data/` directory as specified in `data/README.md`.
2.  **Model Training**: Train the CycleGAN model using `scripts/train.sh`. This script utilizes `src/train/train_cyclegan.py`.
3.  **Feature Transformation**: Transform test audio features from one domain to another using `scripts/transform.sh`. This script uses `src/transform/feature_transform.py`.
4.  **Evaluation**: Evaluate the transformed features and model performance. You can use `scripts/eval.sh` to run either:
    *   `src/eval/eval_triplet_cnn.py` for 1D-Triplet CNN specific metrics (EER, TMR). This script
        relies on an externally provided 1D-Triplet CNN model (e.g., from [Chowdhury et al., 2019](#reference-chowdhury-2019)).
    *   `src/eval/eval_xvector.py` for X-vector speaker verification evaluation (EER, TMR). This also uses the SpeechBrain toolkit for embedding extraction.

## Setup and Usage

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd AudioDomainAdaptation_CycleGANs
```

### 2. Install Dependencies
It's recommended to use a virtual environment.

Using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data
Follow the instructions in `data/README.md` to set up your datasets.

### 4. Run Scripts
Execute the shell scripts in the `scripts/` directory as needed. Refer to the comments within each `.sh` file for specific usage and available arguments.

*   **Training**: `bash scripts/train.sh`
*   **Transformation**: `bash scripts/transform.sh`
*   **Evaluation**: `bash scripts/eval.sh` (or `python -m src.eval.eval_triplet_cnn` for specific evaluation)

## License

See the `LICENSE` file in the root directory for licensing information.

## References

*   **1D-Triplet CNN Model:**
    Chowdhury, S., Lee, Y., & Lee, S. (2019). "Speaker Verification with 1D-Triplet CNN Embedding for Voice Conversion." *Proc. Interspeech 2019*, 2623-2627. DOI: 10.21437/Interspeech.2019-2708.
    [[PDF]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2708.pdf)
*   **SpeechBrain Toolkit:**
    Ravichander, A., Parcollet, T., Ravanelli, M., & Cornell, S. (2021). SpeechBrain: A General-Purpose Speech Toolkit. *Proc. Interspeech 2021*, 4545-4549. DOI: 10.21437/Interspeech.2021-1250.
    [[GitHub]](https://github.com/speechbrain/speechbrain)
*   **Audio CycleGAN for Singing Voice Conversion:**
    Nidadavolu, P. S., & Lee, Y. (2021). "Audio CycleGAN for Singing Voice Conversion." *arXiv preprint arXiv:2102.04944*.
    [[arXiv]](https://arxiv.org/abs/2102.04944) 
