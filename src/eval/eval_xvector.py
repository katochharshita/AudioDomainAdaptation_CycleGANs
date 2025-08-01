"""
This script performs speaker verification evaluation using X-vector embeddings.

It supports evaluating different X-vector models (pretrained, fine-tuned)
on various datasets (original singing, original speaking, transformed audios).
Metrics computed include Equal Error Rate (EER) and True Match Rate (TMR) at 1% False Match Rate (FMR).
"""

import os
import torch
import pandas as pd
import random
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from numpy import interp

# Dynamically determine the project root for robust relative pathing
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming project root is two levels up from src/eval/
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, '../..'))

# === Configuration ===
# Define paths to datasets and model checkpoints relative to PROJECT_ROOT.
# Please ensure these directories and files exist within your project structure
# or are symlinked appropriately for these paths to be valid.

# Data splits: Paths to CSV files containing audio file paths and speaker labels.
# These CSVs are expected to list files from your 'data/ms_dataset2/' and
# 'transformed_audio_output/' directories to align with feature_eval.py's data sources.
data_splits = {
    # Assuming CSVs will be generated/located within the 'data/ms_dataset2/' directory
    "MS_singing": os.path.join(PROJECT_ROOT, "data/ms_dataset/test/A/metadata.csv"),
    "MS_speaking": os.path.join(PROJECT_ROOT, "data/ms_dataset/test/B/metadata.csv"),
    # Transformed audio output directories already align well
    "transformed_MS_singing": os.path.join(PROJECT_ROOT, "transformed_audio_output/A_to_B/wav/metadata.csv"),
    "transformed_MS_speaking": os.path.join(PROJECT_ROOT, "transformed_audio_output/B_to_A/wav/metadata.csv"),
}

# Model paths: Paths to X-vector model checkpoints.
# 'pretrained': Uses the default SpeechBrain pretrained model.
# 'ft_singing' and 'ft_both': Assumes fine-tuned models are saved within the project's 'saved_models/xvector_eval/' directory.
model_paths = {
    "pretrained": None,  # Use default from SpeechBrain
    "ft_singing": os.path.join(PROJECT_ROOT, "saved_models/xvector_eval/fine_tuned_model_triplet_singing/triplet_finetuned_model_singing.pth"),
    "ft_both": os.path.join(PROJECT_ROOT, "saved_models/xvector_eval/fine_tuned_model_triplet_singingandspeaking/triplet_finetuned_model_singingandspeaking.pth"),
}

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Model Loading ===
def load_model(name: str, checkpoint_path: str = None) -> EncoderClassifier:
    """
    Loads an X-vector speaker embedding model from SpeechBrain.

    Args:
        name (str): A descriptive name for the model (used for savedir).
        checkpoint_path (str, optional): Path to a custom model checkpoint (.pth).
                                         If None, a default pretrained model is used.

    Returns:
        EncoderClassifier: The loaded SpeechBrain EncoderClassifier model in evaluation mode.
    """
    # Define savedir for pretrained models within the project structure
    model_savedir = os.path.join(PROJECT_ROOT, f"pretrained_models/speaker_embeddings/{name}")
    os.makedirs(model_savedir, exist_ok=True) # Ensure directory exists

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=model_savedir,
        run_opts={"device": device}
    )

    if checkpoint_path:
        # Load custom checkpoint state dictionary
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Handle cases where state_dict might be nested (e.g., 'embedding_model' key)
        if "embedding_model" in state_dict:
            model.mods.embedding_model.load_state_dict(state_dict["embedding_model"])
        else:
            model.mods.embedding_model.load_state_dict(state_dict)

    model.eval() # Set model to evaluation mode for inference
    return model

# === Embedding Extraction ===
def get_embedding(wav_path: str, model: EncoderClassifier) -> torch.Tensor:
    """
    Extracts an X-vector embedding from a given waveform file.

    Args:
        wav_path (str): Path to the audio (WAV) file.
        model (EncoderClassifier): The loaded SpeechBrain EncoderClassifier model.

    Returns:
        torch.Tensor: The extracted speaker embedding.
    """
    # Load audio, add batch dimension, and move to appropriate device
    signal = model.load_audio(wav_path).unsqueeze(0).to(device)
    with torch.no_grad(): # Disable gradient computation for inference
        emb = model.encode_batch(signal)
    return emb.squeeze(1).squeeze(0) # Remove batch and channel dimensions

# === Trial Construction and Scoring ===
def build_trials(df: pd.DataFrame, n_trials: int = 500) -> list:
    """
    Constructs a list of speaker verification trials (same-speaker and different-speaker pairs).

    Args:
        df (pd.DataFrame): DataFrame containing 'speaker' and 'wav' columns.
        n_trials (int): Number of trials to generate for each type (same/different speaker).

    Returns:
        list: A list of tuples, each containing (wav_path1, wav_path2, label).
              Label is 1 for same-speaker, 0 for different-speaker.
    """
    spk2utts = df.groupby("speaker")["wav"].apply(list).to_dict()
    speakers = list(spk2utts.keys())
    same_trials, diff_trials = [], []

    if len(speakers) < 2:
        print(f"Warning: Not enough speakers ({len(speakers)}) to build meaningful trials.")
        return []

    for _ in range(n_trials):
        # Generate a same-speaker trial
        spk = random.choice(speakers)
        # Ensure the chosen speaker has at least two utterances
        if len(spk2utts[spk]) >= 2:
            pair = random.sample(spk2utts[spk], 2)
            same_trials.append((pair[0], pair[1], 1))

        # Generate a different-speaker trial
        # Ensure selection of two distinct speakers
        if len(speakers) >= 2:
            spk1, spk2 = random.sample(speakers, 2)
            utt1 = random.choice(spk2utts[spk1])
            utt2 = random.choice(spk2utts[spk2])
            diff_trials.append((utt1, utt2, 0))
        else:
            print("Warning: Only one speaker available, cannot generate different-speaker trials.")

    # Combine and return all trials
    return same_trials + diff_trials

def evaluate(trials: list, model: EncoderClassifier) -> tuple[float, float]:
    """
    Evaluates speaker verification performance (EER and TMR@1%FMR) for given trials.

    Args:
        trials (list): List of (wav_path1, wav_path2, label) tuples.
        model (EncoderClassifier): The loaded SpeechBrain EncoderClassifier model.

    Returns:
        tuple[float, float]: A tuple containing EER (%) and TMR@1%FMR (%).
    """
    scores, labels = [], []
    for wav1, wav2, label in trials:
        emb1 = get_embedding(wav1, model)
        emb2 = get_embedding(wav2, model)
        # Compute cosine similarity between embeddings
        score = F.cosine_similarity(emb1, emb2, dim=0).item()
        scores.append(score)
        labels.append(label)

    if not scores or sum(labels) == 0 or sum(labels) == len(labels):
        print("Warning: Insufficient data to compute ROC curve. Skipping EER/TMR.")
        return float('nan'), float('nan')

    # Calculate False Positive Rate (FPR), True Positive Rate (TPR) using ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    # Compute Equal Error Rate (EER)
    # EER is the point where FPR equals FNR (1 - TPR)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100
    except ValueError:
        # Occurs if there is no root in the interval [0,1]
        print("Warning: Could not find EER. ROC curve may not cross the y=1-x line.")
        eer = float('nan')

    # Compute True Match Rate (TMR) at 1% False Match Rate (FMR)
    # Interpolate TPR at a specific FPR (FMR)
    tmr = interp(0.01, fpr, tpr) * 100
    return eer, tmr

# === Main Execution ===
if __name__ == "__main__":
    print("Starting X-vector speaker verification evaluation...")
    # Print table header for results
    print(f"{'Model':<18} | {'Data Split':<25} | {'EER (%)':>10} | {'TMR@1%FMR (%)':>15}")
    print("-" * 75)

    results = {} # Dictionary to store all evaluation results

    for model_name, ckpt in model_paths.items():
        print(f"
    Evaluating Model: {model_name.upper()}")
        model = load_model(model_name, ckpt) # Load the appropriate model

        for data_name, csv_path in data_splits.items():
            print(f"Data Split: {data_name}")
            # Load metadata (audio paths and speaker labels) from CSV
            try:
                df = pd.read_csv(csv_path)
            except FileNotFoundError:
                print(f"Error: CSV file not found at {csv_path}. Skipping this data split.")
                results[(model_name, data_name)] = {"EER": float('nan'), "TMR@1%": float('nan')}
                continue

            # Construct trials (pairs of audio files for comparison)
            trials = build_trials(df, n_trials=500)

            if not trials: # Check if trials were successfully built
                print("No trials could be built for this data split. Skipping evaluation.")
                results[(model_name, data_name)] = {"EER": float('nan'), "TMR@1%": float('nan')}
                continue

            # Perform evaluation and get EER, TMR
            eer, tmr = evaluate(trials, model)
            print(f"EER: {eer:.2f}% | TMR@FMR=1%: {tmr:.2f}%")
            # Store results
            results[(model_name, data_name)] = {"EER": eer, "TMR@1%": tmr}

    # Save all evaluation results to a CSV file
    results_df = pd.DataFrame.from_dict(results, orient='index')
    output_csv_path = os.path.join(PROJECT_ROOT, "evaluation_results/xvector_evaluation_summary.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True) # Ensure output directory exists
    results_df.to_csv(output_csv_path)
    print(f"\nEvaluation complete. Results saved to {output_csv_path}") 