"""
Evaluates speaker embeddings using a 1D-Triplet CNN model.

This script loads a pre-trained 1D-Triplet CNN, extracts embeddings from specified
feature sets (e.g., original singing, transformed speaking), and then computes
speaker verification metrics such as Equal Error Rate (EER) and True Match Rate
@ 1% False Match Rate (TMR@1%FMR).

It is designed to assess the quality of speaker information preserved or adapted
within the transformed audio features.

Dependencies:
    - numpy
    - hdf5storage (for .mat file loading)
    - sklearn (for roc_curve)
    - torch
    - src.models.triplet_cnn (for the CNN model)
"""
import os
import glob
import sys
import numpy as np
import hdf5storage
from sklearn.metrics import roc_curve
import torch
import torch.nn.functional as F

# Adjust import path for the 1D-Triplet CNN model
from src.models.triplet_cnn import cnn

# --- Configuration: Model Checkpoints & Test-set Folders ---
# These paths should point to your actual model checkpoints and feature directories.
# It's recommended to place trained models within the project's 'saved_models/' directory
# and processed features in a structured 'data/features/' or 'output/' directory.

# Dynamically determine the project root for robust relative pathing
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming project root is two levels up from src/eval/
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, '../..' ))

# Example model checkpoint paths (update to your actual saved models within the project)
# You might want to copy your .pth.tar models into a directory like 'saved_models/triplet_cnn/'
experiments = {
    'Baseline':       os.path.join(PROJECT_ROOT, 'saved_models/triplet_cnn/oned_triplet_cnn.pth.tar'),
    'Domain-Adapted': os.path.join(PROJECT_ROOT, 'saved_models/triplet_cnn/domain_adapted_final_best.pth.tar'),
    'Fine-tuned':     os.path.join(PROJECT_ROOT, 'saved_models/triplet_cnn/finetuned_singing.pth.tar'),
}

# Example test set feature folders (update to your actual feature locations within the project)
# These should ideally point to .mat files containing features (e.g., MFCCs, Fbanks).
# You might need to copy your .mat feature files into a directory like 'data/features/'
test_sets = {
    "MS (Singing)": os.path.join(PROJECT_ROOT, "data/features/MS_singing"),
    "MS (Spoken)": os.path.join(PROJECT_ROOT, "data/features/MS_speaking"),
    "MS_transformed (Singing)": os.path.join(PROJECT_ROOT, "data/features/MS_singing_transformed"),
    "MS_transformed (Spoken)": os.path.join(PROJECT_ROOT, "data/features/MS_speaking_transformed"),
}

# --- Helper Functions ---

def load_model(ckpt_path):
    """
    Loads a 1D-Triplet CNN model from a given checkpoint path.

    Args:
        ckpt_path (str): Path to the model checkpoint file (.pth.tar).

    Returns:
        torch.nn.Module: Loaded CNN model in evaluation mode, moved to CUDA if available.
    """
    print(f"Loading model from: {ckpt_path}")
    # Initialize the CNN model and move to GPU
    model = cnn().cuda().eval()
    # Load the state dictionary from the checkpoint
    state_dict = torch.load(ckpt_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['state_dict']

    # Handle DataParallel prefix if the model was saved with nn.DataParallel
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load the cleaned state dictionary into the model
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
    return model

def extract_embeddings(model, feature_root):
    """
    Extracts speaker embeddings and corresponding labels from a feature folder.

    Assumes features are stored as .mat files within speaker-specific subfolders.
    Each .mat file is expected to contain a 'data' key with the features.

    Args:
        model (torch.nn.Module): The embedding extraction model (1D-Triplet CNN).
        feature_root (str): Root directory containing speaker subfolders with .mat features.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Stacked embeddings for all processed audio samples.
            - numpy.ndarray: Corresponding speaker labels for each embedding.
    """
    print(f"Extracting embeddings from: {feature_root}")
    embs, labels = [], []
    if not os.path.exists(feature_root):
        print(f"Warning: Feature root directory not found: {feature_root}")
        return np.array([]), np.array([])

    # Iterate through each speaker subdirectory
    for spk in sorted(os.listdir(feature_root)):
        spk_folder = os.path.join(feature_root, spk)
        if not os.path.isdir(spk_folder): # Skip if not a directory
            continue

        # Process each .mat file in the speaker's folder
        for matf in glob.glob(os.path.join(spk_folder, '*.mat')):
            try:
                data = hdf5storage.loadmat(matf)['data']
                # Prepare data for model: add batch dim, ensure float32, move to CUDA
                # Original data shape is expected to be [features, time_steps, channels]
                # Transpose to [channels, features, time_steps] for CNN input, then add batch dim [1, C, F, T]
                x = np.expand_dims(data.transpose(2,0,1), 0).astype(np.float32)
                
                with torch.no_grad(): # Disable gradient calculation for inference
                    # Pass through the model to get embeddings
                    emb = model(torch.from_numpy(x).cuda()).cpu().numpy().ravel()
                embs.append(emb)
                labels.append(spk) # Assign speaker label
            except Exception as e:
                print(f"Error processing {matf}: {e}")
                continue
    print(f"Extracted {len(embs)} embeddings.")
    return np.vstack(embs), np.array(labels)

def compute_metrics(embs, labels):
    """
    Computes Equal Error Rate (EER) and True Match Rate (TMR) at 1% False Match Rate (FMR).

    Args:
        embs (np.ndarray): Stacked speaker embeddings.
        labels (np.ndarray): Corresponding speaker labels.

    Returns:
        tuple: EER (%) and TMR@1%FMR (%). Returns 0.0 for TMR if no valid FMR point.
    """
    n = len(embs)
    if n < 2: # Need at least two embeddings to form pairs
        print("Not enough embeddings to compute metrics.")
        return None, None

    scores, gts = [], []
    # Generate all unique pairs of embeddings to compute similarity scores
    for i in range(n):
        for j in range(i + 1, n):
            # Compute cosine similarity between embedding pairs
            s = F.cosine_similarity(
                  torch.tensor(embs[i]).unsqueeze(0).cuda(), # Ensure tensors are on CUDA
                  torch.tensor(embs[j]).unsqueeze(0).cuda()
                ).item()
            scores.append(s)
            # Ground truth: 1 if same speaker, 0 if different speaker
            gts.append(int(labels[i] == labels[j]))

    scores = np.array(scores)
    gts = np.array(gts)

    # Calculate False Positive Rate (FPR), True Positive Rate (TPR) using ROC curve
    fpr, tpr, thr = roc_curve(gts, scores, pos_label=1)
    fnr = 1 - tpr # False Negative Rate

    # EER: find the threshold where FPR is approximately equal to FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2

    # TMR @ FMR=1%: find the highest TPR when FPR is less than or equal to 1%
    valid_fmr_idx = np.where(fpr <= 0.01)[0]
    tmr1 = tpr[valid_fmr_idx[-1]] if len(valid_fmr_idx) > 0 else 0.0

    return eer * 100, tmr1 * 100 # Return as percentage

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting 1D-Triplet CNN evaluation...")
    # Print table header
    print(f"{'Model':<14} | {'Test Set':<20} | {'EER (%)':>7} | {'TMR@1%FMR (%)':>14}")
    print("-"*60)

    # Iterate through each defined experiment (model checkpoint)
    for model_name, checkpoint_path in experiments.items():
        model = load_model(checkpoint_path)
        # Evaluate the model on each specified test set
        for test_set_name, test_set_folder in test_sets.items():
            embs, labels = extract_embeddings(model, test_set_folder)
            if embs.size > 0:
                eer, tmr1 = compute_metrics(embs, labels)
                if eer is not None and tmr1 is not None:
                    print(f"{model_name:<14} | {test_set_name:<20} | {eer:7.2f} | {tmr1:14.2f}")
                else:
                    print(f"{model_name:<14} | {test_set_name:<20} | {'N/A':>7} | {'N/A':>14} (Metrics not computable)")
            else:
                print(f"{model_name:<14} | {test_set_name:<20} | {'N/A':>7} | {'N/A':>14} (No embeddings extracted)")
    print("\n1D-Triplet CNN evaluation complete.") 