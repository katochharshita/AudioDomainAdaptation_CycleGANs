"""
This script provides functionalities to transform audio features using a trained CycleGAN generator.
It loads pre-trained generator models, extracts Mel spectrograms from WAV files,
transforms these features using the generator, and then saves the transformed features
as .npy files and reconstructs them back into WAV audio using a HiFi-GAN vocoder.

Usage:
    This script is intended to be run from the command line, accepting arguments for
    model paths, input directories, and output directories.

Example (after setting up CLI argument parsing):
    python -m src.transform.feature_transform \
        --generator_A_to_B_path /path/to/G_AB_best.pth \
        --generator_B_to_A_path /path/to/G_BA_best.pth \
        --test_dir_A /path/to/singing_data \
        --test_dir_B /path/to/speaking_data \
        --output_dir /path/to/transformed_output

Dependencies:
    - torch
    - torchaudio
    - librosa
    - numpy
    - soundfile
    - tqdm
    - speechbrain (for HIFIGAN vocoder)
    - src.models.cyclegan (for GeneratorResNet)
"""

import os
import librosa
import torch
import numpy as np
import torchaudio
import soundfile as sf
from tqdm import tqdm
import argparse

# Assuming GeneratorResNet is in src/models/cyclegan.py
from src.models.cyclegan import GeneratorResNet

# SpeechBrain HIFIGAN vocoder for waveform synthesis
from speechbrain.inference.vocoders import HIFIGAN

# Determine the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize HiFi-GAN vocoder globally to avoid re-loading for each call
hifigan_vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir="pretrained_models/tts-hifigan-libritts-16kHz", # Ensure this path is correct relative to execution
    run_opts={"device": device}
)

def load_generator(model_path, input_shape=(1, 80, 256), num_residual_blocks=12):
    """
    Loads a pre-trained CycleGAN Generator model from the specified path.

    Args:
        model_path (str): Path to the saved generator model (.pth file).
        input_shape (tuple): Expected input shape for the generator (channels, mels, time_steps).
        num_residual_blocks (int): Number of residual blocks used in the generator architecture.

    Returns:
        torch.nn.Module: Loaded generator model in evaluation mode on the correct device.
    """
    print(f"Loading generator from: {model_path}")
    generator = GeneratorResNet(input_shape, num_residual_blocks).to(device)
    state_dict = torch.load(model_path, map_location=device)

    # Handle DataParallel saved models (remove 'module.' prefix if present)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    generator.load_state_dict(state_dict)
    generator.eval() # Set to evaluation mode for inference
    print("Generator loaded successfully.")
    return generator

def load_wav_and_extract_mel(wav_dir, sr=16000, n_mels=80, n_fft=512, hop_length=64):
    """
    Loads WAV audio files from a directory, resamples them if necessary,
    and extracts log-Mel spectrogram features.

    Args:
        wav_dir (str): Directory containing WAV files.
        sr (int): Target sampling rate.
        n_mels (int): Number of Mel bands.
        n_fft (int): Number of FFT points.
        hop_length (int): Hop length.

    Returns:
        tuple: A tuple containing:
            - list: List of extracted log-Mel spectrograms (numpy arrays).
            - list: List of original WAV file paths.
    """
    print(f"Loading WAV files and extracting Mel spectrograms from {wav_dir}...")
    wav_files = sorted([os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')])
    features = []

    # Define MelSpectrogram transform once and move to GPU for efficiency
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=1.0 # Use 1.0 for amplitude spectrogram, consistent with CycleGAN training
    ).to(device)

    for path in tqdm(wav_files, desc=f"Extracting Mels from {os.path.basename(wav_dir)}"):
        try:
            waveform, orig_sr = torchaudio.load(path, normalize=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        # Ensure waveform is on the correct device before resampling/Mel extraction
        waveform = waveform.to(device)

        # Resample if original sample rate differs from target sample rate
        if orig_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_sr, sr).to(device) # Resampler also on device
            waveform = resampler(waveform)

        # Convert stereo to mono by averaging if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extract Mel spectrogram and apply log1p for stability
        mel = mel_transform(waveform) # [1, n_mels, T]
        log_mel = torch.log1p(mel) # log(1 + mel) for numerical stability

        # Check for NaNs or Infs in the Mel-spectrogram before appending
        if not torch.isfinite(log_mel).all():
            print(f"Warning: Non-finite values detected in {path}. Skipping this file.")
            continue

        features.append(log_mel.squeeze(0).cpu().numpy()) # Remove channel dim and move to CPU as NumPy array

    print(f"Extracted {len(features)} Mel spectrograms from {wav_dir}")
    return features, wav_files

def transform_features_on_gpu(generator, features, batch_size=16):
    """
    Transforms a list of log-Mel spectrograms using the provided generator model on GPU.
    Handles padding within batches to ensure consistent input dimensions for the model.

    Args:
        generator (torch.nn.Module): The trained CycleGAN generator model.
        features (list): List of input log-Mel spectrograms (NumPy arrays).
        batch_size (int): Batch size for processing features on GPU.

    Returns:
        list: List of transformed log-Mel spectrograms (NumPy arrays).
    """
    generator.eval() # Ensure the generator is in evaluation mode
    transformed_features = []

    with torch.no_grad(): # Disable gradient calculation for inference
        for i in tqdm(range(0, len(features), batch_size), desc="Transforming features"):
            batch = features[i:i + batch_size]

            # Determine the maximum time length in the current batch for padding
            max_len = max(feat.shape[-1] for feat in batch)

            padded_batch = []
            for feat in batch:
                # Ensure feature is 2D (n_mels, time_steps) before padding
                if feat.ndim == 3: # If somehow [1, n_mels, T], squeeze
                    feat = feat.squeeze(0)
                if feat.ndim != 2: # Should be (n_mels, T)
                     raise ValueError(f"Unexpected feature dimension: {feat.shape}. Expected 2D or 3D.")

                # Pad the time dimension (last dimension) to max_len
                padded_feat = np.pad(feat, ((0, 0), (0, max_len - feat.shape[-1])), mode='constant')
                padded_batch.append(padded_feat)

            # Stack the padded batch into a single NumPy array and convert to PyTorch tensor
            # Add a channel dimension: [B, 1, n_mels, T]
            input_tensors = torch.from_numpy(np.stack(padded_batch)).unsqueeze(1).float().to(device)

            # Forward pass through the generator and clamp output to a reasonable range
            # Squeeze removes the channel dimension, then detach and move to CPU as NumPy array
            output_tensors = torch.clamp(generator(input_tensors), min=-10, max=10).squeeze().detach().cpu().numpy()
            transformed_features.extend(output_tensors)

    return transformed_features

def features_to_waveform_hifigan(mel_spectrogram_batch, sr=16000):
    """
    Converts a batch of log-Mel spectrograms to waveforms using the global HiFi-GAN vocoder.

    Args:
        mel_spectrogram_batch (np.ndarray or torch.Tensor): Batch of log-Mel spectrograms.
        sr (int): Target sample rate for the output waveform.

    Returns:
        torch.Tensor: Generated waveforms (batch_size, num_samples).
    """
    if isinstance(mel_spectrogram_batch, np.ndarray):
        mel_spectrogram_batch = torch.from_numpy(mel_spectrogram_batch).float()

    # Ensure linear scale for HiFi-GAN (reverse log1p)
    mel_linear = torch.expm1(mel_spectrogram_batch).to(hifigan_vocoder.device)

    # Ensure correct dimensions: [B, n_mels, T]
    if mel_linear.ndim == 2: # If a single (n_mels, T) array, add batch dim
        mel_linear = mel_linear.unsqueeze(0)
    elif mel_linear.ndim == 4: # If [B, 1, n_mels, T], squeeze channel dim
        mel_linear = mel_linear.squeeze(1)

    # HiFi-GAN expects 80 mel channels
    if mel_linear.shape[1] != 80:
        raise ValueError(f"HiFi-GAN expects 80 mel channels. Got: {mel_linear.shape[1]}")

    with torch.no_grad():
        waveform = hifigan_vocoder.infer(mel_linear) # [B, 1, T] or [1, T]

    return waveform.squeeze(1).cpu() # Remove channel dim and move to CPU

def save_transformed_features_and_audio(
    transformed_features,
    original_files,
    wav_output_dir,
    npy_output_dir,
    sr=16000
):
    """
    Saves transformed features as .npy files and corresponding waveforms as .wav files.

    Args:
        transformed_features (list): List of transformed log-Mel spectrograms (NumPy arrays).
        original_files (list): List of original file paths (used for naming output files).
        wav_output_dir (str): Directory to save generated WAV files.
        npy_output_dir (str): Directory to save transformed .npy feature files.
        sr (int): Sample rate for saving WAV files.
    """
    os.makedirs(wav_output_dir, exist_ok=True)
    os.makedirs(npy_output_dir, exist_ok=True)
    print(f"Saving transformed outputs to WAV: {wav_output_dir}, NPY: {npy_output_dir}")

    for i, feature in enumerate(tqdm(transformed_features, desc="Saving outputs")):
        original_file = original_files[i]
        base_filename = os.path.splitext(os.path.basename(original_file))[0]

        # Save .npy feature
        npy_output_file = os.path.join(npy_output_dir, f'{base_filename}_transformed.npy')
        np.save(npy_output_file, feature)

        # Convert feature to waveform and save as WAV
        waveform_tensor = features_to_waveform_hifigan(feature, sr=sr)
        wav_output_file = os.path.join(wav_output_dir, f'{base_filename}_transformed.wav')
        torchaudio.save(wav_output_file, waveform_tensor.unsqueeze(0), sr) # Add channel dim for torchaudio.save


def process_transformation(
    generator_A_to_B,
    generator_B_to_A,
    test_dir_A,
    test_dir_B,
    output_root_dir,
    batch_size=16,
    sr=16000,
    n_mels=80,
    n_fft=512,
    hop_length=64,
    num_residual_blocks=12 # Parameter for loading generator
):
    """
    Main function to orchestrate the audio feature transformation process.
    Handles both A-to-B and B-to-A transformations.

    Args:
        generator_A_to_B (torch.nn.Module): Trained generator for A to B transformation.
        generator_B_to_A (torch.nn.Module): Trained generator for B to A transformation.
        test_dir_A (str): Directory containing test audio files for domain A.
        test_dir_B (str): Directory containing test audio files for domain B.
        output_root_dir (str): Root directory to save all transformed outputs.
        batch_size (int): Batch size for GPU transformation.
        sr (int): Sample rate for audio processing.
        n_mels (int): Number of Mel bands for spectrograms.
        n_fft (int): N_FFT for spectrograms.
        hop_length (int): Hop length for spectrograms.
        num_residual_blocks (int): Number of residual blocks in the generator.
    """
    os.makedirs(output_root_dir, exist_ok=True)

    # Process A-to-B (e.g., Singing to Speaking)
    print("\n--- Processing A-to-B transformations (e.g., Singing to Speaking) ---")
    output_A_to_B_wav = os.path.join(output_root_dir, 'A_to_B', 'wav')
    output_A_to_B_npy = os.path.join(output_root_dir, 'A_to_B', 'npy')
    
    features_A, original_files_A = load_wav_and_extract_mel(test_dir_A, sr, n_mels, n_fft, hop_length)
    if features_A:
        transformed_features_A_to_B = transform_features_on_gpu(generator_A_to_B, features_A, batch_size)
        save_transformed_features_and_audio(
            transformed_features_A_to_B,
            original_files_A,
            output_A_to_B_wav,
            output_A_to_B_npy,
            sr=sr
        )
    else:
        print(f"No valid audio files found in {test_dir_A} for A-to-B transformation.")

    # Process B-to-A (e.g., Speaking to Singing)
    print("\n--- Processing B-to-A transformations (e.g., Speaking to Singing) ---")
    output_B_to_A_wav = os.path.join(output_root_dir, 'B_to_A', 'wav')
    output_B_to_A_npy = os.path.join(output_root_dir, 'B_to_A', 'npy')

    features_B, original_files_B = load_wav_and_extract_mel(test_dir_B, sr, n_mels, n_fft, hop_length)
    if features_B:
        transformed_features_B_to_A = transform_features_on_gpu(generator_B_to_A, features_B, batch_size)
        save_transformed_features_and_audio(
            transformed_features_B_to_A,
            original_files_B,
            output_B_to_A_wav,
            output_B_to_A_npy,
            sr=sr
        )
    else:
        print(f"No valid audio files found in {test_dir_B} for B-to-A transformation.")

# Main execution block for command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform audio features using a trained CycleGAN generator.")
    parser.add_argument(
        "--generator_A_to_B_path",
        type=str,
        default="saved_models/concat_ms/best_model/G_AB_best.pth", # Default relative path
        help="Path to the trained A-to-B generator model (.pth)."
    )
    parser.add_argument(
        "--generator_B_to_A_path",
        type=str,
        default="saved_models/concat_ms/best_model/G_BA_best.pth", # Default relative path
        help="Path to the trained B-to-A generator model (.pth)."
    )
    parser.add_argument(
        "--test_dir_A",
        type=str,
        default="data/test/A", # Default relative path
        help="Directory containing test audio files for domain A (e.g., singing)."
    )
    parser.add_argument(
        "--test_dir_B",
        type=str,
        default="data/test/B", # Default relative path
        help="Directory containing test audio files for domain B (e.g., speaking)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="transformed_audio_output", # Default relative path
        help="Root directory to save transformed audio and features."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing features on GPU."
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate for audio processing and saving."
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="Number of Mel bands for spectrogram extraction."
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=512,
        help="Number of FFT points for spectrogram extraction."
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=64,
        help="Hop length for spectrogram extraction."
    )
    parser.add_argument(
        "--num_residual_blocks",
        type=int,
        default=12, # Consistent with training config
        help="Number of residual blocks in the generator network."
    )

    args = parser.parse_args()

    # Resolve paths relative to the script's location if they are not absolute
    # This assumes the script is run from the project root or similar consistent setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..' )) # Adjust as per your actual project structure

    # Adjust paths to be relative to the inferred project root
    # or ensure they are absolute if user provides them that way
    gen_A_to_B_path_abs = os.path.join(project_root, args.generator_A_to_B_path) if not os.path.isabs(args.generator_A_to_B_path) else args.generator_A_to_B_path
    gen_B_to_A_path_abs = os.path.join(project_root, args.generator_B_to_A_path) if not os.path.isabs(args.generator_B_to_A_path) else args.generator_B_to_A_path
    test_dir_A_abs = os.path.join(project_root, args.test_dir_A) if not os.path.isabs(args.test_dir_A) else args.test_dir_A
    test_dir_B_abs = os.path.join(project_root, args.test_dir_B) if not os.path.isabs(args.test_dir_B) else args.test_dir_B
    output_dir_abs = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    # Load generators
    generator_A_to_B = load_generator(
        gen_A_to_B_path_abs,
        input_shape=(1, args.n_mels, args.mels_samples), # Use mels_samples for consistent input shape
        num_residual_blocks=args.num_residual_blocks
    )
    generator_B_to_A = load_generator(
        gen_B_to_A_path_abs,
        input_shape=(1, args.n_mels, args.mels_samples),
        num_residual_blocks=args.num_residual_blocks
    )

    # Execute the transformation process
    process_transformation(
        generator_A_to_B=generator_A_to_B,
        generator_B_to_A=generator_B_to_A,
        test_dir_A=test_dir_A_abs,
        test_dir_B=test_dir_B_abs,
        output_root_dir=output_dir_abs,
        batch_size=args.batch_size,
        sr=args.sr,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        num_residual_blocks=args.num_residual_blocks
    )

    print("\nFeature transformation complete.") 