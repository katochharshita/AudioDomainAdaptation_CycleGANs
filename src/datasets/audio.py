import glob
from multiprocessing import Pool
import os
import librosa
import numpy as np
from torch.utils.data import Dataset
import torch

class InMemoryAudioDataset(Dataset):
    """
    A PyTorch Dataset that loads all audio files into memory as log-mel spectrograms
    for fast access during training. This is suitable for smaller to medium-sized datasets.

    Args:
        root (str): Root directory containing 'train', 'val', 'test' subdirectories.
        n_fft (int): Number of FFT points for Mel spectrogram calculation.
        hop_length (int): Hop length for Mel spectrogram calculation.
        power (float): Exponent for the magnitude spectrogram (e.g., 1.0 for amplitude, 2.0 for power).
        mels_samples (int): Number of time steps to crop/pad Mel spectrograms to.
        n_mels (int): Number of Mel bands to generate.
        sr (int, optional): Target sampling rate for audio. Defaults to 22050.
        mode (str, optional): Dataset mode ('train', 'val', 'test'). Defaults to "train".
    """
    def __init__(self, root, n_fft, hop_length, power, mels_samples, n_mels, sr=22050, mode="train"):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.mels_samples = mels_samples
        self.n_mels = n_mels
        self.sr = sr

        print(f"Initializing dataset in {mode} mode...")

        # Discover and load files for domain A and B
        self.files_A = self._discover_files(os.path.join(root, f"{mode}/A"))
        self.files_B = self._discover_files(os.path.join(root, f"{mode}/B"))
        print(f"Discovered {len(self.files_A)} files in A and {len(self.files_B)} files in B.")

        self.data_A = self._load_audio_files(self.files_A, "A")
        self.data_B = self._load_audio_files(self.files_B, "B")
        print("Dataset loaded into memory. Total samples: A = {}, B = {}".format(len(self.data_A), len(self.data_B)))

    @staticmethod
    def _discover_files(directory):
        """Helper to discover files in a directory."""
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            return []
        # Using glob directly as multiprocessing pool for this simple task might be overkill
        # for subdirectories = [directory], pool.map(list_files, subdirectories) is list_files(directory)
        return sorted(glob.glob(directory + "/*.*"))

    def _load_audio_files(self, file_list, label):
        """
        Loads and preprocesses audio files into memory using multiprocessing.
        Each audio file is converted to a log-mel spectrogram.
        """
        print(f"Loading {label} audio files into memory...")

        # Prepare parameters for _load_audio static method
        params = [(file, self.sr, self.n_fft, self.hop_length, self.power, self.n_mels, self.mels_samples)
                  for file in file_list]

        # Use multiprocessing to load audio files for efficiency
        with Pool(processes=os.cpu_count() // 2 or 1) as pool: # Use half of CPU cores or at least 1
            data = pool.map(self._load_audio, params)

        print(f"Completed loading {label} audio files.")
        return data

    @staticmethod
    def _load_audio(file_and_params):
        """
        Static method to load and preprocess a single audio file.
        Converts audio to a log-mel spectrogram and pads/crops it to a fixed size.
        """
        file, sr, n_fft, hop_length, power, n_mels, mels_samples = file_and_params

        audio, _ = librosa.load(file, sr=sr)

        # Convert to melspectrogram
        spec = librosa.feature.melspectrogram(y=audio, sr=sr,
                                          n_fft=n_fft, hop_length=hop_length, power=power,
                                          n_mels=n_mels)
        spec_crop = np.log1p(spec[:, :mels_samples]) # Apply log1p for numerical stability and better distribution

        # Pad to fixed dimensions if smaller than mels_samples
        spec_pad = np.zeros((n_mels, mels_samples)) # Initialize with zeros
        spec_pad[:, :spec_crop.shape[1]] = spec_crop # Place cropped spectrogram

        return spec_pad[np.newaxis, :].astype(np.float32) # Add channel dimension [channel, height, width]

    def __getitem__(self, index):
        """Retrieves a pair of audio spectrograms from domains A and B."""
        # Use modulo to cycle through the smaller dataset if lengths differ
        A = torch.from_numpy(self.data_A[index % len(self.data_A)]).float()
        B = torch.from_numpy(self.data_B[index % len(self.data_B)]).float()

        # Ensure the channel dimension is present (e.g., [1, n_mels, mels_samples])
        if A.dim() == 2:
            A = A.unsqueeze(0)
        if B.dim() == 2:
            B = B.unsqueeze(0)

        return {"A": A, "B": B}

    def __len__(self):
        """Returns the maximum length of the two datasets to ensure full iteration."""
        return max(len(self.data_A), len(self.data_B))


class AudioDataset(Dataset):
    """
    A PyTorch Dataset that loads audio files on-the-fly from disk.
    Converts audio to log-mel spectrograms during __getitem__.
    Suitable for large datasets where preloading into memory is not feasible.

    Args:
        root (str): Root directory containing 'train', 'val', 'test' subdirectories.
        n_fft (int): Number of FFT points for Mel spectrogram calculation.
        hop_length (int): Hop length for Mel spectrogram calculation.
        power (float): Exponent for the magnitude spectrogram (e.g., 1.0 for amplitude, 2.0 for power).
        mels_samples (int): Number of time steps to crop/pad Mel spectrograms to.
        n_mels (int): Number of Mel bands to generate.
        sr (int, optional): Target sampling rate for audio. Defaults to 22050.
        mode (str, optional): Dataset mode ('train', 'val', 'test'). Defaults to "train".
    """
    def __init__(self, root, n_fft, hop_length, power, mels_samples, n_mels, sr=22050, mode="train"):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.mels_samples = mels_samples
        self.n_mels = n_mels
        self.sr = sr

        # Discover files for domain A and B
        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}/A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

    def __getitem__(self, index):
        """Loads audio, converts to log-mel spectrogram, and returns a pair."""
        # Use modulo to cycle through the smaller dataset if lengths differ
        audio_A, _ = librosa.load(self.files_A[index % len(self.files_A)], sr=self.sr)
        audio_B, _ = librosa.load(self.files_B[index % len(self.files_B)], sr=self.sr)

        return {"A": self.to_logspec(audio_A), "B": self.to_logspec(audio_B)}

    def __len__(self):
        """Returns the maximum length of the two datasets to ensure full iteration."""
        return max(len(self.files_A), len(self.files_B))
    
    def to_logspec(self, audio):
        """Convert an audio time series to a cropped/padded log1p melspectrogram."""
        spec = librosa.feature.melspectrogram(y=audio, sr=self.sr,
            n_fft=self.n_fft, hop_length=self.hop_length, power=self.power, 
            n_mels=self.n_mels)
        spec_crop = np.log1p(spec[:, :self.mels_samples])
        spec_pad = np.zeros((self.n_mels, self.mels_samples))
        spec_pad[:, :spec_crop.shape[1]] = spec_crop
        
        return spec_pad[np.newaxis, :] # Add channel dimension [channel, height, width] 