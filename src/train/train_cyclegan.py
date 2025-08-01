"""
This script implements the training loop for the CycleGAN model for audio domain adaptation.
It handles data loading, model initialization, optimization, loss calculation,
and logging of training progress.
"""
import argparse
import datetime
import itertools
import os
import glob # This import is not used.
import sys
import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from src.datasets.audio import InMemoryAudioDataset
from src.models.cyclegan import Discriminator, GeneratorResNet, weights_init_normal 
from src.utils.utils import ReplayBuffer 
from src.utils.utils import LambdaLR 
import pandas as pd
import matplotlib.pyplot as plt

from speechbrain.inference.vocoders import HIFIGAN
import torchaudio

# TensorBoard writer setup
writer = SummaryWriter(log_dir="training_logs/tensorboard")

# --- Hyperparameters ---
# These parameters configure the training process. 

epoch = 0 # Starting epoch, useful for resuming training
n_epochs = 151 # Total number of training epochs
dataset_name = "concat_ds" # Name of the dataset subfolder in 'data/'
checkpoint_interval = 10 # Interval for saving model checkpoints
sample_interval = 10 # Interval for saving generated audio samples
best_g_loss = float("inf") # Variable to track the best generator loss for early stopping

# Dynamically determine the project root for robust relative pathing
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, '../../'))

# Ensure save_dir is relative to the project root
best_model_path = os.path.join(PROJECT_ROOT, f"saved_models/{dataset_name}/best_model")
os.makedirs(best_model_path, exist_ok=True) # Create directory for best models

batch_size = 32
accumulation_steps = 8 # Accumulate gradients over multiple batches to simulate larger batch size
lr = 2e-4 # Learning rate for generators
lr_D = 5e-5 # Learning rate for discriminators
b1 = 0.5 # Adam optimizer beta1
b2 = 0.999 # Adam optimizer beta2
decay_epoch = 20 # Epoch from which learning rate starts linearly decaying

n_cpu = 8 # Number of CPU threads for data loading

# Audio processing parameters
n_fft = 512
hop_length = 64
power = 1.0 # Power for the magnitude spectrogram (1.0 for amplitude, 2.0 for power)
n_mels = 80 # Number of Mel bands
mels_samples = 256 # Fixed length for Mel spectrograms

# CycleGAN-specific loss weights
n_residual_blocks = 12 # Number of residual blocks in the Generator
lambda_cyc = 15.0 # Weight for cycle consistency loss
lambda_id = 1.0 # Weight for identity loss (encourages preserving input characteristics)
CHANNEL_MONO = 1 # Number of audio channels (mono)

# Attention and regularization parameters (used in some discriminator variants)
attention_scaling_factor = 0.5 # Controls attention spread in the self-attention block
discriminator_dropout_rate = 0.3 # Regularization to prevent overfitting in discriminators

# Early stopping parameters
patience = 10 # Number of epochs to wait for improvement before stopping
min_delta = 1e-3 # Minimum change in the monitored quantity to qualify as an improvement
epochs_no_improve = 0 # Counter for epochs without improvement

# Load HiFi-GAN vocoder (16kHz) for Mel-spectrogram to waveform conversion
# Ensure savedir is relative to the project root or configured appropriately
hifi_gan_vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir=os.path.join(PROJECT_ROOT, "pretrained_models/tts-hifigan-libritts-16kHz"), # Use relative path
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"} # Ensure device is set correctly
)

# Create output directories if they don't exist
os.makedirs(os.path.join(PROJECT_ROOT, f"audio_samples/{dataset_name}"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, f"saved_models/{dataset_name}"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, f"training_logs/{dataset_name}"), exist_ok=True)

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Determine device for training
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Initialize generator and discriminator models
input_shape = (CHANNEL_MONO, n_mels, mels_samples)
G_AB = GeneratorResNet(input_shape, n_residual_blocks) # Generator A to B
G_BA = GeneratorResNet(input_shape, n_residual_blocks) # Generator B to A
D_A = Discriminator(input_shape) # Discriminator for domain A
D_B = Discriminator(input_shape) # Discriminator for domain B

# Move models and loss functions to GPU if CUDA is available
if cuda:
    print("Running the code on CUDA (GPU).")
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_cycle = criterion_cycle.cuda()
    criterion_identity = criterion_identity.cuda()

# Load pretrained models or initialize weights
if epoch != 0: # If resuming training from a specific epoch
    print(f"Resuming training from epoch {epoch}. Loading checkpoints...")
    G_AB.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, f"saved_models/{dataset_name}/G_AB_{epoch}.pth"))) # Use relative path
    G_BA.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, f"saved_models/{dataset_name}/G_BA_{epoch}.pth"))) # Use relative path
    D_A.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, f"saved_models/{dataset_name}/D_A_{epoch}.pth"))) # Use relative path
    D_B.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, f"saved_models/{dataset_name}/D_B_{epoch}.pth"))) # Use relative path
else: # Initialize weights for new training run
    print("Initializing model weights normally.")
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers for generators and discriminators
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr_D, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr_D, betas=(b1, b2))

# Learning rate schedulers (Cosine Annealing for smooth decay)
lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=n_epochs)
lr_scheduler_D_A = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D_A, T_max=n_epochs)
lr_scheduler_D_B = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D_B, T_max=n_epochs)

# Data loaders for training and validation datasets
# Adjust dataset path to point to the new 'data' directory relative to project root
dataloader = DataLoader(
    InMemoryAudioDataset(os.path.join(PROJECT_ROOT, "data", dataset_name),
        n_fft=n_fft, hop_length=hop_length, power=power,
        mels_samples=mels_samples, n_mels=n_mels),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
    persistent_workers=False,
    pin_memory=True,
    prefetch_factor=4
)

val_dataloader = DataLoader(
    InMemoryAudioDataset(os.path.join(PROJECT_ROOT, "data", dataset_name),
        n_fft=n_fft, hop_length=hop_length, power=power,
        mels_samples=mels_samples, n_mels=n_mels, mode="val"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

# Replay buffers to store and sample from previously generated images
# This helps stabilize GAN training
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Gradient Scaler for Automatic Mixed Precision (AMP) training
scaler = GradScaler()

def mel_to_waveform_hifigan(mel_spec):
    """
    Converts a Mel-spectrogram batch [B, n_mels, T] to waveform using a pretrained HiFi-GAN vocoder.
    Assumes input is log-Mel (e.g. log(1 + mel)).
    Output: [B, T] waveform (batched)
    """
    with torch.no_grad():
        # Convert log-Mel back to linear scale Mel spectrogram
        mel_linear = torch.expm1(mel_spec).to(hifi_gan_vocoder.device) # Ensure vocoder is on correct device

        if mel_linear.dim() == 4:
            mel_linear = mel_linear.squeeze(1) # [B, 1, M, T] -> [B, M, T]

        # HiFi-GAN expects 80 mel channels
        if mel_linear.shape[1] != 80:
            raise ValueError(f"HiFi-GAN expects 80 mel channels. Got: {mel_linear.shape[1]}")

        # Infer waveform using the vocoder
        waveform = hifi_gan_vocoder.infer(mel_linear) # [B, 1, T]

        return waveform.squeeze(1) # [B, T]

def sample_sounds(batches_done, G_AB_model, G_BA_model, val_loader, dataset_name_str, Tensor_type):
    """
    Generates and saves sample audio files from the validation set during training.
    Converts real and fake Mel-spectrograms to waveforms using HiFi-GAN.
    """
    G_AB_model.eval() # Set generators to evaluation mode
    G_BA_model.eval()

    # Get a batch from the validation dataloader
    # Ensure val_loader provides the correct batch format (e.g., {"A": ..., "B": ...})
    try:
        audios_spec = next(iter(val_loader))
    except StopIteration:
        # Re-initialize iterator if it runs out of samples
        val_loader_iterator = iter(val_loader)
        audios_spec = next(val_loader_iterator)

    # Move real audio spectrograms to the correct device
    real_A = Variable(audios_spec["A"].type(Tensor_type))
    real_B = Variable(audios_spec["B"].type(Tensor_type))

    # Generate fake audio spectrograms
    fake_B = G_AB_model(real_A) # Translate A to B
    fake_A = G_BA_model(real_B) # Translate B to A

    # Convert Mel-spectrograms to waveforms using HiFi-GAN
    wave_real_A = mel_to_waveform_hifigan(real_A)
    wave_fake_B = mel_to_waveform_hifigan(fake_B)
    wave_real_B = mel_to_waveform_hifigan(real_B)
    wave_fake_A = mel_to_waveform_hifigan(fake_A)

    # Define the path to save generated samples relative to project root
    sample_path = os.path.join(PROJECT_ROOT, f"audio_samples/{dataset_name_str}")
    os.makedirs(sample_path, exist_ok=True) # Ensure directory exists

    # Save the generated waveforms as WAV files
    torchaudio.save(f"{sample_path}/real_A_{batches_done}.wav", wave_real_A.cpu(), 16000)
    torchaudio.save(f"{sample_path}/fake_B_{batches_done}.wav", wave_fake_B.cpu(), 16000)
    torchaudio.save(f"{sample_path}/real_B_{batches_done}.wav", wave_real_B.cpu(), 16000)
    torchaudio.save(f"{sample_path}/fake_A_{batches_done}.wav", wave_fake_A.cpu(), 16000)


# Lists to store losses for plotting and CSV export
g_losses, d_losses, cycle_losses, id_losses = [], [], [], []

# --- Training Loop ---
print("Starting training...")
prev_time = time.time()
for epoch in range(epoch, n_epochs):
    epoch_g_loss, epoch_d_loss, epoch_cycle_loss, epoch_identity_loss = [], [], [], []

    for i, batch in enumerate(dataloader):
        # Prepare real audio data for both domains
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Define adversarial "real" and "fake" labels for GAN loss
        # These are tensors filled with ones (for real) or zeros (for fake)
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators (G_AB and G_BA)
        # ------------------
        G_AB.train() # Set generators to training mode
        G_BA.train()

        # Zero gradients for generators before accumulation if starting a new accumulation cycle
        if i % accumulation_steps == 0:
            optimizer_G.zero_grad()

        # Use Automatic Mixed Precision for faster training and reduced memory usage
        with autocast():
            # Generate fake audio spectrograms
            fake_B = G_AB(real_A) # A -> B
            fake_A = G_BA(real_B) # B -> A

            # GAN loss for generators (G wants to fool discriminators)
            # G_AB tries to make fake_B look real to D_B
            pred_fake_B = D_B(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake_B, valid)

            # G_BA tries to make fake_A look real to D_A
            pred_fake_A = D_A(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake_A, valid)

            # Average GAN loss for generators
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle consistency loss (A -> B -> A and B -> A -> B)
            # This ensures that the generated image can be translated back to the original domain
            recov_A = G_BA(fake_B) # A -> B -> A
            loss_cycle_A = criterion_cycle(recov_A, real_A)

            recov_B = G_AB(fake_A) # B -> A -> B
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Identity loss (optional, preserves color/identity of input)
            # G_AB should not change real_B much if it's already in domain B (similarly for G_BA and real_A)
            loss_id_A = criterion_identity(G_BA(real_A), real_A) # G_BA with real_A input
            loss_id_B = criterion_identity(G_AB(real_B), real_B) # G_AB with real_B input

            loss_identity = (loss_id_A + loss_id_B) / 2

            # Total generator loss
            # The adversarial loss is typically weighted (e.g., multiplied by 2)
            # to balance with cycle consistency loss.
            loss_G = 2 * loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
            scaled_loss = loss_G / accumulation_steps # Scale loss for gradient accumulation

        # Backward pass for generators and update weights
        scaler.scale(scaled_loss).backward()

        # Perform optimizer step and scaler update only after accumulation steps
        if (i + 1) % accumulation_steps == 0:
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(G_AB.parameters(), max_norm=5)
            torch.nn.utils.clip_grad_norm_(G_BA.parameters(), max_norm=5)
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad() # Zero gradients after updating

        # -----------------------
        #  Train Discriminators (D_A and D_B)
        # -----------------------
        # Discriminator A training
        optimizer_D_A.zero_grad()
        with autocast():
            # Use a replay buffer for fake samples to stabilize discriminator training
            # This avoids discriminators quickly learning to distinguish specific fakes
            fake_A_ = fake_A_buffer.push_and_pop(fake_A).detach() # Detach to prevent gradients flowing to G

            # Discriminator A loss for real A samples
            pred_real_A = D_A(real_A)
            loss_real_A = criterion_GAN(pred_real_A, valid)

            # Discriminator A loss for fake A samples
            pred_fake_A = D_A(fake_A_)
            loss_fake_A = criterion_GAN(pred_fake_A, fake)

            # Total discriminator A loss
            loss_D_A = (loss_real_A + loss_fake_A) / 2

        # Backward pass and update for Discriminator A
        scaler.scale(loss_D_A).backward()
        scaler.step(optimizer_D_A)
        scaler.update()

        # Discriminator B training
        optimizer_D_B.zero_grad()
        with autocast():
            # Use a replay buffer for fake samples
            fake_B_ = fake_B_buffer.push_and_pop(fake_B).detach() # Detach to prevent gradients flowing to G

            # Discriminator B loss for real B samples
            pred_real_B = D_B(real_B)
            loss_real_B = criterion_GAN(pred_real_B, valid)

            # Discriminator B loss for fake B samples
            pred_fake_B = D_B(fake_B_)
            loss_fake_B = criterion_GAN(pred_fake_B, fake)

            # Total discriminator B loss
            loss_D_B = (loss_real_B + loss_fake_B) / 2

        # Backward pass and update for Discriminator B
        scaler.scale(loss_D_B).backward()
        scaler.step(optimizer_D_B)
        scaler.update()

        # Combined discriminator loss for logging
        loss_D = (loss_D_A + loss_D_B) / 2

        # Store the current batch losses for averaging per epoch
        epoch_g_loss.append(loss_G.item())
        epoch_d_loss.append(loss_D.item())
        epoch_cycle_loss.append(loss_cycle.item())
        epoch_identity_loss.append(loss_identity.item())

        # --------------
        #  Log Progress (per batch, optional)
        # --------------
        batches_done = epoch * len(dataloader) + i
        batches_left = n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print progress to console (can be set to a lower frequency if too verbose)
        if i % 10 == 0: # Log every 10 batches
             sys.stdout.write(
                 "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f, adv: %.4f, cycle: %.4f, identity: %.4f] ETA: %s"
                 % (
                     epoch,
                     n_epochs,
                     i,
                     len(dataloader),
                     loss_D.item(),
                     loss_G.item(),
                     loss_GAN.item(),
                     loss_cycle.item(),
                     loss_identity.item(),
                     time_left,
                 )
             )
        # Log to TensorBoard (per batch)
        writer.add_scalar("Batch_Loss/Generator", loss_G.item(), batches_done)
        writer.add_scalar("Batch_Loss/Discriminator", loss_D.item(), batches_done)
        writer.add_scalar("Batch_Loss/Cycle", loss_cycle.item(), batches_done)
        writer.add_scalar("Batch_Loss/Identity", loss_identity.item(), batches_done)
        writer.add_scalars("Batch_Loss/GAN_Components", {'GAN_AB': loss_GAN_AB.item(), 'GAN_BA': loss_GAN_BA.item()}, batches_done)


    # Handle remaining accumulated gradients if the last batch wasn't a multiple of accumulation_steps
    if (i + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(G_AB.parameters(), max_norm=5)
        torch.nn.utils.clip_grad_norm_(G_BA.parameters(), max_norm=5)
        scaler.step(optimizer_G)
        scaler.update()
        optimizer_G.zero_grad() # Zero gradients after updating

    # Generate and save audio samples at the end of each epoch or at sample_interval
    if epoch % sample_interval == 0:
        print(f"\nGenerating audio samples for epoch {epoch}...")
        sample_sounds(epoch, G_AB, G_BA, val_dataloader, dataset_name, Tensor)

    # Compute mean losses for the epoch
    mean_g_loss = np.mean(epoch_g_loss)
    mean_d_loss = np.mean(epoch_d_loss)
    mean_cycle_loss = np.mean(epoch_cycle_loss)
    mean_identity_loss = np.mean(epoch_identity_loss)

    # Early Stopping Logic
    # Check if the generator loss has improved significantly
    if mean_g_loss < best_g_loss - min_delta:
        best_g_loss = mean_g_loss
        epochs_no_improve = 0 # Reset patience counter

        # Save the best performing models (generators and discriminators)
        print(f"Saving new best model at epoch {epoch} (G Loss: {mean_g_loss:.4f})")
        torch.save(G_AB.state_dict(), os.path.join(best_model_path, "G_AB_best.pth")) # Use relative path
        torch.save(D_A.state_dict(), os.path.join(best_model_path, "D_A_best.pth")) # Save D_A if needed for consistent checkpointing, use relative path
        torch.save(G_BA.state_dict(), os.path.join(best_model_path, "G_BA_best.pth")) # Use relative path
        torch.save(D_B.state_dict(), os.path.join(best_model_path, "D_B_best.pth")) # Save D_B if needed, use relative path
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s). Best G Loss so far: {best_g_loss:.4f}")

    # Stop training if no improvement for `patience` epochs
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch}. No improvement for {patience} consecutive epochs.")
        break # Exit the training loop

    # Log mean losses for TensorBoard (per epoch)
    writer.add_scalar("Epoch_Loss/Generator (mean)", mean_g_loss, epoch)
    writer.add_scalar("Epoch_Loss/Discriminator (mean)", mean_d_loss, epoch)
    writer.add_scalar("Epoch_Loss/Cycle (mean)", mean_cycle_loss, epoch)
    writer.add_scalar("Epoch_Loss/Identity (mean)", mean_identity_loss, epoch)

    # Append mean losses to lists for CSV logging later
    g_losses.append(mean_g_loss)
    d_losses.append(mean_d_loss)
    cycle_losses.append(mean_cycle_loss)
    id_losses.append(mean_identity_loss)

    # Print mean losses for the epoch to console
    print(f"\n[Epoch {epoch}/{n_epochs}] Mean Losses - G loss: {mean_g_loss:.4f}, D loss: {mean_d_loss:.4f}, "
        f"Cycle loss: {mean_cycle_loss:.4f}, Identity loss: {mean_identity_loss:.4f}")

    # Clear the lists for the next epoch's accumulation
    epoch_g_loss.clear()
    epoch_d_loss.clear()
    epoch_cycle_loss.clear()
    epoch_identity_loss.clear()

    # Update learning rates for the next epoch
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save epoch checkpoints at specified intervals
    # Note: Only save G_AB and G_BA for primary use, D_A/D_B can be optional if not resuming D.
    if checkpoint_interval != -1 and (epoch + 1) % checkpoint_interval == 0: # (epoch + 1) to save at epoch 10, 20 etc.
        print(f"Saving epoch {epoch} checkpoints...")
        torch.save(G_AB.state_dict(), os.path.join(PROJECT_ROOT, f"saved_models/{dataset_name}/G_AB_{epoch}.pth")) # Use relative path
        torch.save(G_BA.state_dict(), os.path.join(PROJECT_ROOT, f"saved_models/{dataset_name}/G_BA_{epoch}.pth")) # Use relative path
        torch.save(D_A.state_dict(), os.path.join(PROJECT_ROOT, f"saved_models/{dataset_name}/D_A_{epoch}.pth")) # Use relative path
        torch.save(D_B.state_dict(), os.path.join(PROJECT_ROOT, f"saved_models/{dataset_name}/D_B_{epoch}.pth")) # Use relative path
        # To manage disk space, one might add logic here to remove older checkpoints.

# --- Training Completion & Final Logging ---
print("\nTraining finished.")

# Ensure the training log directory exists for final output files
final_log_dir = os.path.join(PROJECT_ROOT, f"training_logs/{dataset_name}") # Use relative path
os.makedirs(final_log_dir, exist_ok=True)

# Save training losses to a CSV file
loss_df = pd.DataFrame({
    "Epoch": list(range(len(g_losses))),
    "G Loss": g_losses,
    "D Loss": d_losses,
    "Cycle Loss": cycle_losses,
    "Identity Loss": id_losses
})
loss_df.to_csv(os.path.join(final_log_dir, f"losses_{dataset_name}.csv"), index=False) # Use relative path
print(f"Training losses saved to {os.path.join(final_log_dir, f'losses_{dataset_name}.csv')}")

# Plot losses and save the figure
plt.figure(figsize=(12, 6))
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(cycle_losses, label="Cycle Loss")
plt.plot(id_losses, label="Identity Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"CycleGAN Training Losses - Dataset: {dataset_name}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(final_log_dir, f"loss_plot_{dataset_name}.png")) # Use relative path
plt.show()
print(f"Training loss plot saved to {os.path.join(final_log_dir, f'loss_plot_{dataset_name}.png')}")

# Close TensorBoard writer
writer.close()
print("TensorBoard writer closed.") 