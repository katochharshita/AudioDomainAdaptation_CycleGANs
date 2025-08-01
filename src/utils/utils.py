import random
import torch
from torch.autograd import Variable

class ReplayBuffer:
    """
    A replay buffer to store previously generated samples (e.g., fake images in GANs).
    This helps to stabilize training of the discriminator by providing a mix of real
    and generated samples, preventing it from overfitting to the latest generated outputs.

    Args:
        max_size (int, optional): The maximum number of samples the buffer can hold. Defaults to 50.
    """
    def __init__(self, max_size=50):
        assert max_size > 0, "Max size for replay buffer must be greater than 0."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        """
        Adds new data to the buffer and returns a mix of new and old data.
        If the buffer is not full, new data is added.
        If the buffer is full, there's a 50% chance to replace an old sample with new data,
        otherwise, the new data is returned directly without being added to the buffer.

        Args:
            data (torch.Tensor): The new data (e.g., a batch of fake images) to process.

        Returns:
            torch.Tensor: A batch of samples, comprising a mix of new and old data from the buffer.
        """
        to_return = []
        for element in data.data: # Assuming data is a Variable containing a tensor
            element = torch.unsqueeze(element, 0) # Add a batch dimension if missing
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5: # 50% chance to replace an old sample
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone()) # Return an old sample
                    self.data[i] = element # Replace old sample with new one
                else:
                    to_return.append(element) # Return the new sample without adding it to buffer
        return Variable(torch.cat(to_return)) # Concatenate and return as Variable

class LambdaLR:
    """
    A learning rate scheduler that applies a linear decay to the learning rate
    starting from a specified epoch.

    Args:
        n_epochs (int): Total number of training epochs.
        offset (int): Starting epoch number (e.g., if training resumes from epoch X, offset is X).
        decay_start_epoch (int): The epoch from which learning rate decay begins.
    """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        """
        Calculates the learning rate multiplier for the current epoch.
        The multiplier is 1.0 before `decay_start_epoch` and linearly decays to 0.0.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: The learning rate multiplier for the current epoch.
        """
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch) 