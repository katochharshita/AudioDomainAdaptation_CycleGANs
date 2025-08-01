import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    """
    Enhanced weight initialization for consistent training stability.
    Applies a normal distribution to Conv, BatchNorm, InstanceNorm, and Linear layers' weights.
    Biases are initialized to zero.
    """
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and m.weight is not None:
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

class ResidualBlock(nn.Module):
    """
    A standard Residual Block as used in ResNet-based generators.
    Consists of two convolutional layers with Instance Normalization and ReLU activation,
    with a skip connection adding the input to the output.
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        out = self.block(x)
        # Handle mismatched shapes explicitly if interpolation is needed
        if x.size() != out.size():
            out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False)
        return x + out  # Residual connection

class GeneratorResNet(nn.Module):
    """
    ResNet-based generator for CycleGAN, designed for image-to-image translation.
    Features initial convolution, downsampling, residual blocks, and upsampling.
    """
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]
        out_features = 64

        # Initial convolution block
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

        # Downsampling layers
        self.downsample = nn.Sequential(
            nn.Conv2d(out_features, out_features * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features * 2, out_features * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features * 4),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(out_features * 4) for _ in range(num_residual_blocks)]
        )

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_features * 4, out_features * 2, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_features * 2, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        initial_features = self.initial(x)
        downsampled = self.downsample(initial_features)
        res_output = self.res_blocks(downsampled)
        upsampled_output = self.upsample(res_output)

        # Match dimensions explicitly for residual connection
        if x.size() != upsampled_output.size():
            min_h = min(x.size(2), upsampled_output.size(2))
            min_w = min(x.size(3), upsampled_output.size(3))
            x = x[:, :, :min_h, :min_w]
            upsampled_output = upsampled_output[:, :, :min_h, :min_w]

        return x + upsampled_output  # Residual connection for System C

class SelfAttention(nn.Module):
    """
    Self-attention block for integrating global context into feature maps.
    Calculates attention weights and applies them to the input features.
    """
    def __init__(self, in_channels, attention_scaling_factor=1.0):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scaling_factor = attention_scaling_factor

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        attention = torch.bmm(query, key) / self.scaling_factor
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, channels, height, width)
        return self.gamma * out + x

class Discriminator(nn.Module):
    """
    PatchGAN discriminator for CycleGAN, designed to classify image patches as real or fake.
    This version is a basic discriminator without explicit attention or multi-scale features.
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        # Calculate output shape for the discriminator (1, H', W')
        self.output_shape = (1, max(1, height // 2 ** 4), max(1, width // 2 ** 4))

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)), # Pad to handle input size variations
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, img):
        return self.model(img)
