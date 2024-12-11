import torch
import torch.nn as nn
from unet import Unet3D, EncoderBlock, ConvBnSiLu


class ZeroConvBlock(nn.Module):
    """
    Zero-initialized convolution block to ensure that ControlNet starts with no effect.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.zero_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x):
        return self.zero_conv(x)

class ControlBlock(nn.Module):
    """
    A wrapper for EncoderBlock with support for time embedding.
    """
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.encoder_block = EncoderBlock(in_channels, out_channels, time_embedding_dim)

    def forward(self, x, t):
        return self.encoder_block(x, t)

class ControlNet(nn.Module):
    """
    Copies the encoder blocks from the original UNet and adds zero convolution for control.
    """
    def __init__(self, unet: Unet3D, timesteps, time_embedding_dim):
        super().__init__()
        # Retrieve channels from the UNet structure
        channels = unet._cal_channels(unet.base_dim, unet.dim_mults)
        self.init_zero_conv = ZeroConvBlock(1, 1)
        self.init_conv = ConvBnSiLu(1, unet.base_dim, 3, 1, 1)
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        self.control_blocks = nn.ModuleList()
        for in_channels, out_channels in channels:
            control_block = ControlBlock(in_channels, out_channels, time_embedding_dim)
            self.control_blocks.append(control_block)
        self.zero_convblocks = nn.ModuleList()
        for in_channels, out_channels in channels:
            zero_convblock = nn.Sequential(
                ZeroConvBlock(out_channels//2, out_channels//2)
            )
            self.zero_convblocks.append(zero_convblock)

    def forward(self, x, c, t=None):
        control_features = []

        c_out = self.init_zero_conv(c)

        x = self.init_conv(x + c_out)
        for control_block, zero_convblock in zip(self.control_blocks, self.zero_convblocks):
            x, x_shortcut = control_block(x, t)  # Custom ControlBlock now accepts both x and t
            x_shortcut = zero_convblock(x_shortcut)
            control_features.append(x_shortcut)

        return control_features


class UnetWithControlNet(nn.Module):
    """
    Combines the original UNet and ControlNet.
    """
    def __init__(self, unet: Unet3D,timesteps, time_embedding_dim):
        super().__init__()
        self.unet = unet
        self.control_net = ControlNet(unet,timesteps, time_embedding_dim)

    def forward(self, x, c, t=None):
        # Process input through the ControlNet
        if t is not None:
            t = self.unet.time_embedding(t)
        control_features = self.control_net(x, c, t)

        # Process input through the original UNet encoder
        x = self.unet.init_conv(x)
        encoder_shortcuts = []
        for encoder_block in self.unet.encoder_blocks:
            x, x_shortcut = encoder_block(x, t)
            encoder_shortcuts.append(x_shortcut)

        # Process mid block
        x = self.unet.mid_block(x)

        # Process decoder blocks (merge ControlNet features here)
        encoder_shortcuts.reverse()
        control_features.reverse()
        for decoder_block, shortcut, control_feature in zip(
            self.unet.decoder_blocks, encoder_shortcuts, control_features
        ):
            
            x = decoder_block(x, shortcut + control_feature, t)  # Merge control features in skip connections

        # Final convolution
        x = self.unet.final_conv(x)
        return x


if __name__ == "__main__":
    # Example usage
    x = torch.randn(2, 1, 64, 64, 64)  # Example input (batch, channel, depth, height, width)
    t = torch.randint(0, 1000, (2,))  # Example timesteps
    c = torch.randn(2, 1, 64, 64, 64)
    # Load the original Unet3D
    unet = Unet3D(
        timesteps= 1000,
        time_embedding_dim=256,
        base_dim=32,  # Base dimension for the UNet
        dim_mults=(1, 2, 4, 8),  # Multiplier for channel dimensions
        in_channels=1,
        out_channels=1
    )

    # Wrap it with ControlNet
    model = UnetWithControlNet(unet, timesteps= 1000,
        time_embedding_dim=256,)
    # Logic to freeze or unfreeze parameters
    for name, param in model.named_parameters():
        if "control" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Verify which parameters are trainable
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # Forward pass
    y = model(x, c, t)
    print(f"Output shape: {y.shape}")
