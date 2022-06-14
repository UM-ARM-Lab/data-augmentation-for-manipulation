from torch import nn
import torch.nn.functional as F

class Res3D(nn.Module):
    """ simple residual block for convolutions on voxelgrids or SDFs """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding='same')
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding='same')

    def convblock(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

    def forward(self, x):
        z = self.convblock(x)
        return F.relu(x + z)
