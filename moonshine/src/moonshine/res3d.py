class ResBlock(nn.Module):
    """ simple residual block for convolutions on voxelgrids or SDFs """

    def __init__(self, in_size: int, hidden_size: int, out_size: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv3d(in_size, hidden_size, kernel_size)

    def convblock(self, x):
        x = F.relu(self.conv(x))
        return x

    def forward(self, x):
        return x + self.convblock(x)
