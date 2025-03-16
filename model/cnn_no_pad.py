import torch
from torch import nn

class ConvBlockMaxPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.pooling = nn.MaxPool2d(2, 2)
        self.batch_norm = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x) 
        x = self.pooling(x)
        x = self.batch_norm(x) 
        x = self.relu(x) 
        return x
    
class FullyConnectedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_output=False):
        super().__init__()
        self.is_output = is_output
        self.dense = nn.Linear(in_channels, out_channels)
        if not self.is_output:
            self.batch_norm = nn.BatchNorm1d(out_channels, track_running_stats=False)
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dense(x)
        if not self.is_output: 
            x = self.batch_norm(x)
            x = self.relu(x)
        return x
    
class NetNoPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = ConvBlockMaxPool(3, 8)
        self.conv_block2 = ConvBlockMaxPool(8, 16)
        self.conv_block3 = ConvBlockMaxPool(16, 32)
        flatten_channels = 32 * 6 * 6
        self.fc_block1 = FullyConnectedBlock(flatten_channels, flatten_channels // 2)
        self.fc_block2 = FullyConnectedBlock(flatten_channels // 2, flatten_channels // 2**2)
        self.fc_block3 = FullyConnectedBlock(flatten_channels // 2**2, flatten_channels // 2**3)
        self.fc_block4 = FullyConnectedBlock(flatten_channels // 2**3, 10, is_output=True)
        self.conv_blocks = nn.Sequential(
            self.conv_block1, 
            self.conv_block2, 
            self.conv_block3
        )
        self.fc_blocks = nn.Sequential(
            self.fc_block1,
            self.fc_block2,
            self.fc_block3,
            self.fc_block4
        )

    def forward(self, x):
        # x = self.conv_blocks(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        #  flatten all dimensions except batch
        x = torch.flatten(x, 1) 
        x = self.fc_blocks(x)
        return x
    
if __name__ == '__main__':
    # ensuring that ConvBlockMaxPool reduces dimension by half and increases channel
    a = torch.rand((1, 3, 10, 10)) 
    b = ConvBlockMaxPool(3, 8)(a)
    print(b.shape)

    # testing shape of model output (should be (batch,10))
    test_model = NetNoPad()
    test_img = torch.rand((10, 3, 64, 64))
    print(test_model(test_img).shape)