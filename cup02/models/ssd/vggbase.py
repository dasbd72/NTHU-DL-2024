import torch
import torch.nn as nn

class SSDBackbone(nn.Module):
    def __init__(self):
        super(SSDBackbone, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # 298
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 296
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # 148
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 146
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 144
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # 72
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 70
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # 68
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 66
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # 33
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 31
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 29
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Block 5
        self.block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        
        # Additional SSD layers
        self.extra_layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Forward pass through each block
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Get the output of block 4-3
        block4_3_output = self.block4(x)
        x = self.block5(block4_3_output)
        
        # Continue with additional SSD layers
        x = self.extra_layers(x)
        
        return block4_3_output, x  # Return block5_3 output along with final output