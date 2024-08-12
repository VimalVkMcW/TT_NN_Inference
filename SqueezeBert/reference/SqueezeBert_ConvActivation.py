from torch import nn
import sys
sys.path.append('../')
from utils.activations import ACT2FN

class ConvActivation(nn.Module):
    """
    ConvActivation: Conv, Activation
    """

    def __init__(self, cin, cout, groups, act):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.act = ACT2FN[act]

    def forward(self, x):
        output = self.conv1d(x)
        return self.act(output)