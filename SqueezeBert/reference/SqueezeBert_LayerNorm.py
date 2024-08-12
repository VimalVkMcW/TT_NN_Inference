from torch import nn

class SqueezeBertLayerNorm(nn.LayerNorm):
    """
    This is a nn.LayerNorm subclass that accepts NCW data layout and performs normalization in the C dimension.

    N = batch C = channels W = sequence length
    """

    def __init__(self, hidden_size, eps=1e-12):
        nn.LayerNorm.__init__(self, normalized_shape=hidden_size, eps=eps)  # instantiates self.{weight, bias, eps}

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = nn.LayerNorm.forward(self, x)
        return x.permute(0, 2, 1)
