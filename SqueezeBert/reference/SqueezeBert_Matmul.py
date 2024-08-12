import torch
from torch import nn
class MatMulWrapper(nn.Module):
    """
    Wrapper for torch.matmul(). This makes flop-counting easier to implement. Note that if you directly call
    torch.matmul() in your code, the flop counter will typically ignore the flops of the matmul.
    """

    def __init__(self):
        super().__init__()

    def forward(self, mat1, mat2):
        """

        :param inputs: two torch tensors :return: matmul of these tensors

        Here are the typical dimensions found in BERT (the B is optional) mat1.shape: [B, <optional extra dims>, M, K]
        mat2.shape: [B, <optional extra dims>, K, N] output shape: [B, <optional extra dims>, M, N]
        """
        return torch.matmul(mat1, mat2)

