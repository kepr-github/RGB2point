import torch
import torch.nn as nn

class ChamferDistance(nn.Module):
    """Simple PyTorch implementation of Chamfer Distance."""

    def __init__(self):
        super().__init__()

    def forward(self, x, y, bidirectional=True):
        """Compute Chamfer Distance between two point clouds.

        Parameters
        ----------
        x: torch.Tensor (B, N, D)
        y: torch.Tensor (B, M, D)
        bidirectional: bool
            If True, return symmetric Chamfer distance, otherwise y->x only.
        Returns
        -------
        torch.Tensor
            Scalar distance averaged over the batch.
        """
        # Pairwise distances [B, N, M]
        dist = torch.cdist(x, y)
        # Minimum distance from x to y
        min_xy = dist.min(dim=2)[0]
        loss = min_xy.mean(dim=1)
        if bidirectional:
            min_yx = dist.min(dim=1)[0]
            loss = loss + min_yx.mean(dim=1)
        return loss.mean()
