
import torch

class LPIPSWrapper(torch.nn.Module):
    """
    Thin wrapper around 'lpips' package (if installed). If not installed,
    returns zero loss.
    """
    def __init__(self):
        super().__init__()
        try:
            import lpips  # type: ignore
            self.net = lpips.LPIPS(net='vgg')
            self.enabled = True
        except Exception as e:
            self.net = None
            self.enabled = False

    def forward(self, x, y):
        if not self.enabled or self.net is None:
            return torch.tensor(0.0, device=x.device)
        # expects [-1,1]
        def to_m1p1(z):
            return z*2 - 1
        return self.net(to_m1p1(x), to_m1p1(y)).mean()
