import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import *


class HighFrequencyGenerator(nn.Module):
    """
        DCT + High-Pass Filter + iDCT + CP/SP
        α‌ = 0, Fi = Ci; α‌ = 1, Fi = 0;
    """
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # DCT
        X = dct_2d(x)

        z = build_hpf_mask(H, W, self.alpha, device)
        z = z.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        X_f = z * X

        # iDCT
        Fi = idct_2d(X_f)

        return Fi


class CP(nn.Module):
    def __init__(self, channels, reduction=16, k=16):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d((k, k))
        self.gmp = nn.AdaptiveMaxPool2d((k, k))

        self.conv_gap = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels // reduction,
                kernel_size=1,
                groups=channels // reduction,
                bias=False
            )
        )

        self.conv_gmp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels // reduction,
                kernel_size=1,
                groups=channels // reduction,
                bias=False
            )
        )

        self.conv_fuse = nn.Conv2d(
            2 * (channels // reduction),
            channels,
            kernel_size=1,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, Fi, Ci):

        gap = self.conv_gap(self.gap(Fi))
        gmp = self.conv_gmp(self.gmp(Fi))

        gap = gap.sum(dim=(2, 3), keepdim=True)
        gmp = gmp.sum(dim=(2, 3), keepdim=True)

        c = torch.cat([gap, gmp], dim=1)
        ucp = self.sigmoid(self.conv_fuse(c))

        return Ci * ucp


class SP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fi, Ci):
        u_sp = self.sigmoid(self.conv(Fi))

        return Ci * u_sp

class HFP(nn.Module):
    def __init__(self, channels, alpha=0.2):
        super().__init__()

        self.hfg = HighFrequencyGenerator(alpha)
        self.cp = CP(channels)
        self.sp = SP(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, Ci):
        Fi = self.hfg(Ci)

        C_cp = self.cp(Fi, Ci)
        C_sp = self.sp(Fi, Ci)

        C_fuse = C_cp + C_sp

        Ci_hfp = self.conv3(C_fuse)

        return Ci_hfp


# if __name__ == "__main__":
#     import torchvision.transforms as T
#     from PIL import Image
#     import matplotlib.pyplot as plt
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     img = Image.open("0a2e15e29.png").convert("L")
#     transform = T.ToTensor()
#     x = transform(img).unsqueeze(0).to(device)
#
#     model = HighFrequencyGenerator(alpha=0.1).to(device)
#
#     with torch.no_grad():
#         y = model(x)
#
#     x_np = x[0, 0].cpu().numpy()
#     y_np = y[0, 0].cpu().numpy()
#
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.title("Input")
#     plt.imshow(x_np, cmap="gray")
#     plt.axis("off")
#
#     plt.subplot(1, 2, 2)
#     plt.title("Filtered")
#     plt.imshow(y_np, cmap="gray")
#     plt.axis("off")
#
#     plt.tight_layout()
#     plt.show()