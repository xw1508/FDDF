import torch
import torch.nn as nn
import torch.nn.functional as F


class FSA(nn.Module):


    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ):
        super().__init__()

        C = channels
        hidden = max(C // reduction, 1)

        self.fc1 = nn.Conv2d(C, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, C, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.dw = nn.Conv2d(
            C, C,
            kernel_size=3,
            padding=1,
            groups=C,
            bias=False
        )
        self.bn_dw = nn.BatchNorm2d(C, eps=bn_eps, momentum=bn_momentum)


        self.pw = nn.Conv2d(
            C, C,
            kernel_size=1,
            groups=1,
            bias=False
        )
        self.bn_pw = nn.BatchNorm2d(C, eps=bn_eps, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, F_in: torch.Tensor) -> torch.Tensor:

        F_avg = F.adaptive_avg_pool2d(F_in, 1)
        F_c = self.sigmoid(self.fc2(self.fc1(F_avg)))
        F_att = F_in * F_c


        B_map = self.relu(self.bn_dw(self.dw(F_in)))
        B_map = self.relu(self.bn_pw(self.pw(B_map)))

        F_fsa = torch.cat([B_map, F_att], dim=1)

        return F_fsa



