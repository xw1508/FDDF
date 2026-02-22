import torch
import torch.nn as nn
import torch.nn.functional as F


class FSC(nn.Module):


    def __init__(
        self,
        in_channels: int,     # C
        node_num: int = 64,   # N
        node_dim: int = 64,   # C'
        pool_stride: int = 2, #
        use_bias: bool = True,
    ):
        super().__init__()
        C = in_channels
        N = node_num
        Cp = node_dim
        self.C = C
        self.N = N
        self.Cp = Cp
        self.pool_stride = pool_stride


        self.proj_v = nn.Conv2d(C, N, kernel_size=1, bias=use_bias)


        self.proj_w = nn.Conv2d(C, Cp, kernel_size=1, bias=use_bias)

        self.Ag = nn.Parameter(torch.zeros(2 * N, 2 * N))
        nn.init.normal_(self.Ag, mean=0.0, std=0.02)

        self.Wg = nn.Linear(Cp, Cp, bias=True)

        self.fuse_1x1 = nn.Conv2d(C + Cp, C, kernel_size=1, bias=use_bias)


        self.pool = nn.MaxPool2d(kernel_size=pool_stride, stride=pool_stride) if pool_stride > 1 else nn.Identity()
        self.out_1x1 = nn.Conv2d(C, C, kernel_size=1, bias=use_bias)

    def _to_graph(self, X: torch.Tensor):

        B, C, H, W = X.shape

        Bmat = self.proj_v(X).flatten(2).transpose(1, 2).contiguous()


        R = self.proj_w(X).flatten(2).contiguous()


        G = torch.bmm(R, Bmat)
        return Bmat, G, H, W

    def _graph_conv(self, G_spa: torch.Tensor, G_fre: torch.Tensor):

        B, Cp, N = G_spa.shape


        G = torch.cat([G_spa, G_fre], dim=2)


        Gt = G.transpose(1, 2).contiguous()  # [B, 2N, C']


        I = torch.eye(2 * N, device=Gt.device, dtype=Gt.dtype)
        M = (I - self.Ag)  # [2N, 2N]
        Gt = torch.matmul(M, Gt)  # [B, 2N, C']


        Gt = self.Wg(Gt)

        # split 回两路增量
        d_spa = Gt[:, :N, :].transpose(1, 2).contiguous()  # [B, C', N]
        d_fre = Gt[:, N:, :].transpose(1, 2).contiguous()  # [B, C', N]
        return d_spa, d_fre

    def _from_graph(self, X: torch.Tensor, Bmat: torch.Tensor, Ghat: torch.Tensor, H: int, W: int):

        Ghat_t = Ghat.transpose(1, 2).contiguous()

        # [B, HW, N] x [B, N, C'] -> [B, HW, C']
        Y = torch.bmm(Bmat, Ghat_t)

        # reshape -> [B, C', H, W]
        Y = Y.transpose(1, 2).contiguous().view(X.size(0), self.Cp, H, W)


        Xcat = torch.cat([X, Y], dim=1)
        Xfuse = self.fuse_1x1(Xcat)
        return Xfuse

    def forward(self, X_fre: torch.Tensor, X_spa: torch.Tensor):

        B_fre, G_fre, H, W = self._to_graph(X_fre)
        B_spa, G_spa, _, _ = self._to_graph(X_spa)


        d_spa, d_fre = self._graph_conv(G_spa, G_fre)


        Ghat_spa = G_spa + d_spa
        Ghat_fre = G_fre + d_fre


        Xhat_spa = self._from_graph(X_spa, B_spa, Ghat_spa, H, W)
        Xhat_fre = self._from_graph(X_fre, B_fre, Ghat_fre, H, W)

        Xhat_spa = self.out_1x1(self.pool(Xhat_spa))
        Xhat_fre = self.out_1x1(self.pool(Xhat_fre))

        return Xhat_fre, Xhat_spa

