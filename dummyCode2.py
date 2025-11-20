import torch
import torch.nn as nn


class GraphConvUnit(nn.Module):
    """
    A simplified Graph Convolution Unit.

    This approximates the graph convolution operation using a node-wise MLP
    with residual connections:

        G_out = MLP(G) + G

    Input:
        G:  [B, N, Cg] - node features

    Output:
        G_out: [B, N, Cg]
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim)
        )

    def forward(self, G: torch.Tensor):
        return G + self.mlp(G)


class FSC(nn.Module):
    """
    Frequency–Spatial Domain Feature Global Co-occurrence (FSC)

    This module performs cross-domain graph reasoning between:
        - Spatial-domain features (x_spa)
        - Frequency-domain features (x_freq)

    Following the formulation in the paper, the module:
        1. Projects both domains into graph space using 1x1 convs.
        2. Constructs node features by element-wise interaction.
        3. Applies a graph convolution unit for global reasoning.
        4. Maps back to 2D feature maps.
        5. Fuses graph features with original features.
        6. Aligns output resolution with downstream backbone via pooling.

    Inputs:
        x_spa:  [B, C, H, W]
        x_freq: [B, C, H, W]

    Outputs:
        x_spa_out:  [B, C, H_out, W_out]
        x_freq_out: [B, C, H_out, W_out]
    """

    def __init__(
        self,
        in_channels: int,
        graph_channels: int = 64,
        pool_kernel: int = 2,
        pool_stride: int = 2,
    ):
        super().__init__()

        # Projection to graph space
        self.v_spa = nn.Conv2d(in_channels, graph_channels, kernel_size=1)
        self.v_freq = nn.Conv2d(in_channels, graph_channels, kernel_size=1)

        self.w_spa = nn.Conv2d(in_channels, graph_channels, kernel_size=1)
        self.w_freq = nn.Conv2d(in_channels, graph_channels, kernel_size=1)

        # Graph convolution
        self.graph_conv = GraphConvUnit(in_dim=graph_channels)

        # Feature fusion
        self.fuse_spa = nn.Conv2d(in_channels + graph_channels, in_channels, kernel_size=1)
        self.fuse_freq = nn.Conv2d(in_channels + graph_channels, in_channels, kernel_size=1)

        # Output alignment (downsample to match backbone stage)
        self.out_spa = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        )
        self.out_freq = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        )

    def _to_graph_nodes(self, x_v: torch.Tensor, x_w: torch.Tensor):
        """
        Convert 2D feature maps into graph node features.

        Node feature = element-wise product of v(·) and w(·),
        followed by flattening:

            [B, Cg, H, W] → [B, H*W, Cg]
        """
        B, Cg, H, W = x_v.shape
        node_feat = (x_v * x_w).flatten(2)           # [B, Cg, N]
        node_feat = node_feat.permute(0, 2, 1)        # [B, N, Cg]
        return node_feat, (H, W)

    def _from_graph_nodes(self, G: torch.Tensor, hw: tuple):
        """
        Recover spatial feature maps from graph node features:

            [B, N, Cg] → [B, Cg, H, W]
        """
        H, W = hw
        B, N, Cg = G.shape
        x = G.permute(0, 2, 1).contiguous()
        return x.view(B, Cg, H, W)

    def forward(self, x_spa: torch.Tensor, x_freq: torch.Tensor):
        """
        Forward pass of FSC.

        Args:
            x_spa:  [B, C, H, W]
            x_freq: [B, C, H, W]

        Returns:
            x_spa_out:  [B, C, H_out, W_out]
            x_freq_out: [B, C, H_out, W_out]
        """

        # 1) Project to graph space
        v_spa = self.v_spa(x_spa)
        w_spa = self.w_spa(x_spa)

        v_freq = self.v_freq(x_freq)
        w_freq = self.w_freq(x_freq)

        # 2) Construct graph nodes
        G_spa, hw = self._to_graph_nodes(v_spa, w_spa)
        G_freq, _ = self._to_graph_nodes(v_freq, w_freq)

        # 3) Cross-domain fusion (node-level average)
        G = 0.5 * (G_spa + G_freq)

        # 4) Graph reasoning
        G_hat = self.graph_conv(G)

        # Domain-specific updates
        G_spa_hat = G_spa + G_hat
        G_freq_hat = G_freq + G_hat

        # 5) Map back to 2D feature maps
        x_spa_graph = self._from_graph_nodes(G_spa_hat, hw)
        x_freq_graph = self._from_graph_nodes(G_freq_hat, hw)

        # 6) Fuse graph-enhanced features with original features
        x_spa_fused = torch.cat([x_spa, x_spa_graph], dim=1)
        x_freq_fused = torch.cat([x_freq, x_freq_graph], dim=1)

        x_spa_fused = self.fuse_spa(x_spa_fused)
        x_freq_fused = self.fuse_freq(x_freq_fused)

        # 7) Output pooling (match backbone)
        x_spa_out = self.out_spa(x_spa_fused)
        x_freq_out = self.out_freq(x_freq_fused)

        return x_spa_out, x_freq_out
