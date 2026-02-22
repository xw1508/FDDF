import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:

    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, \
        f"H,W必须能被window_size整除: H={H}, W={W}, ws={window_size}"

    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()              # [B, nH, nW, M, M, C]
    windows = x.view(-1, window_size * window_size, C)        # [B*nW, M*M, C]
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:

    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()              # [B, C, nH, M, nW, M]
    x = x.view(B, C, H, W)
    return x


class WindowMHSA(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) 必须能整除 num_heads({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        BnW, N, C = x.shape
        qkv = self.qkv(x).reshape(BnW, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                       # [3, BnW, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale           # [BnW, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(BnW, N, C)     # [BnW, N, C]
        out = self.proj_drop(self.proj(out))
        return out


class SinCosPositionalEncoding2D(nn.Module):

    def __init__(self, dim: int, base: float = 10000.0, cache: bool = True):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.cache = bool(cache)
        self._cache = {}  # key: (H,W,device,dtype) -> pe

    def forward(self, H: int, W: int, device=None, dtype=None) -> torch.Tensor:
        device = device if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        key = (H, W, device, dtype)

        if self.cache and key in self._cache:
            return self._cache[key]

        C = self.dim

        c_half = C // 2
        c_quarter = c_half // 2


        y = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)  # [H,1]
        x = torch.arange(W, device=device, dtype=dtype).unsqueeze(1)  # [W,1]


        i = torch.arange(c_quarter, device=device, dtype=dtype)  # [c_quarter]
        div = torch.pow(self.base, (2 * i) / max(c_half, 1.0))   # [c_quarter]


        y_arg = y / div  # broadcast -> [H, c_quarter]
        pe_y = torch.cat([torch.sin(y_arg), torch.cos(y_arg)], dim=1)  # [H, 2*c_quarter] = [H, c_half]


        x_arg = x / div
        pe_x = torch.cat([torch.sin(x_arg), torch.cos(x_arg)], dim=1)  # [W, c_half]


        pe = torch.zeros((C, H, W), device=device, dtype=dtype)
        # 前 c_half 通道放 y（沿高度变化），后 c_half 通道放 x（沿宽度变化）
        pe[:pe_y.shape[1], :, :] = pe_y.transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
        pe[c_half:c_half + pe_x.shape[1], :, :] = pe_x.transpose(0, 1).unsqueeze(1).repeat(1, H, 1)

        #
        pe = pe.unsqueeze(0)  # [1, C, H, W]

        if self.cache:
            self._cache[key] = pe
        return pe


class SDCI(nn.Module):

    def __init__(
        self,
        C: int,
        window_size: int = 7,
        num_heads: int = 6,
        shift_size: int = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pe_base: float = 10000.0,
        pe_cache: bool = True,
    ):
        super().__init__()
        self.C = C
        self.window_size = window_size
        self.shift_size = (window_size // 2) if shift_size is None else shift_size

        # 2C -> C
        self.reduce = nn.Conv2d(2 * C, C, kernel_size=1, stride=1, padding=0, bias=True)

        # 位置编码（无学习参数）
        self.pos_enc = SinCosPositionalEncoding2D(dim=C, base=pe_base, cache=pe_cache)

        # 两个 attention
        self.attn1 = WindowMHSA(dim=C, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=proj_drop)
        self.attn2 = WindowMHSA(dim=C, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C2, H, W = x.shape
        assert C2 == 2 * self.C, f"输入通道应为2C={2*self.C}, 但得到{C2}"
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"H/W 必须能被 window_size 整除: H={H}, W={W}, ws={self.window_size}"


        x = self.reduce(x)                                    # [B, C, H, W]


        pe = self.pos_enc(H, W, device=x.device, dtype=x.dtype)  # [1, C, H, W]
        x = x + pe

        shortcut = x


        w1 = window_partition(x, self.window_size)            # [B*nW, N, C]
        w1 = self.attn1(w1)
        x1 = window_reverse(w1, self.window_size, H, W, B)    # [B, C, H, W]

        # 4) shift
        x_shift = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))


        w2 = window_partition(x_shift, self.window_size)
        w2 = self.attn2(w2)
        x2 = window_reverse(w2, self.window_size, H, W, B)

        x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        out = x2 + shortcut
        return out

