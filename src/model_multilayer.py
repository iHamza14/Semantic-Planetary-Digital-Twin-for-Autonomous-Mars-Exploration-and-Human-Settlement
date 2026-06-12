import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Conv2d -> BN -> ReLU (the atomic unit everywhere)."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BottleneckCSP(nn.Module):
    """
    YOLOv5-style Cross-Stage Partial bottleneck.
    Splits input into two branches, processes one through residual
    bottlenecks, then concatenates & fuses.
    """
    def __init__(self, c, n_bottlenecks=2, expansion=0.5):
        super().__init__()
        hidden = int(c * expansion)
        self.cv1 = ConvBNReLU(c, hidden, k=1, p=0)          # branch A
        self.cv2 = ConvBNReLU(c, hidden, k=1, p=0)          # branch B (shortcut)
        self.bottlenecks = nn.Sequential(*[
            nn.Sequential(
                ConvBNReLU(hidden, hidden, k=1, p=0),
                ConvBNReLU(hidden, hidden, k=3, p=1),
            )
            for _ in range(n_bottlenecks)
        ])
        self.fuse = ConvBNReLU(2 * hidden, c, k=1, p=0)

    def forward(self, x):
        a = self.bottlenecks(self.cv1(x))
        b = self.cv2(x)
        return self.fuse(torch.cat([a, b], dim=1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling – Fast (YOLOv5 variant).
    Uses three sequential max-pools to capture 5×5, 9×9, 13×13 receptive
    fields cheaply, then fuses all four feature sets.
    """
    def __init__(self, c, pool_size=5):
        super().__init__()
        hidden = c // 2
        self.cv1 = ConvBNReLU(c, hidden, k=1, p=0)
        self.pool = nn.MaxPool2d(pool_size, stride=1,
                                 padding=pool_size // 2)
        self.cv2 = ConvBNReLU(hidden * 4, c, k=1, p=0)

    def forward(self, x):
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# ---------------------------------------------------------------------------
# Scale branches
# ---------------------------------------------------------------------------

class UpBranch(nn.Module):
    """36×36 → 72×72  (fine scale, high resolution)."""
    def __init__(self, in_c=128, out_c=128):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False)
        self.conv = nn.Sequential(
            ConvBNReLU(in_c, out_c),
            ConvBNReLU(out_c, out_c),
        )

    def forward(self, x):               # x: [B, in_c, 36, 36]
        return self.conv(self.up(x))    # → [B, out_c, 72, 72]


class DownBranch(nn.Module):
    """36×36 → 18×18  (coarse scale, semantic-rich)."""
    def __init__(self, in_c=128, out_c=128):
        super().__init__()
        self.conv = nn.Sequential(
            # strided conv as the downsampler (better than pooling for learned features)
            ConvBNReLU(in_c, out_c, k=3, s=2, p=1),
            ConvBNReLU(out_c, out_c),
        )

    def forward(self, x):               # x: [B, in_c, 36, 36]
        return self.conv(x)             # → [B, out_c, 18, 18]


# ---------------------------------------------------------------------------
# YOLO-inspired multi-scale neck
# ---------------------------------------------------------------------------

class MultiScaleNeck(nn.Module):
    """
    Fuses three feature maps of sizes 18×18, 36×36, 72×72 into a single
    compact representation via a bidirectional top-down / bottom-up FPN
    pass followed by an SPPF + CSP stack and global average pooling.

    Architecture (mirrors YOLOv5 PANet neck philosophy):

        P3 (72×72) ─────────────────────────────────┐
                                                     ▼
        P4 (36×36) ──► lateral_4 ──► [cat+CSP] ──► P4'──► [cat+CSP] ──► P4''
                                          ▲                      ▲
        P5 (18×18) ──► lateral_5 ──► [up→P4'] ──► P5' ──► [down→P4'']
                                   SPPF here ↑

    Final global pooling over P5'' → 1-D feature vector.
    """

    def __init__(self, c_fine=128, c_mid=128, c_coarse=128, out_dim=512):
        super().__init__()

        # ---- lateral projections (align channels to neck_c) ----
        neck_c = 256
        self.lat_fine   = ConvBNReLU(c_fine,   neck_c, k=1, p=0)  # 72×72
        self.lat_mid    = ConvBNReLU(c_mid,    neck_c, k=1, p=0)  # 36×36
        self.lat_coarse = ConvBNReLU(c_coarse, neck_c, k=1, p=0)  # 18×18

        # ---- SPPF on the coarsest (most semantic) level ----
        self.sppf = SPPF(neck_c)

        # ---- Top-down pathway (coarse → fine) ----
        # P5→P4  (upsample 18→36, cat with lat_mid, CSP)
        self.td_csp_4 = BottleneckCSP(neck_c * 2, n_bottlenecks=2)
        self.td_proj_4 = ConvBNReLU(neck_c * 2, neck_c, k=1, p=0)

        # P4'→P3  (upsample 36→72, cat with lat_fine, CSP)
        self.td_csp_3 = BottleneckCSP(neck_c * 2, n_bottlenecks=2)
        self.td_proj_3 = ConvBNReLU(neck_c * 2, neck_c, k=1, p=0)

        # ---- Bottom-up pathway (fine → coarse) ----
        # P3'→P4''  (stride-2, cat with P4', CSP)
        self.bu_down_4 = ConvBNReLU(neck_c, neck_c, k=3, s=2, p=1)
        self.bu_csp_4  = BottleneckCSP(neck_c * 2, n_bottlenecks=2)
        self.bu_proj_4 = ConvBNReLU(neck_c * 2, neck_c, k=1, p=0)

        # P4''→P5''  (stride-2, cat with P5', CSP)
        self.bu_down_5 = ConvBNReLU(neck_c, neck_c, k=3, s=2, p=1)
        self.bu_csp_5  = BottleneckCSP(neck_c * 2, n_bottlenecks=2)
        self.bu_proj_5 = ConvBNReLU(neck_c * 2, neck_c, k=1, p=0)

        # ---- Final aggregation head ----
        # Collapse all three refined levels → 1-D
        # Coarse level gets SPPF again; then GAP each and concat → MLP
        self.head_sppf = SPPF(neck_c)
        self.gap       = nn.AdaptiveAvgPool2d(1)

        # Three levels × neck_c channels each, then project to out_dim
        self.mlp = nn.Sequential(
            nn.Linear(neck_c * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, p3, p4, p5):
        """
        p3: [B, c_fine,   72, 72]
        p4: [B, c_mid,    36, 36]
        p5: [B, c_coarse, 18, 18]
        Returns: [B, out_dim]
        """
        # ---- lateral projections ----
        f3 = self.lat_fine(p3)      # [B, neck_c, 72, 72]
        f4 = self.lat_mid(p4)       # [B, neck_c, 36, 36]
        f5 = self.sppf(self.lat_coarse(p5))  # [B, neck_c, 18, 18]

        # ---- top-down ----
        f5_up = F.interpolate(f5, size=f4.shape[-2:],
                              mode='bilinear', align_corners=False)
        f4_td = self.td_proj_4(
            self.td_csp_4(torch.cat([f5_up, f4], dim=1))
        )                           # [B, neck_c, 36, 36]

        f4_up = F.interpolate(f4_td, size=f3.shape[-2:],
                              mode='bilinear', align_corners=False)
        f3_td = self.td_proj_3(
            self.td_csp_3(torch.cat([f4_up, f3], dim=1))
        )                           # [B, neck_c, 72, 72]

        # ---- bottom-up ----
        f3_dn = self.bu_down_4(f3_td)       # [B, neck_c, 36, 36]
        f4_bu = self.bu_proj_4(
            self.bu_csp_4(torch.cat([f3_dn, f4_td], dim=1))
        )                                   # [B, neck_c, 36, 36]

        f4_dn = self.bu_down_5(f4_bu)       # [B, neck_c, 18, 18]
        f5_bu = self.bu_proj_5(
            self.bu_csp_5(torch.cat([f4_dn, f5], dim=1))
        )                                   # [B, neck_c, 18, 18]

        # ---- aggregate to vector ----
        v3 = self.gap(f3_td).flatten(1)                     # [B, neck_c]
        v4 = self.gap(f4_bu).flatten(1)                     # [B, neck_c]
        v5 = self.gap(self.head_sppf(f5_bu)).flatten(1)     # [B, neck_c]

        return self.mlp(torch.cat([v3, v4, v5], dim=1))     # [B, out_dim]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MultiScaleDINOv2(nn.Module):
    """
    Pipeline
    ────────
    Image (B,3,H,W) with H,W divisible by 14
        │
        ▼
    DINOv2 ViT-S  (frozen)
        │  patch tokens → reshape to [B, 384, H/14, W/14]
        ▼
    Adapter  384 → 128  @ (H/14 × W/14)  ← this is the 36×36 "mid" level
        │                                    when H=W=504
        ├──► UpBranch    → [B, 128, H/7,  W/7 ]   ← 72×72  (fine)
        ├──► identity    → [B, 128, H/14, W/14]   ← 36×36  (mid)
        └──► DownBranch  → [B, 128, H/28, W/28]   ← 18×18  (coarse)
                                │
                                ▼
                    MultiScaleNeck  (YOLO-PANet)
                                │
                                ▼
                        [B, out_dim]  ← single output vector
    """

    def __init__(self, out_dim: int = 512, input_size: int = 504):
        """
        Args:
            out_dim:    dimension of the final output vector
            input_size: spatial size of input images; must be divisible by 14.
                        504 gives 36×36 patch grid (504/14 = 36).
        """
        super().__init__()
        assert input_size % 14 == 0, "input_size must be divisible by 14"
        self.input_size = input_size

        # ── backbone ────────────────────────────────────────────────────────
        print("Loading DINOv2 backbone …")
        try:
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14')
        except Exception as e:
            print(f"Online load failed, trying local cache … ({e})")
            hub_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "torch", "hub",
                "facebookresearch_dinov2_main")
            if os.path.exists(hub_dir):
                self.backbone = torch.hub.load(
                    hub_dir, 'dinov2_vits14', source='local')
                print(">> Loaded DINOv2 from local cache.")
            else:
                raise RuntimeError("Could not load DINOv2.") from e

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.embed_dim = 384          # ViT-Small patch embedding dimension

        # ── adapter: 384 → 128  (the 36×36 feature map) ────────────────────
        self.adapter = nn.Sequential(
            ConvBNReLU(self.embed_dim, 256, k=1, p=0),
            ConvBNReLU(256, 128, k=3, p=1),
        )

        # ── multi-scale branches ─────────────────────────────────────────────
        self.up_branch   = UpBranch(in_c=128, out_c=128)    # 36→72
        self.down_branch = DownBranch(in_c=128, out_c=128)  # 36→18

        # ── YOLO-style neck → single vector ──────────────────────────────────
        self.neck = MultiScaleNeck(
            c_fine=128, c_mid=128, c_coarse=128, out_dim=out_dim
        )

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]  –  H = W = self.input_size (e.g. 504)
        Returns:
            [B, out_dim]
        """
        B, _, H, W = x.shape
        h_grid, w_grid = H // 14, W // 14   # e.g. 36, 36

        # 1. Extract patch tokens (frozen backbone)
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            tokens = features["x_norm_patchtokens"]          # [B, N, 384]
            tokens = tokens.reshape(B, h_grid, w_grid, self.embed_dim)
            tokens = tokens.permute(0, 3, 1, 2).contiguous()# [B, 384, 36, 36]

        # 2. Adapt to mid-level feature map
        f_mid = self.adapter(tokens)        # [B, 128, 36, 36]

        # 3. Build pyramid
        f_fine   = self.up_branch(f_mid)    # [B, 128, 72, 72]
        f_coarse = self.down_branch(f_mid)  # [B, 128, 18, 18]

        # 4. Fuse through YOLO neck → output vector
        out = self.neck(f_fine, f_mid, f_coarse)  # [B, out_dim]
        return out


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = MultiScaleDINOv2(out_dim=512, input_size=504)
    model.eval()

    dummy = torch.randn(2, 3, 504, 504)
    with torch.no_grad():
        vec = model(dummy)

    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(vec.shape)}")   # expected: (2, 512)