import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from .regime_module import FPN_Router

# ==========================================
# 1. 基础 Swin 组件
# ==========================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, (window_size, window_size), num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        coords_h, coords_w = torch.arange(self.window_size[0]), torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        relative_coords = (torch.flatten(coords, 1)[:, :, None] - torch.flatten(coords, 1)[:, None, :]).permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop, self.proj, self.proj_drop = nn.Dropout(attn_drop), nn.Linear(dim, dim), nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim, self.input_resolution, self.num_heads, self.window_size, self.shift_size = dim, input_resolution, num_heads, window_size, shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l, pad_t = (self.window_size - W % self.window_size) % self.window_size, (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, 0, pad_t, 0))
        _, Hp, Wp, _ = x.shape
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=None)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        if pad_t > 0 or pad_l > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution, self.dim = input_resolution, dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0, x1, x2, x3 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution, self.dim = input_resolution, dim
        self.expand = nn.Linear(dim, 2*dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = x.view(B, H, W, 2, 2, C // 4).permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, C // 4)
        return x.view(B, -1, C // 4)

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, upsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer) if upsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

# ==========================================
# 2. GlobalAlignBlock（时序对齐，保留）
# ==========================================

class GlobalAlignBlock(nn.Module):
    def __init__(self, dim, num_heads, input_resolution):
        super().__init__()
        self.input_resolution = input_resolution

        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        H, W = self.input_resolution
        self.pos_embed = nn.Parameter(torch.zeros(1, H * W, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.smooth_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        nn.init.constant_(self.smooth_conv[1].weight, 0)
        nn.init.constant_(self.smooth_conv[1].bias, 0)

        self.beta = nn.Parameter(torch.zeros(1))
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4.0))
        nn.init.constant_(self.mlp.fc2.weight, 0)
        nn.init.constant_(self.mlp.fc2.bias, 0)

    def forward(self, x, skip):
        B, L, C = x.shape
        H, W = self.input_resolution
        shortcut = x

        q = self.norm_q(x) + self.pos_embed
        k = self.norm_kv(skip) + self.pos_embed
        v = self.norm_kv(skip)

        aligned_x, _ = self.cross_attn(query=q, key=k, value=v)

        aligned_x_4d = aligned_x.transpose(1, 2).view(B, C, H, W).contiguous()
        smoothed_aligned_4d = self.smooth_conv(aligned_x_4d)
        smoothed_aligned = smoothed_aligned_4d.flatten(2).transpose(1, 2)

        fused_skip = skip + self.beta * (smoothed_aligned - skip)

        x = shortcut + fused_skip
        x = x + self.mlp(self.norm_out(x))
        return x

# ==========================================
# 3. 高分辨率轻量级专家 (High-Res Experts)
# ==========================================

class BackgroundExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)


class StratiformExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = self.act(x)
        x = self.pwconv(x)
        return res + x


# 修改 models/swin_unet_moe.py 中的 ConvectiveExpert
# class ConvectiveExpert(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # 分支 A: 局部细节 (1x1 Conv)
#         self.local_branch = nn.Conv2d(dim, dim, kernel_size=1)
#         # 分支 B: 上下文结构 (空洞卷积，d=2)
#         self.context_branch = nn.Sequential(
#             nn.Conv2d(dim, dim // 2, kernel_size=1),
#             nn.GELU(),
#             nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=2, dilation=2, bias=False),
#             nn.BatchNorm2d(dim // 2),
#             nn.GELU(),
#             nn.Conv2d(dim // 2, dim, kernel_size=1)
#         )
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         return x + self.gamma * (self.local_branch(x) + self.context_branch(x))

class ConvectiveExpert(nn.Module):
    """
    加强版对流专家：专门为捕捉极值设计
    引入多尺度空洞卷积 (ASPP 思想)，大幅扩大感受野
    """
    def __init__(self, dim):
        super().__init__()
        # 分支 1: 局部细节 (1x1)
        self.branch1 = nn.Conv2d(dim, dim // 4, kernel_size=1)
        
        # 分支 2: 中等感受野 (3x3, dilation=2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )
        
        # 分支 3: 大感受野 (3x3, dilation=4)
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(dim // 4),
            nn.GELU()
        )
        
        # 分支 4: 全局池化 (捕捉整体强度)
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        size = x.shape[2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = F.interpolate(self.branch4(x), size=size, mode='bilinear', align_corners=False)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(out)
        
        # 残差连接
        return x + self.gamma * out
        
# ==========================================
# 4. 主模型: SwinUNetMoE
# ==========================================
class FinalProjection(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, 1, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

class SwinUNetMoE(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=3, num_regimes=3,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 stats_path='/root/autodl-tmp/normalization_stats.json',
                 class_prior=None):
        super().__init__()
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.register_buffer('norm_zero', torch.tensor(-stats['mean'] / stats['std'], dtype=torch.float32))

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if patch_norm else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList([BasicLayer(
            dim=int(embed_dim * 2 ** i),
            input_resolution=(self.patches_resolution[0] // (2 ** i), self.patches_resolution[1] // (2 ** i)),
            depth=depths[i], num_heads=num_heads[i], window_size=window_size, mlp_ratio=mlp_ratio,
            norm_layer=norm_layer, downsample=PatchMerging if (i < self.num_layers - 1) else None
        ) for i in range(self.num_layers)])

        self.router = FPN_Router(in_channels=[int(embed_dim * 2 ** i) for i in range(self.num_layers)], fpn_dim=128, num_regimes=num_regimes, class_prior=class_prior)
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.shared_decoder = self._build_shared_decoder(depths, num_heads, window_size)
        self.feat_dim = embed_dim // 4
        
        self.expert_bg = BackgroundExpert(self.feat_dim)
        self.expert_strat = StratiformExpert(self.feat_dim)
        self.expert_conv = ConvectiveExpert(self.feat_dim)
        
        # 独立投影头
        self.head_bg = nn.Conv2d(self.feat_dim, 1, kernel_size=1)
        self.head_strat = nn.Conv2d(self.feat_dim, 1, kernel_size=1)
        self.head_conv = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1), # 先混合通道
            nn.GELU(),                                              # 关键的非线性激活
            nn.Dropout(0.1),                                        # 防止过拟合
            nn.Conv2d(self.feat_dim, 1, kernel_size=1)              # 最终输出
        )
        

    def _build_shared_decoder(self, depths, num_heads, window_size):
        layers_up = nn.ModuleList()
        align_blocks = nn.ModuleList()
        for i in range(self.num_layers - 1):
            dim_in = int(self.embed_dim * 2 ** (self.num_layers - 1 - i))
            res = (self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i)), self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i)))
            layers_up.append(BasicLayerUp(dim=dim_in, input_resolution=res, depth=depths[self.num_layers - 1 - i], num_heads=num_heads[self.num_layers - 1 - i], window_size=window_size, mlp_ratio=self.mlp_ratio, norm_layer=self.norm_layer, upsample=PatchExpanding))
            align_blocks.append(GlobalAlignBlock(dim=dim_in // 2, num_heads=num_heads[self.num_layers - 2 - i], input_resolution=(res[0]*2, res[1]*2)) if i >= 1 else nn.Identity())
        
        feature_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(self.embed_dim, self.embed_dim // 2, kernel_size=3, padding=1), nn.BatchNorm2d(self.embed_dim // 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=3, padding=1), nn.BatchNorm2d(self.embed_dim // 4), nn.LeakyReLU(0.2, inplace=True)
        )
        return nn.ModuleDict({'layers_up': layers_up, 'align_blocks': align_blocks, 'norm_up': nn.LayerNorm(self.embed_dim), 'feature_up': feature_up})

    def forward(self, x):
        B = x.shape[0]
        x_embed = self.pos_drop(self.norm(self.patch_embed(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2).flatten(2).transpose(1, 2))
        skips, x_dec = [], x_embed
        for layer in self.layers:
            skips.append(x_dec)
            x_dec = layer(x_dec)
        
        router_inputs = [s.transpose(1, 2).view(B, -1, self.patches_resolution[0] // (2**i), self.patches_resolution[1] // (2**i)).contiguous() for i, s in enumerate(skips)]
        logits_128 = self.router(router_inputs, apply_logit_adj=False)
        
        for i in range(self.num_layers - 1):
            x_dec = self.shared_decoder['layers_up'][i](x_dec)
            skip = skips[self.num_layers - 2 - i]
            x_dec = self.shared_decoder['align_blocks'][i](x_dec, skip) if isinstance(self.shared_decoder['align_blocks'][i], GlobalAlignBlock) else x_dec + skip
        
        x_dec = self.shared_decoder['norm_up'](x_dec).view(B, self.patches_resolution[0], self.patches_resolution[1], self.embed_dim).permute(0, 3, 1, 2).contiguous()
        high_res_feats = self.shared_decoder['feature_up'](x_dec)

        router_probs = F.softmax(logits_128, dim=1)
        
        # 独立预测后再加权
        pred_bg = self.head_bg(self.expert_bg(high_res_feats))
        pred_strat = self.head_strat(self.expert_strat(high_res_feats))
        pred_conv = self.head_conv(self.expert_conv(high_res_feats))
        
        # 【修复】使用 .detach() 防止回归 Loss 反向传播破坏 Router 的分类概率
        final_pred = (pred_bg * router_probs[:, 0:1].detach() + 
                      pred_strat * router_probs[:, 1:2].detach() + 
                      pred_conv * router_probs[:, 2:3].detach())
        
        return {
            'final_pred': final_pred, 
            'logits_128': logits_128, 
            'router_probs_128': router_probs,
            'pred_bg': pred_bg,         # 新增
            'pred_strat': pred_strat,   # 新增
            'pred_conv': pred_conv      # 新增
        }
