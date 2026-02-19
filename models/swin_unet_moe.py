#swin_unet_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import json
# 引用同目录下的 regime_module
from .regime_module import RegimeRouter

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
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x); return x

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
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
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
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x); x = self.proj_drop(x); return x

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
        H, W = self.input_resolution; B, L, C = x.shape
        shortcut = x; x = self.norm1(x); x = x.view(B, H, W, C)
        pad_l, pad_t = (self.window_size - W % self.window_size) % self.window_size, (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, 0, pad_t, 0))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=None)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_t > 0 or pad_l > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x); x = x + self.drop_path(self.mlp(self.norm2(x))); return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution, self.dim = input_resolution, dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    def forward(self, x):
        H, W = self.input_resolution; B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0, x1, x2, x3 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x); x = self.reduction(x); return x

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
        x = x.view(B, -1, C // 4)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim, self.input_resolution, self.depth = dim, input_resolution, depth
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, upsample=None):
        super().__init__()
        self.dim, self.input_resolution, self.depth = dim, input_resolution, depth
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

# === SwinUNetMoE ===

class BackgroundSuppressor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        
        # 🔥 关键修复：让初始 Mask 等于 1.0 (全通过)
        # 找到最后一层 Conv2d (索引是 3)
        nn.init.constant_(self.gate_conv[3].bias, 5.0) 
        # 同时也把权重初始化得小一点，防止初始状态有太大的空间波动
        nn.init.normal_(self.gate_conv[3].weight, std=0.01)

    def forward(self, x_shallow):
        B, L, C = x_shallow.shape
        H_feat = W_feat = int(L ** 0.5)
        x_4d = x_shallow.transpose(1, 2).reshape(B, C, H_feat, W_feat).contiguous()
        feat_up = F.interpolate(x_4d, size=(128, 128), mode='bilinear', align_corners=False)
        
        return self.gate_conv(feat_up)

class SwinUNetMoE(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=3, num_regimes=3,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, stats_path='/root/autodl-tmp/normalization_stats.json'):
        super().__init__()
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # 公式: (ln(0+1) - mean) / std = -mean / std
        norm_zero = -stats['mean'] / stats['std']
        self.register_buffer('norm_zero', torch.tensor(norm_zero, dtype=torch.float32))

        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        
        self.num_regimes = num_regimes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]

        # Shared Encoder
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if self.patch_norm: self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer), self.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        # Router
        # self.router = RegimeRouter(in_channels=self.num_features, num_regimes=num_regimes)
        # self.router = RegimeRouter(in_channels=self.embed_dim + in_chans, num_regimes=num_regimes)
        self.router = RegimeRouter(
            bottleneck_dim=self.num_features, 
            skip_dim=self.embed_dim, 
            num_regimes=num_regimes
        )

        self.suppressor = BackgroundSuppressor(self.embed_dim) 

        self.router_upsampler = nn.Sequential(
            nn.ConvTranspose2d(num_regimes, num_regimes * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_regimes * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_regimes * 4, num_regimes, kernel_size=4, stride=2, padding=1),
            # 💡 注意：这里去掉了 nn.Softmax(dim=1)
        )
        # Experts
        self.experts = nn.ModuleList([self._build_decoder() for _ in range(num_regimes)])

    def _build_decoder(self):
        layers_up = nn.ModuleList()
        concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers - 1):
            dim_in = int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            
            layers_up.append(BasicLayerUp(
                dim=dim_in,
                input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                  self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                depth=self.depths[self.num_layers - 1 - i_layer],
                num_heads=self.num_heads[self.num_layers - 1 - i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                norm_layer=nn.LayerNorm,
                upsample=PatchExpanding,
            ))
            
            concat_back_dim.append(nn.Linear(dim_in, dim_in // 2))

        norm_up = nn.LayerNorm(self.embed_dim)
        up_final = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=2, stride=2),
            nn.Conv2d(self.embed_dim // 4, 1, kernel_size=1)
        )
        return nn.ModuleDict({
            'layers_up': layers_up, 'concat_back_dim': concat_back_dim,
            'norm_up': norm_up, 'up_final': up_final
        })

    def _forward_expert(self, expert, x, skips):
        for i, layer_up_module in enumerate(expert['layers_up']):
            x = layer_up_module(x)
            skip = skips[self.num_layers - 2 - i]
            x = torch.cat([x, skip], dim=-1)
            x = expert['concat_back_dim'][i](x)

        x = expert['norm_up'](x)
        H_patch, W_patch = self.patches_resolution
        x = x.view(x.shape[0], H_patch, W_patch, -1).permute(0, 3, 1, 2).contiguous()
        # 拿到专家最后的预测图 [B, 1, 128, 128]
        output = expert['up_final'](x) 

        return output


    def forward(self, x, hard_routing=False):
        B, C, H, W = x.shape
        
        # --- 1. Encoder 阶段 (整理后的简洁逻辑) ---
        x_embed = self.patch_embed(x) 
        if self.patch_norm:
            x_embed = x_embed.permute(0, 2, 3, 1) # [B, H, W, C]
            x_embed = self.norm(x_embed)
            x_embed = x_embed.permute(0, 3, 1, 2) # [B, C, H, W]
            
        B, C, H_p, W_p = x_embed.shape
        x_embed = x_embed.flatten(2).transpose(1, 2) # [B, L, C]
        x_embed = self.pos_drop(x_embed)
        
        skips = []
        for layer in self.layers:
            skips.append(x_embed)
            x_embed = layer(x_embed)
        
        # --- 2. 准备 Router 输入 ---
        bottleneck_4d = x_embed.transpose(1, 2).reshape(B, self.num_features, 4, 4).contiguous()
        skip_shallow_4d = skips[0].transpose(1, 2).reshape(B, self.embed_dim, 32, 32).contiguous()

        # --- 3. Router 决策 ---
        router_logits_32 = self.router(bottleneck_4d, skip_shallow_4d) # [B, 3, 32, 32]
        router_logits_128 = self.router_upsampler(router_logits_32)

        # 3. 🔥【关键：Logit Adjustment】
        # 计算偏移量：tau=1.0 是标准值。
        # 暴雨类的 log(0.0009) 约等于 -7.0，减去它等于加 7.0 分！
        # tau = 1.0 
        # log_priors = torch.log(self.priors + 1e-8)
        # adjusted_logits_32 = router_logits_32 - tau * log_priors.view(1, 3, 1, 1)

        # # 4. 生成 128x128 概率图 (使用调整后的 Logits)
        # router_logits_128 = self.router_upsampler(adjusted_logits_32)
        
        # 🔥【去团雾关键】温度系数 T=0.5
        # 此时 router_probs_128 会变得非常尖锐，边缘不再模糊
        router_probs_128 = F.softmax(router_logits_128 / 0.5, dim=1)

        # 🔥【新增】背景抑制逻辑
        # 初始化 suppressor (在 __init__ 里: self.suppressor = BackgroundSuppressor(embed_dim))
        # 使用 skips[0] 作为高分辨率纹理来源
        fg_mask = self.suppressor(skips[0]) 
        
        # 强行抑制背景区域的权重
        # 注意：这里我们主要想抑制“非背景专家”在“背景区域”的权重
        # 但最简单的做法是直接把所有概率乘上 mask，虽然这会让总和不为1，但在加权求和时效果等同于降低了背景区的响应值
        # 或者，更精细的做法是只抑制 Regime 1 和 Regime 2
        
        router_probs_128 = router_probs_128 * fg_mask

        # --- 5. 专家解码 (物理屏蔽 Expert 0) ---
        expert_preds = []
        for i, expert in enumerate(self.experts):
            if i == 0:
                continue # 🔒 Expert 0 保持静默
            else:
                out = self._forward_expert(expert, x_embed, skips)
                expert_preds.append(out)
        
        expert_0_out = torch.full_like(expert_preds[0], fill_value=self.norm_zero) 
        expert_preds.insert(0, expert_0_out)
        expert_preds_stack = torch.cat(expert_preds, dim=1) 

        # --- 6. 混合输出 ---
        if hard_routing and not self.training:
            max_idx = torch.argmax(router_probs_128, dim=1, keepdim=True)
            router_probs_final = torch.zeros_like(router_probs_128).scatter_(1, max_idx, 1.0)
        else:
            router_probs_final = router_probs_128

        final_pred = (router_probs_final * expert_preds_stack).sum(dim=1, keepdim=True)

        # return final_pred, router_logits_128, router_probs_128, expert_preds_stack
        return final_pred, router_logits_128, router_probs_final, expert_preds_stack, fg_mask