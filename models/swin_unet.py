import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

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
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() # Simplification for inference
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, input_resolution):
        H, W = input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None) # Mask logic simplified for fixed size

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

# This is the corrected version of PatchExpanding
class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: [B, L, C]
        """
        H, W = self.input_resolution
        x = self.expand(x)  # B, L, 2*C
        B, L, C_new = x.shape
        
        # Reshape to (B, H, W, 2*C)
        x = x.view(B, H, W, C_new)
        
        # This is the core of pixel shuffle
        # (B, H, W, 2*C) -> (B, H, W, 4 * (C/2)) -> (B, 2H, 2W, C/2)
        # We need to split the last dimension C_new into (2, 2, C_new/4)
        x = x.view(B, H, W, 2, 2, C_new // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C_new // 4)
        
        # Reshape back to (B, L_new, C_out)
        x = x.view(B, -1, C_new // 4)
        x = self.norm(x)
        
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, self.input_resolution)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, self.input_resolution)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class SwinUNet(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=1, num_classes=1,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Split image into non-overlapping patches
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if patch_norm:
            self.norm = norm_layer(embed_dim)

        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        
        # Absolute Position Embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patches_resolution[0], self.patches_resolution[1]))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Encoder
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=drop_path_rate,
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        # Decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            layer = BasicLayerUp(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                 input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                   self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                 depth=depths[self.num_layers - 1 - i_layer],
                                 num_heads=num_heads[self.num_layers - 1 - i_layer],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=drop_path_rate,
                                 norm_layer=norm_layer,
                                 upsample=PatchExpanding)
            self.layers_up.append(layer)
            self.concat_back_dim.append(nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 2 - i_layer)), 
                                                  int(embed_dim * 2 ** (self.num_layers - 2 - i_layer))))

        self.norm_up = norm_layer(embed_dim)
        
        # Final Expand to get back to 128x128
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x) # B, C, H/4, W/4
        
        if self.patch_norm:
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.view(B, -1, self.embed_dim)

        if self.ape:
            # Add pos embed logic here if needed, simplified for now
            pass
            
        x = self.pos_drop(x)
        
        # Encoder storage for skip connections
        x_downsample = []

        # Encoder pass
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        # Bottleneck is simply the last output of encoder
        
        # Decoder pass with Skip Connections
        for i, layer in enumerate(self.layers_up):
            # Upsample
            x = layer(x)
            
            # Skip connection
            skip = x_downsample[self.num_layers - 2 - i]
            
            # Reshape x to match skip (B, L, C)
            # Concatenate
            x = torch.cat([x, skip], dim=-1)
            x = self.concat_back_dim[i](x) # Reduce dim back

        x = self.norm_up(x) # B, L, C
        
        # Reshape to Image
        H_patch, W_patch = self.patches_resolution
        x = x.view(B, H_patch, W_patch, -1).permute(0, 3, 1, 2) # B, C, H/4, W/4
        
        # Final Upsampling to 128x128
        x = self.up_final(x)
        
        return x

# 测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 针对 128x128 图像，window_size=8
    model = SwinUNet(img_size=128, in_chans=1, num_classes=1, window_size=8).to(device)
    
    # 创建随机输入 (Batch=2, Channel=1, H=128, W=128)
    dummy_input = torch.randn(2, 1, 128, 128).to(device)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    if output.shape == dummy_input.shape:
        print("✅ 模型测试通过：输入输出维度一致！")
    else:
        print("❌ 维度不匹配，请检查代码。")