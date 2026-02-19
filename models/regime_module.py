import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision
import torchvision.transforms as T

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        return x * self.psi(self.relu(g1 + x1))

class RegimeRouter(nn.Module):
    def __init__(self, bottleneck_dim, skip_dim, num_regimes=3):
        super().__init__()
        
        # 1. 语义路径 (瓶颈层上采样)
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(bottleneck_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 2. 注意力门控 (彻底杀灭底噪)
        self.attn_gate = AttentionGate(F_g=128, F_l=skip_dim, F_int=64)
        
        # 3. 核心感知层：点与面的对比 (形态学感知)
        # 这里的 99 通道 = 特征融合后的通道 (128 + skip_dim)
        combined_dim = 128 + skip_dim
        
        # 支路A: 局部细节 (3x3)
        self.local_path = nn.Sequential(
            nn.Conv2d(combined_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 支路B: 宏观面积 (3x3, Dilation=5 -> 视野 11x11)
        # 解决“面积感知”的核心，视野足够大，且不糊
        self.area_path = nn.Sequential(
            nn.Conv2d(combined_dim, 64, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 4. 融合与输出
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_regimes, kernel_size=1)
        )

    def forward(self, bottleneck, skip):
        # A. 获取指挥官信号
        g = self.up_sample(bottleneck)
        
        # B. 过滤掉浅层噪声
        skip_gated = self.attn_gate(g=g, x=skip)
        
        # C. 特征融合
        feat = torch.cat([g, skip_gated], dim=1)
        
        # D. 同时看点和看面
        l_feat = self.local_path(feat)
        a_feat = self.area_path(feat)
        
        # E. 最终分类
        logits = self.classifier(torch.cat([l_feat, a_feat], dim=1))
        
        # 🔥【关键】在这里除以温度系数，锐化概率边缘
        # 0.5 的温度能有效消除“团雾”
        return logits

# =================================================

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=3): 
        super().__init__()
        
        # 🔥 核心修改：使用空洞卷积 (Dilated Conv)
        # kernel=3, dilation=3  --> 等效感受野 = 7x7
        # 既保留了之前大核的感知范围，又避免了全像素平均带来的模糊
        self.conv = nn.Conv2d(
            2, 1, 
            kernel_size=3, 
            padding=3,      # padding 必须等于 dilation 以保持尺寸
            dilation=3,     # 空洞系数
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv(res)
        return self.sigmoid(res)

class RegimeRouter_v4(nn.Module):
    def __init__(self, bottleneck_dim, skip_dim, num_regimes=3):
        super().__init__()
        
        # 1. 语义路径 (从 4x4 恢复到 16x16)
        # self.up1 = nn.Sequential(
        #     # 先无参放大 4 倍 (4x4 -> 16x16)，保证绝对平滑
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #     # 再用卷积提取特征
        #     nn.Conv2d(bottleneck_dim, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )
        self.up1 = nn.Sequential(
            # 第一步: 4x4 -> 8x8
            nn.ConvTranspose2d(bottleneck_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 第二步: 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 2. 空间路径 (从 16x16 恢复到 32x32)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 3. 空间注意力模块 (专门解决面积感知问题)
        self.spatial_attn = SpatialAttentionModule(kernel_size=7)
        
        # 4. 融合卷积 (结合语义、空间、注意力和 Skip)
        self.fuse = nn.Sequential(
            nn.Conv2d(64 + skip_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(64, num_regimes, kernel_size=1)

    def forward(self, bottleneck, skip):
        """
        bottleneck: 来自 Encoder 最深层 [B, 768, 4, 4] (假设维度)
        skip: 来自 Encoder 第一层 [B, 96, 32, 32]
        """
        # 上采样
        x = self.up1(bottleneck) # [B, 128, 16, 16]
        x = self.up2(x)          # [B, 64, 32, 32]
        
        # 拼接浅层特征
        x = torch.cat([x, skip], dim=1) # [B, 64+96, 32, 32]
        
        # 应用空间注意力：先让模型看一眼全图哪里有降水系统
        attn_mask = self.spatial_attn(x)
        x = x * attn_mask # 强化降水核心区域，抑制背景
        
        # 融合与分类
        x = self.fuse(x)
        logits = self.classifier(x)
        
        return logits

class RegimeRouter_v3(nn.Module):
    def __init__(self, in_channels, num_regimes=3):
        super().__init__()
        
        # 1. 局部特征 (Local Path) - 负责抓纹理和边缘
        # 保持原来的卷积结构，这部分是好的
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )
        
        # 2. 空间感知全局特征 (Spatial Path) - 【核心升级】
        # 从 1x1 改为 4x4，保留方位感 (左上角有雨 vs 右下角有雨)
        self.grid_size = 4 
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.grid_size),  # [B, C, 4, 4]
            nn.Conv2d(in_channels, 64, kernel_size=1), # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1)       # 保持 4x4
        )
        
        # 3. 融合层
        self.classifier = nn.Conv2d(64, num_regimes, kernel_size=1)
        
        # 4. 输入归一化 (防止 Raw Input 和 Feature 数量级不匹配)
        self.input_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # x: [B, 99, 32, 32]
        
        # 0. 预处理
        x = self.input_norm(x)
        
        # 1. 局部路径
        local_feat = self.local_conv(x) # [B, 64, 32, 32]
        
        # 2. 区域路径
        global_feat = self.global_path(x) # [B, 64, 4, 4]
        
        # 3. 广播融合
        # 把 4x4 的粗糙特征插值回 32x32，叠加到局部特征上
        global_feat_upsampled = F.interpolate(
            global_feat, 
            size=local_feat.shape[2:], # (32, 32)
            mode='bilinear', 
            align_corners=False
        )
        
        # 融合：局部纹理 + 区域背景
        feat = local_feat + global_feat_upsampled
        
        logits = self.classifier(feat)
        
        return logits


class RegimeRouter_v2(nn.Module):
    def __init__(self, in_channels, num_regimes=3):
        super().__init__()
        # 局部特征 (保留纹理判断能力)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        # 全局特征 (专门解决"面积"不可知问题)
        # 对于 128x128 输入，直接压缩成 1x1
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B, C, 1, 1]
            nn.Flatten(),                    # [B, C]
            nn.Linear(in_channels, 64),      # [B, 64]
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),               # [B, 64]
            nn.Sigmoid()                     # 门控权重
        )
        self.classifier = nn.Conv2d(64, num_regimes, kernel_size=1)

    # regime_module.py 里的 RegimeRouter.forward
    def forward(self, x):
        local_feat = self.local_conv(x)
        global_gate = self.global_path(x).unsqueeze(2).unsqueeze(3)
        feat = local_feat * global_gate + local_feat
        logits = self.classifier(feat)
        
        # 这里不需要做插值，直接返回 32x32 的 logits
        # 统一在 SwinUNetMoE 的 forward 里插值到 128x128
        return logits

class RegimeRouter_v1(nn.Module):
    """
    功能：根据特征图预测每个像素属于哪个 Regime 的概率。
    这是一个轻量级的全卷积网络 (FCN)。
    
    输出: Logits (未归一化的数值)，供后续 Softmax 或 Loss 计算使用。
    """
    def __init__(self, in_channels, num_regimes=3):
        super().__init__()
        
        # 使用 3x3 卷积提取空间上下文，保留空间结构
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        
        # 使用 1x1 卷积进行降维和特征整合
        self.conv2 = nn.Conv2d(in_channels // 2, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 最终的分类层：输出 K 个通道的 Logits
        self.classifier = nn.Conv2d(64, num_regimes, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W] (这里的 H, W 通常是 Bottleneck 的尺寸)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        logits = self.classifier(x) # [B, K, H, W]

        # 为了防止方块效应，这里进行一次上采样 (假设 bottleneck 是原图的 1/32, 这里先 x4，后续再插值)
        # 你也可以根据实际情况调整 scale_factor，或者留给外部处理
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        
        # 【关键】直接返回 Logits，绝对不要在这里做 Softmax！
        return logits

import torchvision.transforms as T


def generate_pseudo_soft_targets_v1_plus(gt_precip, area_kernel_size=5):
    """
    V1 Plus (Halo Killer 版): 
    在原版基础上增加了强制背景截断，消除边缘光晕。
    """
    with torch.no_grad():
        precip = gt_precip.squeeze(1)
        
        # --- 准备工作 ---
        # 计算密度 (保持不变)
        rain_mask = (precip > 0.1).float().unsqueeze(1)
        density = F.avg_pool2d(rain_mask, kernel_size=area_kernel_size, stride=1, padding=area_kernel_size//2).squeeze(1)
        
        # --- 核心逻辑 ---
        
        # 1. 原始概率计算 (保持不变)
        # 条件A: 强降水
        cond_strong = torch.sigmoid((precip - 15.0) / 0.05)
        # 条件B: 孤立强点
        cond_embryo = torch.sigmoid((precip - 8.0) / 0.05) * torch.sigmoid((0.3 - density) / 0.05)
        p_convective_raw = torch.max(cond_strong, cond_embryo)
        
        p_rain_raw = torch.sigmoid((precip - 0.1) / 0.05)

        # 2. 🔥【新增】物理硬截断 (The Halo Killer) 🔥
        # 只要真值 < 0.05 (几乎无雨)，强制掩盖所有降水概率
        true_bg_mask = (precip < 0.05).float()
        
        # 应用截断：如果是背景，降水概率直接归零
        p_rain = p_rain_raw * (1.0 - true_bg_mask)
        p_convective = p_convective_raw * (1.0 - true_bg_mask)
        
        # --- 结果合并 (保持不变) ---
        p_stratiform = p_rain * (1.0 - p_convective)
        p_background = 1.0 - p_rain # 这里会自动变成 1.0 (如果 p_rain 是 0)
        
        soft_targets = torch.stack([p_background, p_stratiform, p_convective], dim=1)
        return soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)


# def generate_pseudo_soft_targets_v1_plus(gt_precip, area_kernel_size=5):
#     """
#     极简版 V1 Plus: 
#     1. 铁律: > 15mm 就是暴雨 (保住 RMSE)
#     2. 补漏: > 8mm 且 孤立 (密度<0.3) 也是暴雨 (保住物理故事)
#     """
#     with torch.no_grad():
#         precip = gt_precip.squeeze(1)
        
#         # --- 准备工作 ---
#         # 计算密度 (5x5 核)
#         rain_mask = (precip > 0.1).float().unsqueeze(1)
#         density = F.avg_pool2d(rain_mask, kernel_size=area_kernel_size, stride=1, padding=area_kernel_size//2).squeeze(1)
        
#         # --- 核心逻辑 (就这两行) ---
#         # 条件A: 只要够强 (稳健性)
#         cond_strong = torch.sigmoid((precip - 15.0) / 0.05)
        
#         # 条件B: 中等强 + 孤立 (物理故事：初生对流)
#         cond_embryo = torch.sigmoid((precip - 8.0) / 0.05) * torch.sigmoid((0.3 - density) / 0.05)
        
#         # --- 结果合并 ---
#         p_convective = torch.max(cond_strong, cond_embryo)
        
#         # --- 其他专家 (自动补齐) ---
#         p_rain = torch.sigmoid((precip - 0.1) / 0.05)
#         p_stratiform = p_rain * (1.0 - p_convective)
#         p_background = 1.0 - p_rain
        
#         soft_targets = torch.stack([p_background, p_stratiform, p_convective], dim=1)
#         return soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)

def generate_pseudo_soft_targets_v5(gt_precip, area_kernel_size=5): # 用你成功的小核(5或7)
    with torch.no_grad():
        precip = gt_precip.squeeze(1)
        
        # 1. 拿回你最成功的 0.05 tau (极锐利)
        tau_intensity = 0.05
        
        # 2. 组织度计算 (用你成功的小核)
        # binary_proxy: 哪里有降水
        binary_proxy = (precip >= 1.0).float().unsqueeze(1)
        density = F.avg_pool2d(binary_proxy, kernel_size=area_kernel_size, stride=1, padding=area_kernel_size//2).squeeze(1)
        
        # 3. 动态阈值故事：
        # 如果组织度高(大雨团)，阈值是 10.0
        # 如果组织度低(孤立点)，阈值下调到 8.0 (捕捉初生对流)
        # 这样逻辑就顺了：孤立的小雨更值得被专家2关注。
        dynamic_thresh = 10.0 - 2.0 * (1.0 - density) # 范围在 8.0 到 10.0 之间
        
        # 4. 极锐利的 Sigmoid
        p_convective = torch.sigmoid((precip - 8.0) / tau_intensity)
        p_rain = torch.sigmoid((precip - 0.1) / tau_intensity)
        p_stratiform = p_rain * (1.0 - p_convective)
        p_background = 1.0 - p_rain
        
        soft_targets = torch.stack([p_background, p_stratiform, p_convective], dim=1)
        return soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)

def generate_pseudo_soft_targets_v4(gt_precip, 
                                       base_threshold=10.0, 
                                       slope=2.0,            # 降到2.0，更温和
                                       area_kernel_size=31,  # 针对128图，31约等于1000px面积
                                       tau=1.0):
    with torch.no_grad():
        precip = gt_precip.squeeze(1)
        
        # 1. 密度计算 (模拟 Log Area)
        # 只有 > 0.1 的地方才算雨区
        rain_mask = (precip > 0.1).float()
        
        # padding=15 保证尺寸不变 (31//2 = 15)
        # 结果代表：当前像素周围 31x31 范围内，下雨的比例是多少 (0~1)
        density = F.avg_pool2d(rain_mask.unsqueeze(1), 
                               kernel_size=area_kernel_size, 
                               stride=1, 
                               padding=area_kernel_size//2).squeeze(1)
        
        # 2. 动态阈值
        # 如果是大片雨云 (density->1): Threshold -> 12.0 (要求更高才算核心)
        # 如果是孤立点 (density->0): Threshold -> 10.0 (稍微强点就算核心)
        dynamic_thresh = base_threshold + slope * density
        
        # 3. 生成标签
        p_convective = torch.sigmoid((precip - dynamic_thresh) / tau)
        p_rain = torch.sigmoid((precip - 0.1) / 0.1)
        p_stratiform = p_rain * (1.0 - p_convective)
        p_background = 1.0 - p_rain
        
        soft_targets = torch.stack([p_background, p_stratiform, p_convective], dim=1)
        # 归一化
        soft_targets = soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)
        
    return soft_targets


# ====================================================================
def generate_pseudo_soft_targets_v3_adaptive(gt_precip, 
                                             # 基础参数
                                             base_threshold=10.0,  # 对应图中蓝线的截距 10.0
                                             slope=5.0,            # 对应图中蓝线的斜率 (控制面积对阈值的影响)
                                             # 密度感知
                                             area_kernel_size=13,  # 感受野，越大越能感知"大面积"
                                             sigma=2.0,            # 高斯模糊的平滑度
                                             # 平滑参数 (Sigmoid 温度)
                                             tau=1.0):
    """
    Soft-V3: 可微分的自适应阈值标签 (Differentiable Adaptive Thresholding)
    核心逻辑：复刻 V3 分析图中的蓝色实线，捕捉"小而强"的对流。
    """
    with torch.no_grad():
        B, _, H, W = gt_precip.shape
        precip = gt_precip.squeeze(1)

        # 1. 计算"局部密度" (Proxy for Cluster Area)
        # 使用大核高斯模糊来感知周边的降水范围
        # 如果一个点处于大片雨区中心，density 会高；如果是孤立点，density 会低
        binary_proxy = (precip >= 1.0).float().unsqueeze(1) # [B, 1, H, W]
        gaussian = T.GaussianBlur(kernel_size=area_kernel_size, sigma=sigma)
        density = gaussian(binary_proxy).squeeze(1) # [B, H, W], 范围 0~1
        
        # 2. 计算动态阈值 (Dynamic Threshold)
        # 对应公式: Threshold = 10.0 + k * log(Area)
        # 这里用 density 近似 log(Area)。Density 越大 (大雨团)，阈值越高 (要求更严)
        # Density 越小 (孤立小雨团)，阈值越低 (更容易被捕获，只要稍微强一点)
        dynamic_threshold = base_threshold + slope * density

        # 3. 判定 Regime 2 (对流核心)
        # 逻辑：强度 > 动态阈值
        # 使用 Sigmoid 实现软判定
        p_convective = torch.sigmoid((precip - dynamic_threshold) / tau)

        # 4. 判定 Regime 1 (层状云/背景雨)
        # 逻辑：有雨 (大于 0.1) 但不是对流
        p_rain = torch.sigmoid((precip - 0.1) / 0.1)
        p_stratiform = p_rain * (1.0 - p_convective)

        # 5. 判定 Regime 0 (无雨)
        p_background = 1.0 - p_rain

        # 6. 堆叠并归一化
        soft_targets = torch.stack([p_background, p_stratiform, p_convective], dim=1)
        soft_targets = soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)

    return soft_targets
# ====================================================================

# ====================================================================
# v5开始
# === 工具函数 1: 像素级标签 (来自 V1+ 的最终版) ===
def generate_pixel_labels(gt_precip, 
                          intensity_thresholds=(0.1, 15.0), 
                          area_kernel_size=5,
                          tau_intensity=0.1, 
                          tau_density=0.1):
    with torch.no_grad():
        precip = gt_precip.squeeze(1)
        p_rain = torch.sigmoid((precip - intensity_thresholds[0]) / tau_intensity)
        p_heavy = torch.sigmoid((precip - intensity_thresholds[1]) / tau_intensity)
        
        binary_heavy_proxy = (precip >= intensity_thresholds[1]).float()
        density = F.avg_pool2d(
            binary_heavy_proxy.unsqueeze(1),
            kernel_size=area_kernel_size,
            stride=1,
            padding=area_kernel_size // 2
        ).squeeze(1)
        p_organized = torch.sigmoid((density - tau_density) / (tau_density / 4 + 1e-6))
        
        p_regime2 = p_heavy * p_organized
        p_regime1 = p_rain * (1.0 - p_regime2)
        p_regime0 = 1.0 - p_rain
        
        soft_targets = torch.stack([p_regime0, p_regime1, p_regime2], dim=1)
    return soft_targets

# === 工具函数 2: 区域级标签 (来自 V2 的最终版) ===
def generate_region_labels(gt_precip,
                           thresholds={'rain': 0.1, 'convective': 15.0},
                           area_limits={'convective_max': 1000, 'stratiform_min': 1000}):
    with torch.no_grad():
        device = gt_precip.device
        B, _, H, W = gt_precip.shape
        
        # 直接生成硬标签 [B, 3, H, W]
        hard_masks = torch.zeros((B, 3, H, W), device=device, dtype=torch.float32)

        for i in range(B):
            precip_map = gt_precip[i, 0].cpu().numpy()
            
            # 1. 背景
            hard_masks[i, 0] = torch.from_numpy(precip_map < thresholds['rain'])
            
            # 2. 连通域分析
            rain_mask_np = (precip_map >= thresholds['rain']).astype(np.uint8)
            num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(rain_mask_np, 8)

            if num_labels > 1:
                for label_id in range(1, num_labels):
                    area = stats[label_id, cv2.CC_STAT_AREA]
                    component_mask = (labels_map == label_id)
                    max_intensity = precip_map[component_mask].max()
                    
                    is_convective = (area < area_limits['convective_max'] and 
                                     max_intensity > thresholds['convective'])
                    is_stratiform = (area >= area_limits['stratiform_min'])
                    
                    if is_convective:
                        hard_masks[i, 2][component_mask] = 1.0
                    elif is_stratiform:
                        hard_masks[i, 1][component_mask] = 1.0
                    else:
                        hard_masks[i, 1][component_mask] = 1.0
        
    return hard_masks.to(device)

# === 最终的混合函数 ===
def generate_hybrid_soft_targets(gt_precip, 
                                 pixel_weight=0.8,
                                 # V1 (pixel) 的参数
                                 intensity_thresholds=(0.1, 15.0),
                                 area_kernel_size=5,
                                 tau_intensity=0.1,
                                 tau_density=0.1,
                                 # V2 (region) 的参数
                                 thresholds={'rain': 0.1, 'convective': 15.0},
                                 area_limits={'convective_max': 1000, 'stratiform_min': 1000}
                                ):
    """
    V7 混合标签：融合像素级精度和区域级物理结构。
    """
    # 1. 获取高精度的像素级软标签
    # 只把 V1 认识的参数传给它
    soft_pixel = generate_pixel_labels(
        gt_precip,
        intensity_thresholds=intensity_thresholds,
        area_kernel_size=area_kernel_size,
        tau_intensity=tau_intensity,
        tau_density=tau_density
    )
    
    # 2. 获取有物理意义的区域级硬标签
    # 只把 V2 认识的参数传给它
    hard_region = generate_region_labels(
        gt_precip,
        thresholds=thresholds,
        area_limits=area_limits
    )
    
    # 3. 混合
    hybrid_targets = pixel_weight * soft_pixel + (1 - pixel_weight) * hard_region
    
    # 4. 归一化
    hybrid_targets = hybrid_targets / (hybrid_targets.sum(dim=1, keepdim=True) + 1e-8)
    
    return hybrid_targets

# ==============================================================
# V5结束

# def generate_pseudo_soft_targets_v4(gt_precip, 
#                                        intensity_thresholds=(0.1, 15.0), 
#                                        # 不再需要 area_kernel_size
#                                        tau_intensity=0.1, 
#                                        tau_density=0.1):
    
#     with torch.no_grad():
#         B, _, H, W = gt_precip.shape
#         precip = gt_precip.squeeze(1)

#         # 1. 像素级强度 (不变)
#         p_rain = torch.sigmoid((precip - intensity_thresholds[0]) / tau_intensity)
#         p_heavy = torch.sigmoid((precip - intensity_thresholds[1]) / tau_intensity)

#         # 2. 结构级组织度 (改用高斯模糊！)
#         binary_heavy_proxy = (precip >= intensity_thresholds[1]).float().unsqueeze(1) # [B, 1, H, W]
        
#         # 定义高斯核
#         # kernel_size=13 (保持范围), sigma=3.0 (控制晕染程度，3.0约为kernel的1/4，比较自然)
#         gaussian = T.GaussianBlur(kernel_size=13, sigma=3.0)
        
#         density = gaussian(binary_heavy_proxy).squeeze(1)
        
#         # 3. 组合 (不变)
#         p_organized = torch.sigmoid((density - tau_density) / (tau_density / 4 + 1e-6))
#         p_regime2 = p_heavy * p_organized
#         p_regime1 = p_rain * (1.0 - p_regime2)
#         p_regime0 = 1.0 - p_rain

#         soft_targets = torch.stack([p_regime0, p_regime1, p_regime2], dim=1)
#         soft_targets = soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)

#     return soft_targets

def generate_pseudo_soft_targets_v3(gt_precip, 
                                       # 1. 坚持你的物理阈值 15.0
                                       intensity_thresholds=(0.1, 15.0), 
                                       # 2. 用大 Kernel (15或21) 来感知“面积”
                                       # 15x15 = 225 pixels, 21x21 = 441 pixels
                                       # 这对应你图一里的“大面积”概念
                                       area_kernel_size=5,              
                                       # 3. 坚持 tau=0.1 来救背景
                                       tau_intensity=0.1, 
                                       tau_density=0.1):
    
    with torch.no_grad():
        B, _, H, W = gt_precip.shape
        precip = gt_precip.squeeze(1)

        # === 1. 像素级强度判断 (Pixel-level) ===
        # 这一步保证了边缘不会被“连坐”，只有真的强的点概率才高
        p_rain = torch.sigmoid((precip - intensity_thresholds[0]) / tau_intensity)
        p_heavy = torch.sigmoid((precip - intensity_thresholds[1]) / tau_intensity)

        # === 2. 结构级面积判断 (Structure-level) ===
        # 逻辑：只有当周围一圈都是强降水时，才认为是“有组织的核心”
        binary_heavy_proxy = (precip >= intensity_thresholds[1]).float()
        
        # 使用大 Kernel 进行池化，计算局部密度
        # density 越高，说明处于大雨团的中心
        density = F.avg_pool2d(
            binary_heavy_proxy.unsqueeze(1),
            kernel_size=area_kernel_size,
            stride=1,
            padding=area_kernel_size // 2
        ).squeeze(1)
        
        p_organized = torch.sigmoid((density - tau_density) / (tau_density / 4 + 1e-6))

        # === 3. 组合逻辑 ===
        # Regime 2 (对流核心): 强度大 AND 处于大雨团中心
        p_regime2 = p_heavy * p_organized
        
        # Regime 1 (层状/边缘): 有雨 BUT (强度不够 OR 处于边缘)
        p_regime1 = p_rain * (1.0 - p_regime2)
        
        # Regime 0 (背景)
        p_regime0 = 1.0 - p_rain

        # 归一化
        soft_targets = torch.stack([p_regime0, p_regime1, p_regime2], dim=1)
        soft_targets = soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)

    return soft_targets

def generate_pseudo_soft_targets_v2(gt_precip,
                                    thresholds={'rain': 0.1, 'convective': 15.0},
                                    area_limits={'convective_max': 1000, 'stratiform_min': 1000}):
    """
    V2 版本：基于连通域面积和强度，显式解耦大尺度层状云和中小尺度对流云。
    """
    with torch.no_grad():
        device = gt_precip.device
        B, _, H, W = gt_precip.shape
        
        # 初始化三个 Regime 的 mask (硬标签)
        mask_regime0 = torch.zeros((B, H, W), device=device, dtype=torch.bool)
        mask_regime1 = torch.zeros((B, H, W), device=device, dtype=torch.bool) # 大尺度层状云
        mask_regime2 = torch.zeros((B, H, W), device=device, dtype=torch.bool) # 中小尺度对流云

        for i in range(B):
            # 取出单张降水图，转为 numpy 用于 cv2 处理
            precip_map = gt_precip[i, 0].cpu().numpy()

            # 1. 确定背景 (Regime 0)
            mask_regime0[i] = torch.from_numpy(precip_map < thresholds['rain']).to(device)
            
            # 2. 识别所有降水区域
            rain_mask_np = (precip_map >= thresholds['rain']).astype(np.uint8)
            
            # 3. 连通域分析
            num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(rain_mask_np, connectivity=8)

            # 遍历所有找到的雨团 (label 0 是背景，跳过)
            if num_labels > 1:
                for label_id in range(1, num_labels):
                    area = stats[label_id, cv2.CC_STAT_AREA]
                    
                    # 获取当前雨团的 mask
                    component_mask_np = (labels_map == label_id)
                    
                    # 获取该雨团内的最大强度
                    max_intensity = precip_map[component_mask_np].max()

                    # === 4. 制定物理规则，分配 Regime ===
                    is_convective = (area < area_limits['convective_max'] and 
                                     max_intensity > thresholds['convective'])
                    
                    is_stratiform = (area >= area_limits['stratiform_min'])

                    component_mask_torch = torch.from_numpy(component_mask_np).to(device)

                    if is_convective:
                        # 规则1：面积不大但强度极高 -> 中小尺度对流 (Regime 2)
                        mask_regime2[i] |= component_mask_torch
                    elif is_stratiform:
                        # 规则2：面积巨大 -> 大尺度层状云 (Regime 1)
                        mask_regime1[i] |= component_mask_torch
                    else:
                        # 规则3：剩下的“不大不小、不强不弱”的降水，也归为层状云
                        mask_regime1[i] |= component_mask_torch

        # 5. 将硬标签转换为软标签 (通过高斯模糊)
        # 这样做可以给模型提供平滑的过渡区域，比纯 0/1 标签更好训练
        soft_regime0 = mask_regime0.float().unsqueeze(1)
        soft_regime1 = mask_regime1.float().unsqueeze(1)
        soft_regime2 = mask_regime2.float().unsqueeze(1)

        # 定义一个小型高斯模糊核
        gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1.0)
        
        soft_regime0 = gaussian_blur(soft_regime0)
        soft_regime1 = gaussian_blur(soft_regime1)
        soft_regime2 = gaussian_blur(soft_regime2)

        # 6. 堆叠并归一化
        soft_targets = torch.cat([soft_regime0, soft_regime1, soft_regime2], dim=1)
        soft_targets = soft_targets.clamp(min=0)
        soft_targets = soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)

    return soft_targets

def generate_pseudo_soft_targets_v1(gt_precip, 
                                 intensity_thresholds=(0.1, 15.0), 
                                 area_kernel_size=5,
                                 tau_intensity=0.05, 
                                 tau_density=0.1):
    """
    生成基于强度的软标签 (Physics-Informed Targets)。
    用于在训练时监督 Router。
    """
    with torch.no_grad():
        B, _, H, W = gt_precip.shape
        precip = gt_precip.squeeze(1)

        # 1. 强度隶属度 (Intensity Membership)
        p_rain = torch.sigmoid((precip - intensity_thresholds[0]) / tau_intensity)
        p_heavy = torch.sigmoid((precip - intensity_thresholds[1]) / tau_intensity)

        # 2. 组织度 (Organization Degree) - 简单的局部密度
        binary_heavy_proxy = (precip >= intensity_thresholds[1]).float()
        density = F.avg_pool2d(
            binary_heavy_proxy.unsqueeze(1),
            kernel_size=area_kernel_size,
            stride=1,
            padding=area_kernel_size // 2
        ).squeeze(1)
        p_organized = torch.sigmoid((density - tau_density) / (tau_density / 4 + 1e-6))

        # 3. 组合计算 Regime 概率
        # Regime 2 (极端): 强度大且有组织
        p_regime2 = p_heavy * p_organized
        # Regime 1 (普通): 有雨但非极端
        p_regime1 = p_rain * (1.0 - p_regime2)
        # Regime 0 (背景): 无雨
        p_regime0 = 1.0 - p_rain

        # 4. 堆叠并归一化
        soft_targets = torch.stack([p_regime0, p_regime1, p_regime2], dim=1)
        soft_targets = soft_targets.clamp(min=0)
        soft_targets = soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-8)

    return soft_targets