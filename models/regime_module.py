import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json
import torchvision
import torchvision.transforms as T

# ==========================================
# 分位数统计（懒加载，仅加载一次）
# ==========================================
PERCENTILE_STATS = None

def _load_percentile_stats():
    global PERCENTILE_STATS
    if PERCENTILE_STATS is None:
        for path in [
            '/root/autodl-tmp/precipitation_percentiles.json',
            './precipitation_percentiles.json',
            '../precipitation_percentiles.json',
        ]:
            try:
                with open(path, 'r') as f:
                    PERCENTILE_STATS = json.load(f)
                print(f"✅ 已加载分位数统计: p90={PERCENTILE_STATS['p90']:.1f}mm, p95={PERCENTILE_STATS['p95']:.1f}mm")
                break
            except FileNotFoundError:
                continue
        if PERCENTILE_STATS is None:
            print("⚠️ 未找到分位数统计文件，使用默认值 (p90=10.5, p95=18.2)。请先运行 compute_percentiles.py")
            PERCENTILE_STATS = {'p90': 10.5, 'p95': 18.2}
    return PERCENTILE_STATS


# ==========================================
# FPN Router v3.0 (Feature Pyramid Network)
# ==========================================

class FPN_Router(nn.Module):
    def __init__(self, in_channels=[96, 192, 384, 768], fpn_dim=128, num_regimes=3, class_prior=None):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, fpn_dim, kernel_size=1, bias=False), nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True))
            for c in in_channels
        ])
        self.smooth_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True))
            for _ in range(3)
        ])
        
        self.router_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fpn_dim, fpn_dim // 2, kernel_size=3, padding=1), nn.BatchNorm2d(fpn_dim // 2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(fpn_dim // 2, fpn_dim // 4, kernel_size=3, padding=1), nn.BatchNorm2d(fpn_dim // 4), nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim // 4, num_regimes, kernel_size=1)
        )
        
        if class_prior is None: class_prior = torch.tensor([0.90, 0.08, 0.02])
        self.register_buffer('class_prior', class_prior)
        self.tau_logit_adj = 1.0

    def forward(self, skips, apply_logit_adj=True):
        laterals = [lateral(skip) for lateral, skip in zip(self.laterals, skips)]
        fpn_features = [laterals[3]]
        for i in range(2, -1, -1):
            upsampled = F.interpolate(fpn_features[-1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False)
            fused = self.smooth_layers[i](laterals[i] + upsampled)
            fpn_features.append(fused)
        
        logits = self.router_decoder(fpn_features[-1])
        if self.training and apply_logit_adj:
            adjustment = self.tau_logit_adj * torch.log(self.class_prior + 1e-8)
            logits = logits - adjustment.view(1, 3, 1, 1)
        return logits


# ==========================================
# 辅助函数
# ==========================================

def soft_pool2d(x, kernel_size=4, stride=4):
    # x: [B, C, H, W]
    # 1. 展开成 [B, C, H//k, k, W//k, k] 的窗口形式
    b, c, h, w = x.shape
    x_unfolded = x.view(b, c, h // kernel_size, kernel_size, w // kernel_size, kernel_size)
    
    # 2. 计算每个窗口内的 Softmax 权重 (数值大的权重极大，数值小的权重极小)
    # 减去最大值是为了数值稳定性 (防止 exp 溢出)
    x_max = x_unfolded.max(dim=3, keepdim=True)[0].max(dim=5, keepdim=True)[0]
    x_exp = torch.exp(x_unfolded - x_max)
    
    # 计算分母 (Sum of exp)
    x_sum_exp = x_exp.sum(dim=(3, 5), keepdim=True)
    weights = x_exp / (x_sum_exp + 1e-8)
    
    # 3. 加权求和：Sum(Weights * Values)
    x_soft = (weights * x_unfolded).sum(dim=(3, 5))
    
    return x_soft

def downsample_pseudo_targets_smart(pseudo_targets_128):
    """
    使用 SoftPool 技术生成 32x32 的软标签。
    完美平衡极值保留与面积真实性。
    """
    # 1. 对流通道 (Class 2) 和 层状云通道 (Class 1) 使用 SoftPool
    # 这样强对流点会以高权重被保留，而不会像 MaxPool 那样膨胀
    conv_soft = soft_pool2d(pseudo_targets_128[:, 2:3], kernel_size=4)
    strat_soft = soft_pool2d(pseudo_targets_128[:, 1:2], kernel_size=4)
    
    # 2. 背景通道 (Class 0) 建议保持 AvgPool 或也用 SoftPool
    # 背景通常比较平滑，AvgPool 足够，用 SoftPool 也可以
    bg_avg = F.avg_pool2d(pseudo_targets_128[:, 0:1], kernel_size=4)
    
    # 3. 堆叠
    pseudo_32 = torch.cat([bg_avg, strat_soft, conv_soft], dim=1)
    
    # 4. 再次归一化 (SoftPool 的输出虽然接近 1 但不严格等于 1)
    pseudo_32 = pseudo_32 / (pseudo_32.sum(dim=1, keepdim=True) + 1e-8)
    
    return pseudo_32


def compute_class_prior_from_loader(train_loader, num_samples=1000):
    """
    从训练集统计类别先验
    Args:
        train_loader: 训练数据加载器
        num_samples: 统计样本数（避免遍历整个数据集）
    Returns:
        class_prior: [3] tensor
    """
    counts = torch.zeros(3)
    samples_processed = 0

    print(f"[开始统计] 预计处理 {num_samples} 个样本...")

    for batch in train_loader:
        if samples_processed >= num_samples:
            break

        # batch是4个值的元组: (input_norm, target_norm, target_raw, region_label)
        _, _, gt_raw, _ = batch
        pseudo_targets = generate_pseudo_soft_targets_v1_plus(gt_raw)
        hard_labels = pseudo_targets.argmax(dim=1)

        for c in range(3):
            counts[c] += (hard_labels == c).sum().item()

        samples_processed += gt_raw.size(0)

        # 每处理100个样本打印一次进度
        if samples_processed % 100 == 0:
            current_prior = counts / (counts.sum() + 1e-8)
            print(f"  已处理 {samples_processed}/{num_samples} 样本, "
                  f"当前先验: 背景={current_prior[0]:.4f}, "
                  f"层状={current_prior[1]:.4f}, 对流={current_prior[2]:.4f}")

    prior = counts / counts.sum()
    print(f"\n[统计完成] 最终类别先验:")
    print(f"  - 背景 (Expert 0): {prior[0]:.4f} ({prior[0]*100:.2f}%)")
    print(f"  - 层状 (Expert 1): {prior[1]:.4f} ({prior[1]*100:.2f}%)")
    print(f"  - 对流 (Expert 2): {prior[2]:.4f} ({prior[2]*100:.2f}%)")
    print(f"  ⚠️  如果对流占比<1%，考虑降低伪标签生成的阈值\n")

    return prior


def generate_pseudo_soft_targets_v1_plus(gt_precip):
    """
    V8 - 激进左移 + 早期接管策略
    """
    precip = gt_precip.squeeze(1)       # [B, H, W]
    precip_4d = precip.unsqueeze(1)     # [B, 1, H, W]

    # 【改进1：左移防止漏报】
    # 将中心从 0.1 改为 0.05。
    # 效果：0.1mm 处的 p_rain ≈ 0.92。Router 会果断选择 Expert 1。
    p_rain = torch.sigmoid((precip - 0.1) / 0.02)
    
    # 【改进2：大雨激进接管】
    # 强度路径 A：基础对流 (保持 10mm，用于过渡)
    p_intense_base = torch.sigmoid((precip - 10.0) / 3.0) 
    
    # 强度路径 B：极端接管 (从 20mm 提前到 15mm)
    # 效果：30mm 处 sigmoid(5.0) ≈ 0.993。强迫 Expert 2 统治高值区。
    p_intense_extreme = torch.sigmoid((precip - 20.0) / 3.0)

    # --- 纹理计算保持不变 ---
    k = 7
    local_mean = F.avg_pool2d(precip_4d, k, stride=1, padding=k//2).squeeze(1)
    local_sq_mean = F.avg_pool2d(precip_4d ** 2, k, stride=1, padding=k//2).squeeze(1)
    local_std = torch.sqrt(torch.clamp(local_sq_mean - local_mean ** 2, min=0) + 1e-6)
    cv = local_std / (local_mean + 0.1) 
    p_high_texture = torch.sigmoid((cv - 0.5) / 0.15)
    
    rain_mask = (precip > 0.1).float().unsqueeze(1)
    local_density = F.avg_pool2d(rain_mask, 11, stride=1, padding=5).squeeze(1)
    p_isolated = torch.sigmoid((0.4 - local_density) / 0.1)

    # --- 融合 ---
    path_classic = p_intense_base * p_high_texture
    path_isolated = p_intense_base * p_isolated
    path_extreme = p_intense_extreme 

    p_convective_raw = torch.max(
        torch.max(path_classic, path_isolated),
        path_extreme
    )
    
    # --- 分配 ---
    p_regime2 = p_convective_raw * p_rain
    p_regime1 = p_rain * (1.0 - p_convective_raw)
    p_regime0 = 1.0 - p_rain
    
    soft_targets = torch.stack([p_regime0, p_regime1, p_regime2], dim=1)
    soft_targets = soft_targets / (soft_targets.sum(dim=1, keepdim=True) + 1e-6)
    
    return soft_targets
