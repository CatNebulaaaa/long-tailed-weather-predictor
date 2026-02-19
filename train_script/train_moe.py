# train_moe_spatial.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd # 新增依赖

from dataset import GPMDataset
from models.swin_unet_moe import SwinUNetMoE
# 导入 V3 标签函数 (请确保 models/regime_module.py 中有此函数)
# from models.regime_module import generate_pseudo_soft_targets_v3 as generate_pseudo_soft_targets
from models.regime_module import generate_pseudo_soft_targets_v1_plus as generate_pseudo_soft_targets
from models.regime_module import generate_pixel_labels

# ================= 配置区 =================
DATA_PATH = '/root/autodl-tmp/gpm_pt_dataset'
STATS_PATH = '/root/autodl-tmp/normalization_stats.json'
SAVE_DIR = '/root/autodl-tmp/checkpoints/hcr_v1_plus_pro_attention_v2.11_fine_tuning' 
BASELINE_PATH = '/root/autodl-tmp/checkpoints/weighted_baseline/best_model.pth' 
resume_path = '/root/autodl-tmp/checkpoints/hcr_v1_plus_pro_attention_v2.11/best_extreme_model.pth' 
LOG_DIR = 'runs/hcr_v1_plus_pro_attention_v2.11_fine_tuning'

BATCH_SIZE = 256
LEARNING_RATE = 2e-6
EPOCHS = 70
NUM_WORKERS = 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAST_FRAMES = 3

# === Loss 权重 ===
# FINAL_LOSS_WEIGHT = 0.1       # 降! 给分类 Loss 让路
# ROUTER_LOSS_WEIGHT = 0.5 
# SPECIALIZED_LOSS_WEIGHT = 1.0 
# GDL_LOSS_WEIGHT = 1.0         # 降! 2.0 有点喧宾夺主
# FINAL_EX_WEIGHT = 0.5 
# DICE_WEIGHT = 1.0             # 保持主力
# ENTROPY_WEIGHT = 0.5          # 升! (从0.2提到0.5) 强迫 Router 站队
# FOCAL_WEIGHT = 10.0           # 新增! Focal数值小，给大权重
# === 修正后的建议权重 ===
# 在配置区修改权重
# === 修正后的建议权重 (完全替换你代码中对应的部分) ===
FINAL_LOSS_WEIGHT = 5.0        
ROUTER_LOSS_WEIGHT = 20.0      # 保持
DICE_WEIGHT = 5.0              # 维持 Dice 优化形状
FINAL_EX_WEIGHT = 5.0         # 保持 40.0，这是把 RMSE 从 6.5 打下来的关键
BIAS_PENALTY_WEIGHT = 100.0    
SPECIALIZED_LOSS_WEIGHT = 10.0 
GDL_LOSS_WEIGHT = 20.0         
FFT_WEIGHT = 0.0002            
ENTROPY_WEIGHT = 0.0           # 不加
FOCAL_WEIGHT = 0.0     

# 建议：背景设低，暴雨设高
lambda_0 = 2.0  # 背景
lambda_1 = 1.0  # 层状云
lambda_2 = 10.0 # 暴雨 (极值最难学，给最大权重)

GRAD_CLIP = 1.0 
WARMUP_EPOCHS = 0

# === 评估指标 ===
EXTREME_BUCKET = (10.0, 200.0)
CSI_THRESHOLD = 10.0 # 训练过程中监控 CSI 的阈值
# 最终测试的分桶和阈值
FINAL_BUCKETS = [(0, 0.1), (0.1, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 200.0)]
FINAL_THRESHOLDS = [0.1, 1.0, 5.0, 10.0, 20.0]

# 加载预计算的 IB 权重
ib_data = torch.load('ib_weights.pt')
# 必须转到 GPU
IB_BIN_EDGES = ib_data['bin_edges'].to(DEVICE)
IB_WEIGHTS = ib_data['weights'].to(DEVICE)
# ==========================================
def ib_weighted_mse(pred_norm, target_norm, target_raw, mask=None):
    """
    【软化版 IB Loss】
    对查表得到的权重开根号，防止梯度爆炸。
    """
    # 1. 查表 (保持不变)
    indices = torch.bucketize(target_raw, IB_BIN_EDGES, right=False) - 1
    indices = torch.clamp(indices, min=0, max=len(IB_WEIGHTS)-1)
    weights_map = IB_WEIGHTS[indices]
    
    # 2. 🔥 核心修改：权重软化 (Square Root)
    # 原始权重范围: 1 ~ 100
    # 软化后范围: 1 ~ 10
    weights_map = torch.sqrt(weights_map)
    
    # 3. 计算加权 MSE (保持不变)
    diff_sq = (pred_norm - target_norm) ** 2
    weighted_diff = diff_sq * weights_map
    
    if mask is not None:
        loss = (weighted_diff * mask).sum() / (mask.sum() + 1e-6)
    else:
        loss = weighted_diff.mean()
        
    return loss

def asymmetric_weighted_mse(pred, target, under_penalty=20.0):
    """
    非对称加权 MSE：
    1. 动态权重：数值越大，基础权重越大 (针对 50mm)
    2. 非对称惩罚：如果预测值 < 真值 (漏报)，惩罚翻倍！
    """
    diff = pred - target
    
    # 1. 基础权重：针对大值区域 (同之前)
    # 50mm -> 基础权重 25
    base_weight = torch.clamp((target / 10.0) ** 2, min=1.0)
    
    # 2. 非对称惩罚 (Asymmetric Penalty)
    # 找出“漏报”的像素 (pred < target)
    under_prediction_mask = (diff < 0).float()
    
    # 如果漏报，权重再乘以 under_penalty (比如 20倍)
    # 如果过报，权重保持为 1
    # 逻辑：宁可过报，不可漏报！
    asymmetric_w = 1.0 + (under_penalty - 1.0) * under_prediction_mask
    
    # 最终权重 = 基础权重 * 非对称权重
    final_weight = base_weight * asymmetric_w
    
    loss = (diff ** 2 * final_weight).mean()
    return loss


def make_weighted_sampler(dataset):
    print("⚖️ 正在扫描数据集以计算采样权重 (这可能需要几分钟)...")
    weights = []
    
    # 为了加快速度，如果你有预先存好的 metadata 最好
    # 这里我们遍历 dataset。如果太慢，可以尝试只遍历一部分
    for i in tqdm(range(len(dataset))):
        # 假设 dataset[i] 返回 (input, norm, raw, label)
        # 我们只需要 raw target (index 2)
        try:
            # 快速读取：只读 tensor，不进行 transform (如果 dataset 支持)
            # 这里直接调用 getitem 可能会慢，但最稳妥
            _, _, target_raw, _ = dataset[i]
            max_val = target_raw.max().item()
        except:
            max_val = 0.0
            
        # 🔥 权重分配策略：让暴雨出现的概率翻倍再翻倍
        if max_val >= 50.0:
            w = 20.0  # 超级暴雨：20倍关注度！
        elif max_val >= 30.0:
            w = 10.0  # 暴雨
        elif max_val >= 10.0:
            w = 3.0   # 大雨
        elif max_val >= 1.0:
            w = 1.0   # 普通雨
        else:
            w = 0.6   # 无雨/毛毛雨：大幅降权，少看点
            
        weights.append(w)
        
    weights = torch.tensor(weights).double()
    # Replacement=True 是必须的，允许同一个暴雨样本在一个 Batch 里出现多次
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

def fft_loss(pred, target):
    """
    高频加权增强版 FFT Loss
    专门解决 PSD 尾部掉落和画面模糊问题
    """
    pred_fft = torch.fft.rfft2(pred)
    target_fft = torch.fft.rfft2(target)
    diff = torch.abs(pred_fft - target_fft)

    B, C, H, W = pred.shape
    u = torch.fft.fftfreq(H).to(pred.device)
    v = torch.fft.rfftfreq(W).to(pred.device)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
    dist = torch.sqrt(u_grid**2 + v_grid**2)
    
    # 🔥 核心修改：不再是线性 dist，而是给高频部分指数级的压力
    # 只有当 dist > 0.4 (即进入 PSD 掉落的高频区) 时，权重才会陡增
    # 这会强迫模型去关注那些丢失的锐利纹理
    weight = 1.0 + torch.pow(dist * 10.0, 3) 
    
    return (diff * weight).mean()
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.flatten()
        targets = targets.flatten()
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean': return loss.mean()
        else: return loss.sum()

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: 模型的原始输出 (经过反归一化后的 raw precipitation)
        # targets: 二值化的真值 (0 或 1)
        
        # 为了可导，我们使用 Sigmoid 近似阶跃函数，或者直接处理概率
        # 这里我们假设输入已经是经过阈值处理的“概率似然”
        
        # 展平
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), -1)
            targets = targets.view(targets.size(0), -1)
        
        intersection = (logits * targets).sum(dim=1)
        union = logits.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# 辅助函数：将连续的降水数值转化为“是强降水的概率”
def get_heavy_rain_probability(precip_raw, threshold=10.0, temperature=5.0):
    # 使用带温度的 Sigmoid 来模拟从 0 到 1 的硬截断
    # 当 precip_raw = 10.0 时，值为 0.5
    # temperature 越小，转换越陡峭（越接近 0/1）
    return torch.sigmoid((precip_raw - threshold) * temperature)

class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: softmax 后的概率 [B, K, H, W]
        b = x * torch.log(x + 1e-8)
        b = -1.0 * b.sum(dim=1)
        return b.mean()

def denormalize(tensor, mean, std):
    x = tensor * std + mean
    x = torch.expm1(x)
    return torch.clamp(x, min=0.0)

def colorize(tensor, cmap_name='jet', vmin=0, vmax=20):
    if tensor.dim() == 4: tensor = tensor.squeeze(0)
    if tensor.dim() == 2: tensor = tensor.unsqueeze(0)
    tensor_norm = np.clip((tensor.detach().cpu().numpy().squeeze() - vmin) / (vmax - vmin), 0, 1)
    cmap = matplotlib.colormaps[cmap_name]
    colored_array = cmap(tensor_norm)
    return torch.from_numpy(colored_array[:, :, :3]).permute(2, 0, 1).float()

class GDLLoss(nn.Module):
    def __init__(self):
        super(GDLLoss, self).__init__()
    def forward(self, pred, target):
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        return torch.mean(torch.abs(pred_dx - target_dx)) + torch.mean(torch.abs(pred_dy - target_dy))


def compute_expert_loss(pred, target, mask, weight_map=None, delta=1.0, gdl_coeff=0.1):
    """
    统一损失函数：Weighted Huber Loss + GDL
    delta: 控制 MSE 和 L1 的切换点。
           delta 越大 -> 越像 MSE (利于降 RMSE)
           delta 越小 -> 越像 L1  (利于保边缘/CSI)
    """
    # 1. 计算 Masked Huber Loss
    # reduction='none' 让我们能应用 mask
    loss_huber_raw = F.huber_loss(pred, target, reduction='none', delta=delta)
    
    # 应用权重 (Weight Map) 和 Mask
    if weight_map is not None:
        loss_main = (loss_huber_raw * weight_map * mask).sum() / (mask.sum() + 1e-6)
    else:
        loss_main = (loss_huber_raw * mask).sum() / (mask.sum() + 1e-6)
    
    # 2. 计算 GDL Loss (梯度损失)
    # GDL 也是为了保边缘，暴雨专家给高一点，层状云给低一点
    # 注意：criterion_gdl 需要在外部定义或者这里实例化
    # 建议直接把 criterion_gdl 传进来，或者在这里简单算一下
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    
    loss_gdl = torch.mean(torch.abs(pred_dx - target_dx) * mask[:, :, :, 1:]) + \
               torch.mean(torch.abs(pred_dy - target_dy) * mask[:, :, 1:, :])
    
    return loss_main + gdl_coeff * loss_gdl

# === 增强版验证函数：返回 RMSE, ExtremeRMSE, CSI 和 可视化张量 ===
# === 全局超参数设置 ===
BG_THRESHOLD = 0.7  # 背景抑制阈值：如果 Router 认为背景概率 > 0.7，强制清零
CSI_THRESHOLD = 10.0 # 你的核心评测阈值

def validate_epoch_detailed(model, loader, stats, device):
    model.eval()
    total_rmse_sq = 0; count_total = 0
    ext_rmse_sq = 0; count_ext = 0
    
    # 指标统计
    hits, misses, false_alarms = 0, 0, 0
    
    max_gt_val = -1
    best_batch_tensors = None
    mean, std = stats['mean'], stats['std']
    
    with torch.no_grad():
        for inputs, targets_norm, targets_raw, _ in loader:
            inputs = inputs.to(device)
            targets_norm = targets_norm.to(device)
            targets_raw = targets_raw.to(device)
            
            # === 核心修改点 ===
            # 模型现在返回 4 个值，我们只需要 第1个(预测) 和 第3个(概率)
            # 使用 _ 忽略不需要的 logits 和 expert_stack
            final_pred_norm, _, router_probs, _ = model(inputs)
            
            # === 背景抑制逻辑 (保持不变) ===
            # 1. 取出背景专家概率 (Channel 0)
            prob_bg = router_probs[:, 0:1, :, :] 
            # 2. 生成 Mask: 如果 Router 认为背景概率 > 0.7，则强制将预测置为 0
            # 这能有效消除因为 Router 犹豫不决导致的“方块底噪”
            suppression_mask = (prob_bg < BG_THRESHOLD).float()
            # 3. 执行抑制
            final_pred_norm = final_pred_norm * suppression_mask
            # =================
            
            # 反归一化
            preds_raw = denormalize(final_pred_norm, mean, std)
            # 物理常识：降水强度低于 0.1mm/h 在气象观测中通常视为无雨
            preds_raw[preds_raw < 0.1] = 0.0
            
            # 1. 数值指标 (RMSE)
            diff = preds_raw - targets_raw
            mask_valid = targets_raw >= 0
            total_rmse_sq += (diff[mask_valid]**2).sum().item()
            count_total += mask_valid.sum().item()
            
            mask_ext = targets_raw >= 10.0
            if mask_ext.sum() > 0:
                ext_rmse_sq += (diff[mask_ext]**2).sum().item()
                count_ext += mask_ext.sum().item()
            
            # 2. 分类指标 (@CSI_THRESHOLD, 通常是 10mm)
            # 这里的 preds_raw 已经是被抑制过的干净结果了
            pred_bin = preds_raw >= CSI_THRESHOLD
            target_bin = targets_raw >= CSI_THRESHOLD
            
            hits += (pred_bin & target_bin).sum().item()
            misses += (~pred_bin & target_bin).sum().item()
            false_alarms += (pred_bin & ~target_bin).sum().item()
            
            # 3. 采样用于可视化
            current_max = targets_raw.max().item()
            if current_max > max_gt_val:
                max_gt_val = current_max
                # 找一个最大值最猛的样本
                batch_max_vals = targets_raw.view(targets_raw.size(0), -1).max(dim=1)[0]
                idx = torch.argmax(batch_max_vals).item()
                
                # 保存可视化数据 (inputs, pred, gt, probs)
                best_batch_tensors = (inputs[idx].cpu(), preds_raw[idx].cpu(), targets_raw[idx].cpu(), router_probs[idx].cpu())
    
    # 计算最终指标
    rmse = np.sqrt(total_rmse_sq / (count_total + 1e-6))
    ext_rmse = np.sqrt(ext_rmse_sq / (count_ext + 1e-6)) if count_ext > 0 else 0
    
    # 核心分类指标公式
    eps = 1e-6
    csi = hits / (hits + misses + false_alarms + eps)
    pod = hits / (hits + misses + eps)
    far = false_alarms / (hits + false_alarms + eps)
    bias = (hits + false_alarms) / (hits + misses + eps)
    
    # 以字典形式返回，方便主循环记录
    metrics = {
        'rmse': rmse,
        'ext_rmse': ext_rmse,
        'csi': csi,
        'pod': pod,
        'far': far,
        'bias': bias
    }
    
    return metrics, best_batch_tensors

# === 最终全面测试函数 ===
def final_test_pipeline(model, loader, stats, device, output_dir, prefix='final'):
    model.eval()
    print(f"\n🚀 Running Final Detailed Test ({prefix}) with Hybrid Gating...")
    
    all_preds = []
    all_targets = []
    
    mean, std = stats['mean'], stats['std']

    with torch.no_grad():
        for inputs, _, targets_raw, _ in tqdm(loader, desc=f"Collecting {prefix}"):
            inputs = inputs.to(device)
            targets_raw = targets_raw.to(device) 
            
            # === 核心修改点 ===
            # 解包 4 个返回值
            final_pred_norm, _, router_probs, _ = model(inputs, hard_routing=False)
            
            # === 背景抑制逻辑 ===
            prob_bg = router_probs[:, 0:1, :, :] 
            suppression_mask = (prob_bg < BG_THRESHOLD).float()
            final_pred_norm = final_pred_norm * suppression_mask
            # =================
            
            preds_raw = denormalize(final_pred_norm, mean, std)
            # 物理常识：降水强度低于 0.1mm/h 在气象观测中通常视为无雨
            preds_raw[preds_raw < 0.1] = 0.0
            
            if preds_raw.dim() == 4: preds_raw = preds_raw.squeeze(1)
            if targets_raw.dim() == 4: targets_raw = targets_raw.squeeze(1)
            
            # 转到 CPU 节省显存
            all_preds.append(preds_raw.cpu())
            all_targets.append(targets_raw.cpu())

    # 拼接
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 1. 计算基础指标 (Standard)
    print(">>> Calculating Standard Metrics (Threshold=10.0)...")
    pred_bin = (all_preds >= 10.0)
    target_bin = (all_targets >= 10.0)
    hits = (pred_bin & target_bin).sum().item()
    misses = (~pred_bin & target_bin).sum().item()
    false_alarms = (pred_bin & ~target_bin).sum().item()
    
    std_csi = hits / (hits + misses + false_alarms + 1e-6)
    std_far = false_alarms / (hits + false_alarms + 1e-6)
    print(f"✅ Standard CSI (Hybrid Gating): {std_csi:.4f}")
    print(f"✅ Standard FAR (Hybrid Gating): {std_far:.4f}")

    # 2. 搜索最优阈值 (Optional)
    print("\n>>> Searching for Optimal Post-Processing Thresholds...")
    best_t = 10.0
    best_csi = 0.0
    
    # 在 5.0 到 25.0 之间搜索
    search_range = torch.arange(5.0, 25.0, 0.5)
    
    for t in search_range:
        pred_bin = (all_preds >= t)
        # 目标永远是 10.0，我们在找哪个预测阈值最匹配 10.0 的真值
        target_bin = (all_targets >= 10.0) 
        
        hits = (pred_bin & target_bin).sum().item()
        misses = (~pred_bin & target_bin).sum().item()
        false_alarms = (pred_bin & ~target_bin).sum().item()
        csi = hits / (hits + misses + false_alarms + 1e-6)
        
        if csi > best_csi:
            best_csi = csi
            best_t = t.item()
            
    print(f"🌟 Optimal Threshold found: {best_t:.2f} (maps to GT 10.0)")
    print(f"🌟 Best Calibrated CSI: {best_csi:.4f}")

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    writer = SummaryWriter(LOG_DIR)
    with open(STATS_PATH, 'r') as f: stats = json.load(f)

    # Dataset
    train_dataset = GPMDataset(DATA_PATH, 'train')
    val_dataset = GPMDataset(DATA_PATH, 'val')
    # 增加 Test Dataset
    test_dataset = GPMDataset(DATA_PATH, 'test')
    
    # sampler = make_weighted_sampler(train_dataset)
    
    # 🔥 2. 修改 DataLoader
    # train_loader = DataLoader(
    #     train_dataset, 
    #     batch_size=BATCH_SIZE, 
    #     sampler=sampler,      # <--- 注入灵魂
    #     shuffle=False,        # 使用 sampler 时必须关掉 shuffle
    #     num_workers=NUM_WORKERS,
    #     pin_memory=True
    # )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,        # 恢复 Shuffle
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Initializing SwinUNetMoE...")
    model = SwinUNetMoE(img_size=128, in_chans=PAST_FRAMES, num_regimes=3, window_size=4).to(DEVICE)


    if os.path.exists(resume_path):
        print(f"🔥🔥🔥 [Resume] Loading best checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        
        # 过滤掉形状不匹配的层 (主要是 router.up1 和 router_upsampler)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # 加载匹配的权重 (Encoder 和 Expert 不变，保留之前的训练成果)
        model.load_state_dict(pretrained_dict, strict=False)
        print(f">>> Partially restored! Router upsampling layers will be re-initialized.")
        # print(">>> 💉 RE-INJECTING BIAS for Expert 2 Rescue...")
        # with torch.no_grad():
        #     # 获取 Expert 2 的最后一层
        #     layer_2 = model.experts[2].up_final
        #     if isinstance(layer_2, nn.Sequential): layer_2 = layer_2[-1]
            
        #     # 现在的 bias 可能已经变成负数或者很小了
        #     current_bias = layer_2.bias.data.mean().item()
        #     print(f"    Current Exp2 Bias: {current_bias:.4f}")
            
        #     # 强行加码！设为 3.0 (比之前的 1.0 更激进，但比 5.0 安全)
        #     # 这会给 Expert 2 一个巨大的向上初速度
        #     layer_2.bias.data.fill_(3.0) 
        #     print(f"    New Exp2 Bias: 3.0")

    # if os.path.exists(BASELINE_PATH):
    #     print(f"🔥🔥🔥 [Strategy: Upcycling] Loading Strong Baseline: {BASELINE_PATH}")
    #     baseline_state = torch.load(BASELINE_PATH, map_location=DEVICE)
    #     model_dict = model.state_dict()
    #     new_state_dict = {}
        
    #     # 关键词匹配: 将 Baseline Decoder 复制给 3 个 Experts
    #     decoder_keywords = ['layers_up', 'concat_back_dim', 'norm_up', 'up_final']
        
    #     for k, v in baseline_state.items():
    #         if any(keyword in k for keyword in decoder_keywords):
    #             for i in range(3): 
    #                 expert_key = f"experts.{i}.{k}"
    #                 if expert_key in model_dict and model_dict[expert_key].shape == v.shape:
    #                     new_state_dict[expert_key] = v
    #         else:
    #             if k in model_dict and model_dict[k].shape == v.shape:
    #                 new_state_dict[k] = v
        
    #     keys = model.load_state_dict(new_state_dict, strict=False)
    #     print(f">>> Weights cloned successfully! Missing keys (Expected only Router): {len(keys.missing_keys)}")

    # print(">>> 🛠️ Initializing Router Upsampler...")
    # # 对 router_upsampler 进行初始化，使其初始状态接近于“平滑插值”，避免冷启动震荡
    # for m in model.router_upsampler.modules():
    #     if isinstance(m, nn.ConvTranspose2d):
    #         nn.init.xavier_normal_(m.weight) # ✅ 使用 Xavier 初始化，简单且有效
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)

    # =========================================================
    # 🔥【最终修正方案】温和偏置 + 专家冻结 (Gentle Bias + Freeze)
    # =========================================================
    # print("\n>>> 🧊 STRATEGY: Gentle Bias + Router Warmup...")
    
    # with torch.no_grad():
    #     # --- 1. 温和偏置 (Gentle Bias) ---
    #     # 不要设太大，防止 Router 因为背景误差太大而抛弃 Expert 2
        
    #     # Expert 0 (背景): 设为 -1.0 (足够压制噪声)
    #     layer_0 = model.experts[0].up_final
    #     if isinstance(layer_0, nn.Sequential): layer_0 = layer_0[-1]
    #     layer_0.bias.data.fill_(-1.0) 
    #     print(f"    [Bias] Exp0 set to -1.0")

    #     # Expert 2 (暴雨): 设为 +1.0 (起步价，告诉 Router 它是正值即可)
    #     # 只要比 Expert 0 (-1.0) 大，Router 就会倾向于选它处理降水
    #     layer_2 = model.experts[2].up_final
    #     if isinstance(layer_2, nn.Sequential): layer_2 = layer_2[-1]
    #     layer_2.bias.data.fill_(1.0) 
    #     print(f"    [Bias] Exp2 set to +1.0")
        
    #     # Expert 1 (层状): 设为 0.0 (中间态)
    #     layer_1 = model.experts[1].up_final
    #     if isinstance(layer_1, nn.Sequential): layer_1 = layer_1[-1]
    #     layer_1.bias.data.fill_(0.0)

    # --- 2. 冻结专家 (Freeze Experts) ---
    # 关键一步！前几个 Epoch 严禁专家乱动，只准 Router 学习如何分配！
    # print(">>> 🔒 FREEZING all Experts. Training Router ONLY.")
    # for name, param in model.experts.named_parameters():
    #     param.requires_grad = False
    
    # 此时优化器里应该只有 Router 和 Encoder 的参数
    # =========================================================
    
    print(">>> 🔒 Permanently Freezing Expert 0 (Background) parameters...")
    for param in model.experts[0].parameters():
        param.requires_grad = False
        
    # 定义优化器 (此时 Expert 0 已被排除)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    criterion_l1 = nn.L1Loss()
    criterion_gdl = GDLLoss()
    criterion_entropy = EntropyLoss()
    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)
    # 在 main 函数开头
    criterion_dice = SoftDiceLoss()
    criterion_router = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 15.0], device=DEVICE))

    best_extreme_rmse = float('inf')
    best_csi = 0.0

    for epoch in range(EPOCHS):
        # === 动态解冻逻辑 ===
        # 在第 3 个 Epoch (即 index=3，实际上是第4轮，或者 index=2 第3轮) 解冻
        # 建议设为 3，给 Router 足够的时间看清 4x4 的特征
        # if epoch == 3: 
        #     print("\n" + "="*40)
        #     print(f">>> 🔥 Epoch {epoch+1}: UNFREEZING EXPERTS! Full Training Starts!")
        #     print("="*40 + "\n")
            
        #     # 1. 解冻所有参数
        #     for param in model.experts.parameters():
        #         param.requires_grad = True
            
        #     # 2. 重置优化器 (重要！否则优化器不知道要更新新参数)
        #     # 注意：这里需要重新把所有参数加进去
        #     optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        #     # 如果有 scheduler 也要重置一下或者不做处理(会延续之前的lr)
        # ====================
        model.train()
        meter = {
            'total': 0, 'router': 0, 'spec': 0, 'gdl': 0, 'final': 0, 'final_ex': 0, 'dice': 0,'bias': 0, 'fft': 0,
            'prob_0': 0, 'prob_1': 0, 'prob_2': 0,
            'out_0': 0, 'out_1': 0, 'out_2': 0
        }
        steps = 0
    
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)
        
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups: pg['lr'] = warmup_lr

        for inputs, targets_norm, targets_raw, region_labels in loop:
            inputs, targets_norm, targets_raw, region_labels = \
                inputs.to(DEVICE), targets_norm.to(DEVICE), targets_raw.to(DEVICE), region_labels.to(DEVICE)
            if targets_norm.dim() == 3: targets_norm = targets_norm.unsqueeze(1)
            
            # Forward
            final_pred_norm, router_logits_128, router_probs, expert_preds_stack = model(inputs)
            
            # --- 1. Router Loss (全高清闭环版) ---
            
            # 🔥【不再需要 F.interpolate】直接使用模型生成的 logits
            # 这能保证梯度直接作用于反卷积层，让它学会锐化边缘
            # router_log_probs = F.log_softmax(router_logits_128, dim=1)
            
            # 生成伪标签
            static_targets = generate_pseudo_soft_targets(targets_raw, area_kernel_size=5)
            
            # # 计算 KL 散度
            # log_target = torch.log(static_targets.clamp(min=1e-8))
            # kl_raw = static_targets * (log_target - router_log_probs)
            
            # # 权重 (保持你的配置)
            # weights = torch.tensor([10.0, 1.0, 5.0], device=DEVICE).view(1, 3, 1, 1)
            # loss_router = (kl_raw * weights).sum() / (inputs.shape[0] * 128 * 128)


            router_labels = torch.argmax(static_targets, dim=1) 
            loss_router = criterion_router(router_logits_128, router_labels)

            # --- 2. Specialized Loss (Huber 版) ---
            mask_0 = (targets_raw < 0.1).float()
            mask_1 = ((targets_raw >= 0.1) & (targets_raw < 10.0)).float()
            mask_2 = (targets_raw >= 10.0).float()
            
            # Expert 0: 背景专家
            # Delta = 0.5: 主要是平滑去噪，不需要太锐利
            loss_e0 = compute_expert_loss(
                expert_preds_stack[:, 0:1], targets_norm, mask_0, 
                weight_map=None, 
                delta=0.5,       # 偏向 L1 一点点，为了彻底归零
                gdl_coeff=0.0    # 背景不需要学梯度，省点计算量
            )
            
            # Expert 1: 层状云专家 (降 RMSE 的主力军！)
            # Delta = 3.0: 几乎全都是 MSE 行为。
            # 只要误差小于 3.0 (归一化数值)，它就是平方惩罚。这会疯狂压低 RMSE。
            loss_e1 = compute_expert_loss(
                expert_preds_stack[:, 1:2], targets_norm, mask_1, 
                weight_map=None, 
                delta=0.5,       # 🔥 核心：大 Delta = MSE 模式 = 降 RMSE
                gdl_coeff=0.5    # 给一点点梯度约束，防止太平滑
            )
            
            # Expert 2: 暴雨专家 (保 CSI 的特种兵)
            # Delta = 0.1: 几乎全都是 L1 行为。
            # 只要误差超过 0.1，就是线性惩罚。这允许它大胆预测极值，不怕偏离。
            # energy_weight = torch.pow(targets_raw, 0.5) 
            loss_e2 = compute_expert_loss(
                expert_preds_stack[:, 2:3], targets_norm, mask_2, 
                weight_map=None, 
                delta=3.0,       # 🔥 核心：小 Delta = L1 模式 = 保极值
                gdl_coeff=5.0    # 🔥 核心：强 GDL，强迫画出锐利边缘
            )

            # 别忘了更新 loss_spec 变量
            loss_spec = (lambda_0 * loss_e0) + (lambda_1 * loss_e1) + (lambda_2 * loss_e2)

            # --- 3. Final & GDL Loss ---
            loss_final = F.huber_loss(final_pred_norm, targets_norm, delta=1.0)
            loss_gdl = criterion_gdl(final_pred_norm, targets_norm)
            
            mask_ex = (targets_raw >= 10.0).float()
            loss_final_ex = 0.0
            if mask_ex.sum() > 0:
                 # 只对强降水区做 MSE
                 loss_final_ex = F.mse_loss(final_pred_norm * mask_ex, targets_norm * mask_ex)

            # 3. 🔥 50mm 专项加权
            # 如果 batch 里有 >30mm 的点，额外加重
            mask_30 = (targets_raw >= 30.0).float()
            loss_super_ex = 0.0
            if mask_30.sum() > 0:
                 loss_super_ex = F.mse_loss(final_pred_norm * mask_30, targets_norm * mask_30)

            # 2. 🔥 背景数值清洗 (MSE)
            # 强迫最终输出为 0
            mask_bg = (targets_raw < 0.1).float()
            loss_bg = 0.0
            if mask_bg.sum() > 0:
                loss_bg = F.mse_loss(final_pred_norm * mask_bg, targets_norm * mask_bg)

            # 3. 🔥 带缓冲区的 Router 抑制 (Buffer Zone Suppression) 🔥
            
            # A. 定义真实的雨区
            rain_mask = (targets_raw >= 0.1).float()
            
            # B. 制造缓冲区：对雨区进行膨胀 (Dilate)
            # 使用 MaxPool2d 模拟膨胀，kernel=5, padding=2 意味着向外扩张 2 个像素
            # 这样，雨团边缘 2 个像素内的背景都会被标记为 1，受到保护
            rain_mask_dilated = F.max_pool2d(rain_mask, kernel_size=5, stride=1, padding=2)
            
            # C. 定义“纯净背景”：既不是雨，也不是雨的边缘
            # 只有在这里，我们才惩罚 Router 的胡乱活跃
            pure_bg_mask = (1.0 - rain_mask_dilated)
            
            loss_router_bg = 0.0
            if pure_bg_mask.sum() > 0:
                # 取出非背景专家的概率
                prob_non_bg = router_probs[:, 1:, :, :].sum(dim=1)
                
                # 只在纯净背景区，且概率 > 0.1 的地方进行惩罚
                bad_bg_mask = pure_bg_mask * (prob_non_bg > 0.1).float()
                
                if bad_bg_mask.sum() > 0:
                    loss_router_bg = (prob_non_bg * bad_bg_mask).sum() / (bad_bg_mask.sum() + 1e-6)

            # === 【核心新增】Dice Loss ===
            # 我们直接优化 10mm 阈值的 CSI
            # 1. 反归一化预测值
            preds_raw = denormalize(final_pred_norm, stats['mean'], stats['std'])
            
            # 2. 生成软概率 (为了可导)
            # 温度设为 2.0，让梯度能传回去，但又足够陡峭
            pred_prob_10mm = get_heavy_rain_probability(preds_raw, threshold=10.0, temperature=2.0)
            
            # 3. 生成二值化真值
            target_bin_10mm = (targets_raw >= 10.0).float()
            
            # 4. 计算 Dice Loss
            loss_dice = criterion_dice(pred_prob_10mm, target_bin_10mm)
            loss_entropy = criterion_entropy(router_probs)

            # --- Total ---
            loss_focal = criterion_focal(pred_prob_10mm, target_bin_10mm)

            # 计算预测概率和真值的全图平均活跃度（即面积）
            # 1. 动态计算归一化空间下的 10mm 阈值 (根据你的 stats 自动匹配)
            # 公式: (log1p(10.0) - mean) / std
            norm_threshold_10mm = (np.log1p(10.0) - stats['mean']) / stats['std']

            # 2. 计算预测和真值在 10mm 以上的平均“面积” (概率空间)
            # pred_prob_10mm 是经过 Sigmoid 的 (0~1)，其均值代表预报的强降水面积占比
            pred_area = torch.mean(pred_prob_10mm)
            # true_area 使用归一化后的阈值判断真值的强降水面积占比
            true_area = torch.mean((targets_norm > norm_threshold_10mm).float())
            
            # 🔥 Bias Penalty (修正版)
            norm_threshold_10mm = (np.log1p(10.0) - stats['mean']) / stats['std']
            pred_area = torch.mean(pred_prob_10mm)
            true_area = torch.mean((targets_norm > norm_threshold_10mm).float())
            
            # 🔥 核心调整：放宽上限至 1.1 (允许 10% 的过报)
            # 只有当 Bias > 1.1 时，excess 才是正数，惩罚才生效
            # 这能保护 POD，同时防止 Bias 失控 (比如飙到 1.3)
            target_limit = true_area * 1.15 
            
            excess = torch.relu(pred_area - target_limit)
            bias_penalty = excess / (true_area + 0.01)
            
            loss_fft = fft_loss(final_pred_norm, targets_norm)

            # 4. 汇总总损失
            total_loss = FINAL_LOSS_WEIGHT * loss_final + \
                         FINAL_EX_WEIGHT * loss_final_ex + \
                         ROUTER_LOSS_WEIGHT * loss_router + \
                         SPECIALIZED_LOSS_WEIGHT * loss_spec + \
                         GDL_LOSS_WEIGHT * loss_gdl + \
                         DICE_WEIGHT * loss_dice + \
                         BIAS_PENALTY_WEIGHT * bias_penalty + \
                         FFT_WEIGHT * loss_fft + \
                         (FINAL_EX_WEIGHT * 2.0) * loss_super_ex 
                         # FINAL_BG_WEIGHT * loss_bg + \
                         # ROUTER_BG_WEIGHT * loss_router_bg # 给大权重！
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            # === Record ===
            with torch.no_grad():
                meter['total'] += total_loss.item()
                meter['router'] += loss_router.item()
                meter['spec'] += loss_spec.item()
                meter['gdl'] += loss_gdl.item()
                meter['final'] += loss_final.item()
                meter['final_ex'] += loss_final_ex.item() if isinstance(loss_final_ex, torch.Tensor) else loss_final_ex
                meter['dice'] += loss_dice.item()
                meter['bias'] += bias_penalty.item()
                meter['fft'] += loss_fft.item()
                
                probs_avg = router_probs.mean(dim=(0, 2, 3))
                meter['prob_0'] += probs_avg[0].item(); meter['prob_1'] += probs_avg[1].item(); meter['prob_2'] += probs_avg[2].item()
                
                ex_out = denormalize(expert_preds_stack, stats['mean'], stats['std'])
                out_avg = ex_out.mean(dim=(0, 2, 3))
                meter['out_0'] += out_avg[0].item(); meter['out_1'] += out_avg[1].item(); meter['out_2'] += out_avg[2].item()
                steps += 1
            
            loop.set_postfix(loss=total_loss.item())

        # === TensorBoard ===
        for k in meter: meter[k] /= steps
        
        writer.add_scalars('Loss/Components', {'Total': meter['total'], 'Router': meter['router'], 'Spec': meter['spec'], 'GDL': meter['gdl'], 'Final_Ex': meter['final_ex'], 'Dice': meter['dice'], 'Bias': meter['bias'], 'Fft': meter['fft']},  epoch)
        writer.add_scalars('Analysis/Router_Prob', {'Exp0': meter['prob_0'], 'Exp1': meter['prob_1'], 'Exp2': meter['prob_2']}, epoch)
        writer.add_scalars('Analysis/Expert_Out', {'Exp0': meter['out_0'], 'Exp1': meter['out_1'], 'Exp2': meter['out_2']}, epoch)

        # Validation (带 CSI 和 可视化)
        # --- Validation ---
        val_metrics, viz_tensors = validate_epoch_detailed(model, val_loader, stats, DEVICE)
        
        # 记录到 TensorBoard
        writer.add_scalar('Metric/Overall_RMSE', val_metrics['rmse'], epoch)
        writer.add_scalar('Metric/Extreme_RMSE', val_metrics['ext_rmse'], epoch)
        writer.add_scalar('Metric/CSI_10', val_metrics['csi'], epoch)
        writer.add_scalar('Metric/POD_10', val_metrics['pod'], epoch)
        writer.add_scalar('Metric/FAR_10', val_metrics['far'], epoch)
        writer.add_scalar('Metric/BIAS_10', val_metrics['bias'], epoch)
        
        # 控制台打印输出，方便观察趋势
        print(f"Ep {epoch+1}: RMSE={val_metrics['rmse']:.4f} | CSI={val_metrics['csi']:.4f} | "
              f"POD={val_metrics['pod']:.4f} | FAR={val_metrics['far']:.4f} | BIAS={val_metrics['bias']:.4f}")
        
        # 2. 可视化
        if viz_tensors:
            inp, pred, gt, r_probs = viz_tensors
            img_input = denormalize(inp[-1].unsqueeze(0), stats['mean'], stats['std'])
            vmax = 20.0
            grid_img = torchvision.utils.make_grid([colorize(img_input, vmax=vmax), colorize(pred, vmax=vmax), colorize(gt, vmax=vmax)], nrow=3)
            writer.add_image('Vis/Comparison', grid_img, epoch)
            grid_router = torchvision.utils.make_grid(r_probs, nrow=3, normalize=True)
            writer.add_image('Vis/Router', grid_router, epoch)
        
        # 3. 控制台打印 (修复原来的 NameError)
        print(f"Ep {epoch+1}: Loss={meter['total']:.4f} | RMSE={val_metrics['rmse']:.4f} | "
              f"CSI={val_metrics['csi']:.4f} | POD={val_metrics['pod']:.4f} | "
              f"FAR={val_metrics['far']:.4f} | BIAS={val_metrics['bias']:.4f}")
        
        # 4. 更新学习率
        scheduler.step(val_metrics['rmse'])

        # 5. 保存最优模型
        # 判断 Extreme RMSE
        if val_metrics['ext_rmse'] < best_extreme_rmse:
            best_extreme_rmse = val_metrics['ext_rmse']
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_extreme_model.pth'))
            print(f"🔥 Saved Best Extreme Model: {best_extreme_rmse:.4f}")

        # 判断 CSI
        if val_metrics['csi'] > best_csi:
            best_csi = val_metrics['csi']
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_csi_model.pth'))
            print(f"🌊 Saved Best CSI Model: {best_csi:.4f}")

    writer.close()
    
    # 1. 测试 RMSE 最好的模型
    if os.path.exists(os.path.join(SAVE_DIR, 'best_extreme_model.pth')):
        print(">>> Testing Best Extreme RMSE Model...")
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_extreme_model.pth')))
        # 注意这里要把所有参数传进去
        final_test_pipeline(model, test_loader, stats, DEVICE, SAVE_DIR, prefix='rmse_best')

    # 2. 测试 CSI 最好的模型
    if os.path.exists(os.path.join(SAVE_DIR, 'best_csi_model.pth')):
        print(">>> Testing Best CSI Model...")
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_csi_model.pth')))
        # 注意这里要把所有参数传进去
        final_test_pipeline(model, test_loader, stats, DEVICE, SAVE_DIR, prefix='csi_best')


if __name__ == "__main__":
    main()