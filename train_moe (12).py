import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os
import json

from models.swin_unet_moe import SwinUNetMoE
from dataset import GPMDataset
from models.regime_module import generate_pseudo_soft_targets_v1_plus

class FinetuneConfig:
    batch_size = 128
    num_epochs = 20         # 稍微多给几轮，让静默 Loss 生效
    weight_decay = 1e-4
    
    # --- 核心权重体系 ---
    # 1. Router 权重：稍微调高，确保它能跟上新的伪标签变化
    weight_gate = 10.0      
    
    # 2. 全局权重：【关键】设为 0，彻底解耦，防止被平均化
    weight_final = 5.0      
    
    # 3. 专家权重：主导训练
    weight_special = 0.0   
    
    # 4. 静默权重：【新增】专门用来压制误报
    # 如果背景区 Expert 1/2 乱动，狠罚！
    weight_silence = 0.0   
    
    # 极值特训参数
    penalty_under = 1.0    # 低估惩罚倍数 (狠一点)
    
    # 路径配置 (请确保路径正确)
    data_dir = '/root/autodl-tmp/gpm_pt_dataset'
    pretrained_path = '/root/autodl-tmp/checkpoints/new_moe/best_model_csi30.pth'
    checkpoint_dir = '/root/autodl-tmp/checkpoints/moe_finetune_v3' # 新目录 v3
    stats_path = '/root/autodl-tmp/normalization_stats.json'
    
    use_wandb = True
    wandb_project = 'swin-unet-moe-finetune-v3'

def train_one_epoch(model, train_loader, optimizer, config, epoch, device):
    model.train()
    metrics = {
        'total': 0, 'gate': 0, 'conv': 0, 'silence': 0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        input_norm, target_norm, target_raw, _ =[b.to(device) for b in batch]
        
        # 前向传播
        outputs = model(input_norm)
        
        # 生成新的阶梯式伪标签
        pseudo_targets = generate_pseudo_soft_targets_v1_plus(target_raw)
        if pseudo_targets.shape[-1] != 128:
            pseudo_targets = F.interpolate(pseudo_targets, size=(128, 128), mode='bilinear')
            
        # -----------------------------------------------------------------
        # LOSS 1: Router Loss (听指挥)
        # -----------------------------------------------------------------
        # 加大对流区的权重，确保 30mm+ 选对人
        pixel_weights = pseudo_targets[:, 0] * 1.0 + pseudo_targets[:, 1] * 2.0 + pseudo_targets[:, 2] * 5.0
        loss_gate = (F.cross_entropy(outputs['logits_128'], pseudo_targets, reduction='none') * pixel_weights).mean()
        
        # -----------------------------------------------------------------
        # LOSS 2: 专家分工 Loss (各司其职)
        # -----------------------------------------------------------------
        # 计算有效面积 (防止除零)
        area_bg = pseudo_targets[:, 0:1].sum() + 1e-6
        area_strat = pseudo_targets[:, 1:2].sum() + 1e-6
        area_conv = pseudo_targets[:, 2:3].sum() + 1e-6

        # Expert 0 (Bg): 只在背景区学习 0
        loss_bg = (F.huber_loss(outputs['pred_bg'], target_norm, reduction='none') * pseudo_targets[:, 0:1]).sum() / area_bg
        
        # Expert 1 (Strat): 只在层状云区学习
        loss_strat = (F.huber_loss(outputs['pred_strat'], target_norm, reduction='none') * pseudo_targets[:, 1:2]).sum() / area_strat
        
        # Expert 2 (Conv): 【核心】不对称 MSE + 强度加权
        mse_conv = (outputs['pred_conv'] - target_norm) ** 2
        asym_weight = torch.where(outputs['pred_conv'] < target_norm, config.penalty_under, 1.0)
        intensity_weight = 1.0 + (target_raw / 10.0) # 越大的雨权重越大
        loss_conv = (mse_conv * asym_weight * intensity_weight * pseudo_targets[:, 2:3]).sum() / area_conv

        # -----------------------------------------------------------------
        # LOSS 3: 交叉抑制 Loss (【新增】为了追上 SimVP)
        # -----------------------------------------------------------------
        # 逻辑：在背景区 (pseudo_bg > 0.8)，Expert 1 和 Expert 2 必须闭嘴 (输出0)
        # 这能大幅降低误报 (FAR)
        bg_mask = (pseudo_targets[:, 0:1] > 0.8).float()
        bg_area = bg_mask.sum() + 1e-6
        
        # 强迫 Expert 1 在背景区输出 0
        silence_strat = (outputs['pred_strat'] ** 2 * bg_mask).sum() / bg_area
        # 强迫 Expert 2 在背景区输出 0
        silence_conv = (outputs['pred_conv'] ** 2 * bg_mask).sum() / bg_area
        
        loss_silence = silence_strat + silence_conv

        # -----------------------------------------------------------------
        # 总 Loss
        # -----------------------------------------------------------------
        total_loss = (config.weight_gate * loss_gate + 
                      config.weight_special * (loss_bg + loss_strat + loss_conv) + 
                      config.weight_silence * loss_silence)
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        metrics['total'] += total_loss.item()
        metrics['gate'] += loss_gate.item()
        metrics['conv'] += loss_conv.item()
        metrics['silence'] += loss_silence.item()
        
        pbar.set_postfix({
            'T': f"{total_loss.item():.1f}",
            'Conv': f"{loss_conv.item():.2f}",
            'Silence': f"{loss_silence.item():.2f}" # 监控这个，看误报是否被压制
        })
        
    return {k: v / len(train_loader) for k, v in metrics.items()}

def validate(model, val_loader, device, stats):
    model.eval()
    thresholds =[0.1, 1.0, 10.0, 30.0, 50.0]
    metrics_stats = {t: {'tp': 0, 'fn': 0, 'fp': 0} for t in thresholds}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            input_norm, _, target_raw, _ = [b.to(device) for b in batch]
            outputs = model(input_norm)
            
            # 反归一化
            pred_phys = torch.clamp(torch.expm1(outputs['final_pred'] * stats['std'] + stats['mean']), min=0.0).view(-1)
            gt_flat = target_raw.view(-1)
            
            for t in thresholds:
                pred_m, gt_m = (pred_phys >= t), (gt_flat >= t)
                metrics_stats[t]['tp'] += (pred_m & gt_m).sum().item()
                metrics_stats[t]['fn'] += (~pred_m & gt_m).sum().item()
                metrics_stats[t]['fp'] += (pred_m & ~gt_m).sum().item()
                
    results = {}
    for t in thresholds:
        tp, fn, fp = metrics_stats[t]['tp'], metrics_stats[t]['fn'], metrics_stats[t]['fp']
        results[f'csi_{t}'] = tp / (tp + fn + fp + 1e-8)
        print(f"  > Threshold {t}mm: CSI={results[f'csi_{t}']:.4f}")
    return results

def main():
    config = FinetuneConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    if config.use_wandb: wandb.init(project=config.wandb_project, config=config.__dict__)

    # 加载数据
    train_loader = DataLoader(GPMDataset(config.data_dir, group='train'), batch_size=config.batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(GPMDataset(config.data_dir, group='val'), batch_size=config.batch_size, shuffle=False, num_workers=12)
    with open(config.stats_path, 'r') as f: stats = json.load(f)
    
    # 模型初始化
    model = SwinUNetMoE(img_size=128, patch_size=4, in_chans=3, num_regimes=3, stats_path=config.stats_path).to(device)
    
    # -------------------------------------------------------
    # 权重加载：严格模式 False (因为我们要用新 Expert 2)
    # -------------------------------------------------------
    print(f"🔥 正在加载基座权重: {config.pretrained_path}")
    checkpoint = torch.load(config.pretrained_path, map_location=device)
    model_dict = model.state_dict()
    # 过滤不匹配的 key (新 Expert 2 参数将被跳过)
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"✅ 加载完成。新层 Expert 2 & Head 将从头训练。")
    
    # -------------------------------------------------------
    # 优化器分组：新层大火猛炒，旧层小火慢炖
    # -------------------------------------------------------
    new_params = []
    base_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        # 只要名字里带 expert_conv 或 head_conv 就视为新层
        if 'expert_conv' in n or 'head_conv' in n:
            new_params.append(p)
        else:
            base_params.append(p)
            
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': 1e-5}, # 极低学习率保护 Encoder/Router
        {'params': new_params,  'lr': 2e-4}  # 高学习率训练新 Expert 2
    ], weight_decay=config.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
    
    # 训练循环
    best_score = 0.0
    for epoch in range(1, config.num_epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, config, epoch, device)
        val_m = validate(model, val_loader, device, stats)
        
        print(f"\n✅ Epoch {epoch} | T:{train_m['total']:.1f} | Conv:{train_m['conv']:.2f} | Silence:{train_m['silence']:.2f}")
        
        # 综合评分：关注高端极值，兼顾低端误报
        # score = CSI-30 + CSI-50 + CSI-0.1
        current_score = val_m['csi_30.0'] + val_m['csi_50.0'] + val_m['csi_0.1']
        
        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, 'best_finetuned_v3.pth'))
            print(f"💾 模型保存! Score: {best_score:.4f}")
            
        if config.use_wandb:
            wandb.log({**train_m, **val_m}, step=epoch)
        scheduler.step()

if __name__ == '__main__':
    main()
