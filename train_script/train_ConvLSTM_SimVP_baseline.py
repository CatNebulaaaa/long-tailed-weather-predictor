import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse  

# === 1. 引入 AMP 模块 (加速核心) ===
from torch.cuda.amp import autocast, GradScaler

# === 2. 导入你的 Dataset ===
from dataset import GPMDataset

# === 3. 导入 OpenSTL 模型 ===
try:
    from openstl.models import SimVP_Model, ConvLSTM_Model
    print("✅ 成功导入 OpenSTL 模型！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit()

# ================= 配置区 =================
# 在这里切换模型: 'SimVP' 或 'ConvLSTM'
MODEL_NAME = 'ConvLSTM' 

DATA_PATH = '/root/autodl-tmp/processed_data_with_raw.zarr'
STATS_PATH = '/root/autodl-tmp/normalization_stats.json'
SAVE_DIR = f'/root/autodl-tmp/checkpoints/baseline_{MODEL_NAME}'

# 显存优化后，BS=64 应该很轻松，甚至可以尝试 128
BATCH_SIZE = 128  
LEARNING_RATE = 1e-3
EPOCHS = 15
NUM_WORKERS = 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据维度
PAST_FRAMES = 3
HEIGHT = 128
WIDTH = 128
CHANNELS = 1

# 评估分桶
FINAL_BUCKETS = [(0, 0.1), (0.1, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 200.0)]
FINAL_THRESHOLDS = [0.1, 1.0, 5.0, 8.0, 10.0, 20.0]
# ==========================================


def get_model_lightweight(model_name):
    """
    模型工厂：根据名字初始化 OpenSTL 模型，并补全缺失配置
    """
    FUTURE_FRAMES = 1 
    
    if model_name == 'SimVP':
        print("🚀 初始化 SimVP (Lightweight版)...")
        model = SimVP_Model(
            in_shape=(PAST_FRAMES, CHANNELS, HEIGHT, WIDTH),
            hid_S=32,
            hid_T=128,
            N_S=4,
            N_T=4,
            model_type='gsta'
        )
    elif model_name == 'ConvLSTM':
        print("🐢 初始化 ConvLSTM (终极配置版)...")
        
        num_layers = 4
        num_hidden = [32, 32, 32, 32] 
        
        # 构建伪造的 configs (补全所有 ConvLSTM 可能用到的参数)
        configs = argparse.Namespace()
        configs.in_shape = (PAST_FRAMES, CHANNELS, HEIGHT, WIDTH)
        configs.filter_size = 5
        configs.stride = 1
        configs.patch_size = 1
        configs.frame_patch_size = 1
        
        configs.layer_norm = 0
        configs.scheduled_sampling = 1
        configs.reverse_scheduled_sampling = 0
        configs.r_sampling_step_1 = 25000
        configs.r_sampling_step_2 = 50000
        configs.r_exp_alpha = 5000
        
        # 关键参数：序列长度
        configs.pre_seq_length = PAST_FRAMES
        configs.aft_seq_length = FUTURE_FRAMES
        configs.total_length = PAST_FRAMES + FUTURE_FRAMES
        
        model = ConvLSTM_Model(
            num_layers=num_layers, 
            num_hidden=num_hidden, 
            configs=configs,
            in_shape=(PAST_FRAMES, CHANNELS, HEIGHT, WIDTH) 
        )
        
    else:
        raise ValueError(f"未知模型: {model_name}")
    return model

def denormalize(tensor, mean, std):
    x = tensor * std + mean
    x = torch.expm1(x)
    return torch.clamp(x, min=0.0)

def validate_detailed(model, loader, stats, device):
    """
    验证函数：兼容 SimVP 和 ConvLSTM 的数据格式差异
    """
    model.eval()
    bucket_stats = {str(b): {'mse': 0, 'count': 0} for b in FINAL_BUCKETS}
    class_stats = {t: {'hits': 0, 'misses': 0, 'fa': 0} for t in FINAL_THRESHOLDS}
    
    total_mse, total_cnt = 0, 0
    mean, std = stats['mean'], stats['std']

    with torch.no_grad():
        for inputs, targets_norm, targets_raw in loader:
            # === 1. 数据上 GPU (关键：targets_norm 也要上，因为 ConvLSTM 需要拼接) ===
            inputs = inputs.to(device)
            targets_norm = targets_norm.to(device)
            targets_raw = targets_raw.to(device)

            # === 2. Forward (根据模型类型分流) ===
            if 'ConvLSTM' in str(type(model)):
                # ConvLSTM: Channel Last [B, T, H, W, C]
                # Inputs [B, T, H, W] -> [B, T, H, W, 1]
                if inputs.dim() == 4: x = inputs.unsqueeze(-1)
                else: x = inputs.permute(0, 1, 3, 4, 2)
                
                # Targets [B, 1, H, W] -> [B, 1, H, W, 1]
                if targets_norm.dim() == 4: y_cat = targets_norm.unsqueeze(-1)
                else: y_cat = targets_norm.permute(0, 1, 3, 4, 2)
                
                # 拼接输入和目标，满足 ConvLSTM 内部循环需求
                x_full = torch.cat([x, y_cat], dim=1)
                
                # 构造 Mask
                B, T_full, H, W, C = x_full.shape
                mask_true = torch.zeros((B, T_full - 1, H, W, C)).to(device)
                
                # 前向传播
                preds = model(x_full, mask_true=mask_true)
                
                # 处理 Tuple 返回值
                if isinstance(preds, tuple): preds = preds[0]
                
                # 取最后一帧: [B, H, W, C]
                pred_last = preds[:, -1, :, :, :]
                # 转置回 Channel First: [B, C, H, W]
                pred_last = pred_last.permute(0, 3, 1, 2)
                
            else:
                # SimVP: Channel First [B, T, C, H, W]
                if inputs.dim() == 4: x = inputs.unsqueeze(2)
                else: x = inputs
                
                preds = model(x)
                # 取最后一帧: [B, C, H, W]
                pred_last = preds[:, -1, :, :, :]

            # === 3. 计算指标 ===
            preds_raw = denormalize(pred_last, mean, std)
            
            # 确保维度匹配 [B, H, W]
            if targets_raw.dim() == 4: targets_raw = targets_raw.squeeze(1)
            # pred_last 可能有 C=1 维度，squeeze 掉
            if preds_raw.dim() == 4: preds_raw = preds_raw.squeeze(1)
            
            diff = preds_raw - targets_raw
            
            # Overall stats
            mask_valid = targets_raw >= 0
            total_mse += (diff[mask_valid]**2).sum().item()
            total_cnt += mask_valid.sum().item()

            # Bucket stats
            for b_min, b_max in FINAL_BUCKETS:
                mask = (targets_raw >= b_min) & (targets_raw < b_max)
                if mask.sum() > 0:
                    bk = str((b_min, b_max))
                    bucket_stats[bk]['mse'] += (diff[mask]**2).sum().item()
                    bucket_stats[bk]['count'] += mask.sum().item()

            # Classification stats
            for t in FINAL_THRESHOLDS:
                p_bin = preds_raw >= t
                t_bin = targets_raw >= t
                class_stats[t]['hits'] += (p_bin & t_bin).sum().item()
                class_stats[t]['misses'] += (~p_bin & t_bin).sum().item()
                class_stats[t]['fa'] += (p_bin & ~t_bin).sum().item()

    # 汇总
    res = {}
    res['Overall_RMSE'] = np.sqrt(total_mse / (total_cnt + 1e-6))
    
    b_data = []
    for b_min, b_max in FINAL_BUCKETS:
        bk = str((b_min, b_max))
        rmse = np.sqrt(bucket_stats[bk]['mse'] / (bucket_stats[bk]['count'] + 1e-6))
        b_data.append({'Range': bk, 'RMSE': rmse, 'Samples': bucket_stats[bk]['count']})
    res['bucket_df'] = pd.DataFrame(b_data)
    
    return res

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    with open(STATS_PATH, 'r') as f: stats = json.load(f)
    train_dataset = GPMDataset(DATA_PATH, 'train', stats, past_frames=PAST_FRAMES)
    val_dataset = GPMDataset(DATA_PATH, 'val', stats, past_frames=PAST_FRAMES)
    
    # 开启 pin_memory 可以加快数据传输
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 初始化模型
    model = get_model_lightweight(MODEL_NAME).to(DEVICE)
    
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() # AMP
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"\n🚀 开始训练 {MODEL_NAME} (AMP + Lightweight)...")
    
    best_ext_rmse = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        for inputs, targets_norm, _ in loop:
            inputs, targets_norm = inputs.to(DEVICE), targets_norm.to(DEVICE)
            
            # === 训练循环的数据适配 ===
            if MODEL_NAME == 'ConvLSTM':
                # Inputs: [B, T, H, W] -> [B, T, H, W, 1]
                if inputs.dim() == 4: x = inputs.unsqueeze(-1)
                else: x = inputs.permute(0, 1, 3, 4, 2)
                
                # Targets: [B, 1, H, W] -> [B, 1, H, W, 1]
                if targets_norm.dim() == 4: y_cat = targets_norm.unsqueeze(-1)
                else: y_cat = targets_norm.permute(0, 1, 3, 4, 2)
                
                # 拼接
                x_full = torch.cat([x, y_cat], dim=1)
                
                # Mask
                B, T, H, W, C = x.shape
                mask_true = torch.zeros((B, T, H, W, C)).to(DEVICE) # Mask长度为T (Input len)
                
                # 目标 Y 调整为 [B, 1, 1, H, W] 用于 Loss 计算
                y_loss = targets_norm.unsqueeze(2) if targets_norm.dim() == 4 else targets_norm
            
            else: # SimVP
                # [B, T, H, W] -> [B, T, 1, H, W]
                if inputs.dim() == 4: x = inputs.unsqueeze(2)
                else: x = inputs
                y_loss = targets_norm.unsqueeze(2) if targets_norm.dim() == 4 else targets_norm
            
            # === Forward & Backward ===
            with autocast():
                if MODEL_NAME == 'ConvLSTM':
                    preds = model(x_full, mask_true=mask_true)
                    if isinstance(preds, tuple): preds = preds[0]
                    
                    # ConvLSTM Out: [B, T, H, W, C] -> 取最后一帧 -> [B, H, W, C]
                    preds_last = preds[:, -1, :, :, :]
                    # Permute -> [B, C, H, W] -> [B, 1, C, H, W]
                    preds_last = preds_last.permute(0, 3, 1, 2).unsqueeze(1)
                else:
                    preds = model(x)
                    preds_last = preds[:, -1, :, :, :].unsqueeze(1) 
                
                loss = criterion(preds_last, y_loss)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss=loss.item())
            
        # 验证
        res = validate_detailed(model, val_loader, stats, DEVICE)
        ext_rmse = res['bucket_df'].iloc[-1]['RMSE']
        
        print(f"\n[Epoch {epoch+1}] Overall RMSE: {res['Overall_RMSE']:.4f} | Extreme RMSE (>10): {ext_rmse:.4f}")
        print(res['bucket_df'].to_string(index=False))
        
        scheduler.step(res['Overall_RMSE'])
        
        if ext_rmse < best_ext_rmse:
            best_ext_rmse = ext_rmse
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print("✅ 模型已保存 (Best Extreme RMSE)")

if __name__ == "__main__":
    main()