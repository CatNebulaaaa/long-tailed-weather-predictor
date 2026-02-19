import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import GPMDataset
from models import SwinUNet
from losses import WeightedL1Loss

# ================= 配置区 =================
DATA_PATH = '/root/autodl-tmp/processed_data_with_raw.zarr'
STATS_PATH = '/root/autodl-tmp/normalization_stats.json'
SAVE_DIR = '/root/autodl-tmp/checkpoints/weighted_baseline'
LOG_DIR = 'runs/weighted_baseline'

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_WORKERS = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAST_FRAMES = 3

# 评估指标
EXTREME_BUCKET = (10.0, 200.0)
HEAVY_RAIN_CSI_THRESHOLD = 10.0
# 最终报告分桶
FINAL_BUCKETS = [(0, 0.1), (0.1, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 200.0)]
FINAL_THRESHOLDS = [0.1, 1.0, 5.0, 10.0, 20.0]
# ==========================================

def denormalize(tensor, mean, std):
    x = tensor * std + mean
    x = torch.expm1(x)
    return torch.clamp(x, min=0.0)

def colorize(tensor, cmap_name='jet', vmin=0, vmax=20):
    """
    将单通道灰度 Tensor 转换为三通道彩色 Tensor。
    Args:
        tensor (torch.Tensor): 输入 Tensor, shape [1, H, W]
        cmap_name (str): Matplotlib colormap. 'jet' 是气象常用.
        vmin, vmax: 归一化的范围.
    Returns:
        torch.Tensor: 彩色 Tensor, shape [3, H, W]
    """
    if tensor.dim() == 4: # B, C, H, W
        tensor = tensor.squeeze(0)
    if tensor.dim() == 2: # H, W
        tensor = tensor.unsqueeze(0)

    # 1. 归一化到 [0, 1]
    tensor_norm = tensor.detach().cpu().numpy().squeeze()
    tensor_norm = np.clip((tensor_norm - vmin) / (vmax - vmin), 0, 1)
    
    # 2. 获取 colormap
    cmap = cm.get_cmap(cmap_name)
    
    # 3. 应用 colormap (返回 RGBA)
    colored_array = cmap(tensor_norm)
    
    # 4. 转换为 PyTorch Tensor 并丢弃 Alpha 通道
    # (H, W, 4) -> (H, W, 3) -> (3, H, W)
    colored_tensor = torch.from_numpy(colored_array[:, :, :3]).permute(2, 0, 1)
    
    return colored_tensor.float()

def validate_epoch(model, loader, criterion, stats, device):
    model.eval()
    val_loss = 0
    total_mse, total_pixels = 0, 0
    extreme_mse, extreme_pixels = 0, 0
    hits, misses, false_alarms = 0, 0, 0
    max_gt_val = -1
    best_batch_tensors = None
    mean, std = stats['mean'], stats['std']

    with torch.no_grad():
        for inputs, targets_norm, targets_raw in loader:
            inputs, targets_norm, targets_raw = inputs.to(device), targets_norm.to(device), targets_raw.to(device)
            preds_norm = model(inputs)
            loss = criterion(preds_norm, targets_norm)
            val_loss += loss.item()
            preds_raw = denormalize(preds_norm, mean, std)
            
            mask_valid = targets_raw >= 0
            diff = preds_raw - targets_raw
            total_mse += (diff[mask_valid] ** 2).sum().item()
            total_pixels += mask_valid.sum().item()

            b_min, b_max = EXTREME_BUCKET
            mask_extreme = (targets_raw >= b_min) & (targets_raw < b_max)
            extreme_pixels_batch = mask_extreme.sum().item()
            if extreme_pixels_batch > 0:
                extreme_mse += (diff[mask_extreme] ** 2).sum().item()
                extreme_pixels += extreme_pixels_batch
            
            t = HEAVY_RAIN_CSI_THRESHOLD
            pred_bool, target_bool = preds_raw >= t, targets_raw >= t
            hits += (pred_bool & target_bool).sum().item()
            misses += (~pred_bool & target_bool).sum().item()
            false_alarms += (pred_bool & ~target_bool).sum().item()

            current_max = targets_raw.max().item()
            if current_max > max_gt_val:
                max_gt_val = current_max
                idx_in_batch = torch.argmax(torch.max(targets_raw.view(targets_raw.size(0), -1), dim=1)[0]).item()
                best_batch_tensors = (inputs[idx_in_batch].cpu(), preds_raw[idx_in_batch].cpu(), targets_raw[idx_in_batch].cpu())

    avg_val_loss = val_loss / len(loader)
    overall_rmse = np.sqrt(total_mse / (total_pixels + 1e-6))
    extreme_rmse = np.sqrt(extreme_mse / (extreme_pixels + 1e-6)) if extreme_pixels > 0 else 0
    csi = hits / (hits + misses + false_alarms + 1e-6)
    
    metrics = {'loss': avg_val_loss, 'rmse_overall': overall_rmse, 'rmse_extreme': extreme_rmse, 'csi_heavy_rain': csi}
    return metrics, best_batch_tensors

def final_evaluation(model, loader, stats, device):
    model.eval()
    bucket_errors = {str(b): {'mse': 0, 'mae': 0, 'count': 0} for b in FINAL_BUCKETS}
    threshold_metrics = {t: {'hits': 0, 'misses': 0, 'false_alarms': 0} for t in FINAL_THRESHOLDS}
    total_mse, total_pixels = 0, 0
    mean, std = stats['mean'], stats['std']

    print("Running Final Detailed Evaluation...")
    with torch.no_grad():
        for inputs, _, targets_raw in tqdm(loader, desc="Final Evaluating"):
            inputs, targets_raw = inputs.to(device), targets_raw.to(device)
            preds_norm = model(inputs)
            preds_raw = denormalize(preds_norm, mean, std)
            mask_valid = targets_raw >= 0
            diff = preds_raw - targets_raw
            total_mse += (diff[mask_valid] ** 2).sum().item()
            total_pixels += mask_valid.sum().item()
            for b_min, b_max in FINAL_BUCKETS:
                mask = (targets_raw >= b_min) & (targets_raw < b_max)
                count = mask.sum().item()
                if count > 0:
                    bk = str((b_min, b_max))
                    bucket_errors[bk]['mse'] += (diff[mask] ** 2).sum().item()
                    bucket_errors[bk]['mae'] += diff[mask].abs().sum().item()
                    bucket_errors[bk]['count'] += count
            for t in FINAL_THRESHOLDS:
                pred_bool, target_bool = preds_raw >= t, targets_raw >= t
                threshold_metrics[t]['hits'] += (pred_bool & target_bool).sum().item()
                threshold_metrics[t]['misses'] += (~pred_bool & target_bool).sum().item()
                threshold_metrics[t]['false_alarms'] += (pred_bool & ~target_bool).sum().item()
    
    results = {}
    results['Overall_RMSE'] = np.sqrt(total_mse / (total_pixels + 1e-6))
    bucket_list = [{'Range': k, 'RMSE': np.sqrt(v['mse']/(v['count']+1e-6)), 'MAE': v['mae']/(v['count']+1e-6), 'Samples': int(v['count'])} for k, v in bucket_errors.items()]
    results['bucket_df'] = pd.DataFrame(bucket_list)
    csi_list = [{'Threshold': t, 'CSI': v['hits']/(v['hits']+v['misses']+v['false_alarms']+1e-6), 'POD': v['hits']/(v['hits']+v['misses']+1e-6), 'FAR': v['false_alarms']/(v['hits']+v['false_alarms']+1e-6)} for t, v in threshold_metrics.items()]
    results['csi_df'] = pd.DataFrame(csi_list)
    return results

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    writer = SummaryWriter(LOG_DIR)
    
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)

    train_dataset = GPMDataset(DATA_PATH, 'train', stats, past_frames=PAST_FRAMES)
    val_dataset = GPMDataset(DATA_PATH, 'val', stats, past_frames=PAST_FRAMES) 

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Initializing Swin-UNet with input channels = {PAST_FRAMES}...")
    model = SwinUNet(img_size=128, in_chans=PAST_FRAMES, num_classes=1, window_size=4).to(DEVICE)
    
    print("Using Weighted L1 Loss for long-tail regression.")
    criterion = WeightedL1Loss(
        thresholds=[0.1, 5.0, 10.0], 
        weights=[1, 5, 20, 50]
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    
    print(f"Start Training for {EPOCHS} epochs on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for inputs, targets_norm in loop:
            inputs, targets_norm = inputs.to(DEVICE), targets_norm.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets_norm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        val_metrics, viz_tensors = validate_epoch(model, val_loader, criterion, stats, DEVICE)
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', val_metrics['loss'], epoch)
        writer.add_scalar('RMSE/overall', val_metrics['rmse_overall'], epoch)
        writer.add_scalar('RMSE/extreme_bucket', val_metrics['rmse_extreme'], epoch)
        writer.add_scalar('CSI/heavy_rain', val_metrics['csi_heavy_rain'], epoch)
        
        if viz_tensors:
            # viz_tensors = (input [3,H,W], pred [1,H,W], gt [1,H,W])
            # 我们只看最后一帧输入 (t时刻)
            img_input_raw = denormalize(viz_tensors[0][-1].unsqueeze(0), stats['mean'], stats['std'])
            img_pred = viz_tensors[1]
            img_gt = viz_tensors[2]
            
            # 定义统一的色标范围
            viz_max = 20.0
            
            # 使用 colorize 函数进行上色
            color_input = colorize(img_input_raw, vmax=viz_max)
            color_pred = colorize(img_pred, vmax=viz_max)
            color_gt = colorize(img_gt, vmax=viz_max)
            
            # 制作网格 (Input | Prediction | Ground Truth)
            grid = torchvision.utils.make_grid([color_input, color_pred, color_gt], nrow=3)
            writer.add_image('Validation/Input_Pred_GT_Color', grid, epoch)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['loss'])
        new_lr = optimizer.param_groups[0]['lr']
        lr_status = f" -> LR reduced to {new_lr:.1e}" if new_lr < old_lr else ""

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.5f} | Val RMSE={val_metrics['rmse_overall']:.4f} | Extreme RMSE={val_metrics['rmse_extreme']:.4f}{lr_status}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            
    writer.close()
    print("\n" + "="*50)
    print("Training Finished! Loading Best Model for Final Detailed Evaluation...")
    print("="*50)
    
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    final_metrics = final_evaluation(model, val_loader, stats, DEVICE)
    
    print(f"\n[Final Result] Overall RMSE: {final_metrics['Overall_RMSE']:.4f} mm/hr")
    print("\n--- Bucket Error Analysis (Evidence for your Paper) ---")
    print(final_metrics['bucket_df'].to_string(index=False))
    print("\n--- Classification Metrics (CSI/POD/FAR) ---")
    print(final_metrics['csi_df'].to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()