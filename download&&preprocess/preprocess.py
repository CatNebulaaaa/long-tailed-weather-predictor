import os
import numpy as np
import xarray as xr
import json
import shutil
from tqdm import tqdm
import cv2

# ================= 配置区 =================
INPUT_FILE = '/root/autodl-tmp/merged_data.nc' 
OUTPUT_ZARR_DIR = '/root/autodl-tmp/processed_america_data_with_raw_and_labels.zarr' # 建议用新名字
STATS_FILE = '/root/autodl-tmp/america_normalization_stats.json'
TARGET_SIZE = 128
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
MAX_PRECIP = 200.0 
# 优化读取的关键：为每个变量设置独立的、优化的 chunks
CHUNKS_RAW = {'time': 100, 'lat': TARGET_SIZE, 'lon': TARGET_SIZE}
CHUNKS_LABELS = {'time': 100, 'regime': 3, 'lat': TARGET_SIZE, 'lon': TARGET_SIZE}

# V2 标签的物理参数
THRESHOLDS = {'rain': 0.1, 'convective': 15.0}
AREA_LIMITS = {'convective_max': 1000, 'stratiform_min': 1000}
# ==========================================

def generate_region_labels_for_preprocess(gt_precip_np):
    """
    一个简化的、只接受 numpy 输入的 V2 标签生成函数。
    """
    H, W = gt_precip_np.shape
    
    # === 核心优化：将 float32 修改为 uint8 ===
    # uint8 使用 1 个字节，而 float32 使用 4 个字节。
    # 对于只有 0 和 1 的掩码，uint8 足够了，并且可以节省 75% 的空间。
    hard_masks = np.zeros((3, H, W), dtype=np.uint8) # <--- 在这里修改！
    # ==========================================

    # 1. 背景 (这部分逻辑不变)
    hard_masks[0] = (gt_precip_np < THRESHOLDS['rain'])
    
    # 2. 连通域分析 (这部分逻辑不变)
    rain_mask_np = (gt_precip_np >= THRESHOLDS['rain']).astype(np.uint8)
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(rain_mask_np, 8)

    if num_labels > 1:
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            component_mask = (labels_map == label_id)
            
            if not np.any(component_mask):
                continue
            max_intensity = gt_precip_np[component_mask].max()
            
            is_convective = (area < AREA_LIMITS['convective_max'] and 
                             max_intensity > THRESHOLDS['convective'])
            is_stratiform = (area >= AREA_LIMITS['stratiform_min'])
            
            if is_convective:
                hard_masks[2][component_mask] = 1
            elif is_stratiform:
                hard_masks[1][component_mask] = 1
            else:
                hard_masks[1][component_mask] = 1
                
    return hard_masks


def process_and_save_group(data_raw, group_name, stats):
    """一个辅助函数，处理单个数据集分组并保存。"""
    print(f"\n--- 开始处理分组: {group_name} ---")
    
    # 1. 计算归一化数据 (懒加载)
    print("计算归一化数据...")
    data_norm = (np.log1p(data_raw) - stats['mean']) / (stats['std'] + 1e-6)
    
    # 2. 计算标签 (立即计算，因为需要遍历)
    print("计算区域标签...")
    # 使用 .load() 将 dask array 加载到内存中，以便 numpy 可以处理
    # 这一步会消耗内存，但对于标签计算是必要的
    raw_numpy = data_raw.load().values 
    
    labels_list = [generate_region_labels_for_preprocess(frame) for frame in tqdm(raw_numpy, desc=f"生成标签 ({group_name})")]
    labels_array = np.stack(labels_list, axis=0)
    
    # 3. 创建包含所有变量的最终 Dataset
    final_ds = xr.Dataset({
        'precipitation_norm': data_norm,
        'precipitation_raw': data_raw,
        'region_labels': xr.DataArray(
            labels_array,
            dims=('time', 'regime', 'lat', 'lon'),
            coords={'regime': [0, 1, 2]} # 只需要提供新的坐标
        )
    })
    
    # 4. 设置分块并保存
    # 确保时间、经纬度坐标也被正确写入
    final_ds = final_ds.assign_coords({
        'time': data_raw.time,
        'lat': data_raw.lat,
        'lon': data_raw.lon
    })

    print("设置分块并保存到 Zarr...")
    final_ds = final_ds.chunk({'time': 100}) # 统一设置 time chunk
    
    mode = 'w' if group_name == 'train' else 'a'
    final_ds.to_zarr(OUTPUT_ZARR_DIR, group=group_name, mode=mode, consolidated=True)
    print(f"✅ 分组 '{group_name}' 保存成功!")


def main():
    if os.path.exists(OUTPUT_ZARR_DIR):
        print(f"检测到旧目录 {OUTPUT_ZARR_DIR}，正在删除...")
        shutil.rmtree(OUTPUT_ZARR_DIR)

    print(f"正在打开数据文件夹: /root/autodl-tmp/GPM_FINAL/")
    # 使用 open_mfdataset 自动逻辑合并
    # chunks 设置建议保持，方便后续 Dask 并行计算
    ds = xr.open_mfdataset('/root/autodl-tmp/GPM_FINAL/*.nc', 
                           combine='by_coords', 
                           chunks={'time': 1000})
    da = ds['precipitation']

    print("1. 数据清洗与裁剪...")
    da = da.isel(lat=slice(0, TARGET_SIZE), lon=slice(0, TARGET_SIZE))
    da = da.sortby('time')
    da = da.where(da >= 0, 0).clip(max=MAX_PRECIP)

    print("2. 数据集划分...")
    n_samples = da.sizes['time']
    n_train = int(n_samples * TRAIN_RATIO)
    n_val = int(n_samples * VAL_RATIO)
    train_raw = da.isel(time=slice(0, n_train))
    val_raw = da.isel(time=slice(n_train, n_train + n_val))
    test_raw = da.isel(time=slice(n_train + n_val, None))

    print("3. 计算训练集统计量...")
    train_log = np.log1p(train_raw)
    mean_val = train_log.mean().compute().item()
    std_val = train_log.std().compute().item()
    stats = {"mean": mean_val, "std": std_val}
    with open(STATS_FILE, 'w') as f: json.dump(stats, f, indent=4)
    print(f"   计算完成: Mean={mean_val:.4f}, Std={std_val:.4f}")

    # 4. 依次处理并保存每个分组
    process_and_save_group(train_raw, 'train', stats)
    process_and_save_group(val_raw, 'val', stats)
    process_and_save_group(test_raw, 'test', stats)

    print(f"\n🎉🎉🎉 全部处理完成！数据已保存至 {OUTPUT_ZARR_DIR}")

if __name__ == "__main__":
    main()