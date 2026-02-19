# convert_zarr_to_pt.py
import xarray as xr
import torch
from tqdm import tqdm
import os

# --- 配置 ---
ZARR_PATH = '/root/autodl-tmp/processed_data_with_raw_and_labels.zarr'
OUTPUT_DIR = '/root/autodl-tmp/gpm_pt_dataset' # 新数据集的根目录
PAST_FRAMES = 3
# ------------

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for group in ['train', 'val', 'test']:
        print(f"\n--- Converting group: {group} ---")
        
        group_dir = os.path.join(OUTPUT_DIR, group)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

        ds = xr.open_zarr(ZARR_PATH, group=group)
        
        # 加载预计算好的索引
        indices_path = f'/root/autodl-tmp/indices/{group}_valid_indices.npy'
        valid_indices = np.load(indices_path)
        
        # 为了更快的读取，我们先把整个 dataarray 加载到内存
        # 注意：这会消耗大量内存！如果内存不足，请告诉我，有替代方案。
        print("Loading data into memory for faster processing...")
        data_norm = ds['precipitation_norm'].load()
        data_raw = ds['precipitation_raw'].load()
        region_labels = ds['region_labels'].load()
        print("Data loaded.")

        for i in tqdm(range(len(valid_indices)), desc=f"Exporting {group}"):
            start_idx = valid_indices[i]
            target_idx = start_idx + PAST_FRAMES

            # 提取数据
            input_norm = torch.from_numpy(data_norm[start_idx : start_idx + PAST_FRAMES].values)
            target_norm = torch.from_numpy(data_norm[target_idx].values).unsqueeze(0)
            target_raw = torch.from_numpy(data_raw[target_idx].values).unsqueeze(0)
            region_label = torch.from_numpy(region_labels[target_idx].values)
            
            # 将所有数据打包成一个字典并保存
            sample = {
                'input_norm': input_norm,
                'target_norm': target_norm,
                'target_raw': target_raw,
                'region_label': region_label
            }
            
            torch.save(sample, os.path.join(group_dir, f'{i:06d}.pt'))

    print("\n🎉 Conversion complete!")

if __name__ == "__main__":
    import numpy as np # 脚本内需要 numpy
    main()