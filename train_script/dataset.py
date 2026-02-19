# import torch
# from torch.utils.data import Dataset
# import xarray as xr
# import numpy as np
# import pandas as pd
# import datetime
# import cftime

# class GPMDataset(Dataset):
#     def __init__(self, zarr_path, group, stats=None, past_frames=1, expected_interval_min=30):
#         super().__init__()
#         self.group = group
#         self.past_frames = past_frames
        
#         # 1. 打开 Zarr 数据集 (这一步很快)
#         self.ds = xr.open_zarr(zarr_path, group=group)
#         self.data_norm = self.ds['precipitation_norm']
#         self.data_raw = self.ds['precipitation_raw']
#         self.region_labels = self.ds['region_labels']
        
#         print(f"[{group}] 正在高效处理时间坐标...")

#         # === 终极性能补丁：向量化提取时间分量 ===
#         # 这个方法不依赖 Pandas 的转换，直接使用 xarray 的能力，
#         # 即使对于 cftime.DatetimeJulian 对象，它也是高效的。

#         # 1. 使用 xarray 的 .dt 访问器一次性提取所有时间分量为 NumPy 数组。
#         #    这是完全向量化的，对于 11 万个时间点也是毫秒级操作。
#         time_da = self.ds.time
#         df = pd.DataFrame({
#             'year': time_da.dt.year.values,
#             'month': time_da.dt.month.values,
#             'day': time_da.dt.day.values,
#             'hour': time_da.dt.hour.values,
#             'minute': time_da.dt.minute.values,
#             'second': time_da.dt.second.values
#         })

#         # 2. 使用 pd.to_datetime 将包含分量的 DataFrame 高效地转换为标准的 DatetimeIndex。
#         #    这是 pandas 中最高效的构建时间序列的方式之一。
#         times_series = pd.to_datetime(df)
#         # ===============================================
        
#         # 3. 后续的连续性检查逻辑保持不变
#         seq_len = past_frames + 1
#         total_minutes = (seq_len - 1) * expected_interval_min
        
#         # .to_series() 是为了方便使用 .shift()
#         diffs = times_series.shift(-(seq_len - 1)) - times_series
#         diffs_in_minutes = diffs.dt.total_seconds() / 60.0
        
#         valid_mask = np.abs(diffs_in_minutes - total_minutes) < 2.0
#         valid_mask = valid_mask.fillna(False)
#         self.valid_indices = np.where(valid_mask.to_numpy())[0]
        
#         total_possible = len(self.ds.time)
#         print(f"[{group}] 可用序列: {len(self.valid_indices)} / {total_possible}")
        
#     def __len__(self):
#         return len(self.valid_indices)

#     def __getitem__(self, idx):
#         start_idx = self.valid_indices[idx]
#         target_idx = start_idx + self.past_frames
        
#         input_norm = torch.tensor(self.data_norm[start_idx : start_idx + self.past_frames].values, dtype=torch.float32) 
#         target_norm = torch.tensor(self.data_norm[target_idx].values, dtype=torch.float32).unsqueeze(0)
#         target_raw = torch.tensor(self.data_raw[target_idx].values, dtype=torch.float32).unsqueeze(0)
        
#         # === 核心修改：返回预处理好的标签 ===
#         if self.region_labels is not None:
#             region_label = torch.tensor(self.region_labels[target_idx].values, dtype=torch.float32)
#         else:
#             # 如果没有，返回一个占位符，避免报错
#             region_label = torch.zeros((3, target_raw.shape[2], target_raw.shape[3]), dtype=torch.float32)

#         # 返回 4 个值
#         return input_norm, target_norm, target_raw, region_label


# dataset.py (最终高性能版)
import torch
from torch.utils.data import Dataset
import os
import glob

class GPMDataset(Dataset):
    def __init__(self, pt_data_path, group, **kwargs):
        super().__init__()
        # pt_data_path 应该是像 '/root/autodl-tmp/gpm_pt_dataset' 这样的路径
        self.group_path = os.path.join(pt_data_path, group)
        
        # 获取所有 .pt 文件的路径
        self.file_paths = sorted(glob.glob(os.path.join(self.group_path, '*.pt')))
        
        print(f"[{group}] Found {len(self.file_paths)} pre-processed samples.")
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 加载单个文件，速度极快！
        sample = torch.load(self.file_paths[idx])
        
        return (
            sample['input_norm'], 
            sample['target_norm'], 
            sample['target_raw'], 
            sample['region_label']
        )