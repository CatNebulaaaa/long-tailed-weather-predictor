# precompute_indices.py
import xarray as xr
import pandas as pd
import numpy as np
import os

# --- 配置 ---
ZARR_PATH = '/root/autodl-tmp/processed_america_data_with_raw_and_labels.zarr' # 确认这是你的 Zarr 文件路径
OUTPUT_DIR = '/root/autodl-tmp/america_indices' # 保存索引文件的目录
PAST_FRAMES = 3
EXPECTED_INTERVAL_MIN = 30
# ------------

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for group in ['train', 'val', 'test']:
        print(f"--- Processing group: {group} ---")
        ds = xr.open_zarr(ZARR_PATH, group=group)

        # 使用我们最终确认的高效方法来转换时间
        time_da = ds.time
        df = pd.DataFrame({
            'year': time_da.dt.year.values,
            'month': time_da.dt.month.values,
            'day': time_da.dt.day.values,
            'hour': time_da.dt.hour.values,
            'minute': time_da.dt.minute.values,
            'second': time_da.dt.second.values
        })
        times_series = pd.to_datetime(df)

        # 计算连续性
        seq_len = PAST_FRAMES + 1
        total_minutes = (seq_len - 1) * EXPECTED_INTERVAL_MIN
        diffs = times_series.shift(-(seq_len - 1)) - times_series
        diffs_in_minutes = diffs.dt.total_seconds() / 60.0
        valid_mask = np.abs(diffs_in_minutes - total_minutes) < 2.0
        valid_mask = valid_mask.fillna(False)
        valid_indices = np.where(valid_mask.to_numpy())[0]

        # 保存结果
        save_path = os.path.join(OUTPUT_DIR, f'{group}_valid_indices.npy')
        np.save(save_path, valid_indices)
        print(f"✅ Saved {len(valid_indices)} indices for '{group}' to {save_path}")

if __name__ == "__main__":
    main()