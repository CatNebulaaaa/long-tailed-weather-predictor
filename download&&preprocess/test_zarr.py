import xarray as xr
import time
import numpy as np

# --- 请修改这里的路径 ---
DATA_PATH = '/root/autodl-tmp/processed_data_with_raw.zarr'
GROUP = 'train'
# -------------------------

print("="*50)
print(f"开始诊断 Zarr 文件: {DATA_PATH}, 分组: {GROUP}")
print("="*50)

try:
    # 步骤 1: 仅打开数据集，不加载任何数据
    print("\n[测试 1/4] 尝试打开数据集...")
    t0 = time.time()
    ds = xr.open_zarr(DATA_PATH, group=GROUP, chunks=None)
    t1 = time.time()
    print(f"✅ 成功打开！耗时: {t1 - t0:.4f} 秒。")
    print(f"数据集概览:\n{ds}\n")

    # 步骤 2: 尝试加载时间坐标到内存
    print("[测试 2/4] 尝试将 'time' 坐标加载到内存...")
    t0 = time.time()
    time_values = ds.time.values  # <--- 这是最关键的嫌疑犯
    t1 = time.time()
    print(f"✅ 'time' 坐标加载成功！耗时: {t1 - t0:.4f} 秒。")
    print(f"    -> 形状: {time_values.shape}, 类型: {time_values.dtype}")
    
    # 步骤 3: 尝试加载一小块 `region_labels` 数据
    print("\n[测试 3/4] 尝试加载一小块 'region_labels' 数据...")
    t0 = time.time()
    # 加载第一个时间点，所有通道，前10x10个像素
    label_slice = ds['region_labels'][0, :, :10, :10].values
    t1 = time.time()
    print(f"✅ 'region_labels' 切片加载成功！耗时: {t1 - t0:.4f} 秒。")
    print(f"    -> 形状: {label_slice.shape}, 类型: {label_slice.dtype}")

    # 步骤 4: 尝试加载一小块 `precipitation_raw` 数据
    print("\n[测试 4/4] 尝试加载一小块 'precipitation_raw' 数据...")
    t0 = time.time()
    # 加载第一个时间点，前10x10个像素
    raw_slice = ds['precipitation_raw'][0, :10, :10].values
    t1 = time.time()
    print(f"✅ 'precipitation_raw' 切片加载成功！耗时: {t1 - t0:.4f} 秒。")
    print(f"    -> 形状: {raw_slice.shape}, 类型: {raw_slice.dtype}")

    print("\n🎉 诊断完成！所有基本读取操作都很快。问题可能更复杂。")

except Exception as e:
    print(f"\n❌ 在某个步骤中发生错误: {e}")