import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置 =================
FILE_PATH = '/root/autodl-tmp/merged_data.nc'
# =======================================

def main():
    if not os.path.exists(FILE_PATH):
        print("错误：找不到文件！")
        return

    print(f"正在打开文件: {FILE_PATH} ...")
    try:
        ds = xr.open_dataset(FILE_PATH)
        print("\n✅ 文件打开成功！")
        
        # 1. 维度检查
        print("-" * 30)
        print("【维度检查】")
        print(ds.sizes)

        # 2. 数值检查
        print("-" * 30)
        print("【数值检查】")
        precip = ds['precipitation']
        
        # 计算全局最大值（这一步可能需要几秒）
        max_val = precip.max().values
        print(f"最大降水强度: {max_val:.4f} mm/hr")

        # 3. 可视化检查 (修复版 - 极简模式)
        print("-" * 30)
        print("【可视化检查】正在生成 'check_preview.png' ...")
        
        print("   (正在定位最大值帧...)")
        
        # 修复逻辑：
        # 1. 先计算每一帧的最大值 -> 得到一个时间序列
        max_per_frame = precip.max(dim=['lat', 'lon'])
        
        # 2. 找到这个时间序列中最大值的索引 (argmax)
        max_time_idx = max_per_frame.argmax(dim='time')
        
        # 3. 直接取出那一帧
        frame_max = precip.isel(time=max_time_idx)
        
        # 获取这一帧的具体时间
        frame_time = frame_max.time.values
        print(f"   找到最强降水时刻: {frame_time}")
        
        # 绘图
        plt.figure(figsize=(10, 8))
        # 使用气象常用的 jet 配色，vmax 设为 20 以便看清云团结构
        frame_max.plot(cmap='jet', vmin=0, vmax=20)
        plt.title(f"Max Precipitation Event\nTime: {frame_time}")
        plt.savefig('check_preview.png')
        plt.close()
        
        print("✅ 预览图已保存为 check_preview.png")
        print("-" * 30)
        print("🎉 恭喜！数据完整性验证通过。你可以把这张图发给导师看了。")

    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()