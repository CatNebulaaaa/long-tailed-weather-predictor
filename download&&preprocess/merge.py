import os
import xarray as xr
from glob import glob
from tqdm import tqdm
import shutil
import h5py # 需要用到 h5py 来检测文件是否损坏
from concurrent.futures import ThreadPoolExecutor

# ================= 配置区 =================
SAVE_DIR = '/root/autodl-tmp/raw_data'
OUTPUT_MERGED_FILE = '/root/autodl-tmp/merged_data.nc'
TEMP_DIR = '/root/autodl-tmp/temp_chunks'
TARGET_VARIABLE = 'precipitation'
BATCH_SIZE = 500 
# ==========================================

def preprocess(ds):
    """预处理函数"""
    if 'time' in ds.data_vars and 'time' not in ds.coords:
        ds = ds.set_coords('time')
    return ds[[TARGET_VARIABLE, 'lat', 'lon', 'time']]

def check_file_integrity(file_path):
    """
    检查单个 HDF5 文件是否完好
    返回: (file_path, is_valid)
    """
    try:
        # 尝试以只读模式打开文件，如果文件截断或损坏，这里会报错
        with h5py.File(file_path, 'r') as f:
            # 尝试访问一下根节点，确保不是空壳
            _ = f.keys()
        return file_path, True
    except Exception:
        return file_path, False

def filter_bad_files(file_list):
    """
    使用多线程快速扫描一批文件，剔除损坏的
    """
    valid_files = []
    bad_files = []
    
    # 使用线程池并发检查，速度很快
    print(f"正在检查本批次 {len(file_list)} 个文件的完整性...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(check_file_integrity, file_list)
        
        for fpath, is_valid in results:
            if is_valid:
                valid_files.append(fpath)
            else:
                bad_files.append(fpath)
    
    if bad_files:
        print(f"⚠️ 发现 {len(bad_files)} 个损坏文件，已自动跳过:")
        for bf in bad_files[:5]: # 只打印前5个名字
            print(f"  - {os.path.basename(bf)}")
        if len(bad_files) > 5:
            print(f"  - ... 等共 {len(bad_files)} 个")
            
        # 可选：如果你想自动删除坏文件，取消下面这行的注释
        # for bf in bad_files: os.remove(bf)
            
    return valid_files

def main():
    # 1. 准备目录
    output_dir = os.path.dirname(OUTPUT_MERGED_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    # 2. 获取文件列表
    file_pattern = os.path.join(SAVE_DIR, '*.HDF5.nc4')
    gpm_files = sorted(glob(file_pattern))
    
    if not gpm_files:
        print("未找到文件。")
        return

    print(f"共找到 {len(gpm_files)} 个文件。开始分批处理...")

    batches = [gpm_files[i:i + BATCH_SIZE] for i in range(0, len(gpm_files), BATCH_SIZE)]
    temp_files_list = []

    # 3. 分批处理
    for i, batch_files in enumerate(tqdm(batches, desc="分批合并进度", unit="batch")):
        try:
            # === 关键修改：在合并前先过滤坏文件 ===
            clean_batch = filter_bad_files(batch_files)
            
            if not clean_batch:
                print(f"批次 {i} 所有文件都损坏，跳过。")
                continue

            temp_file_path = os.path.join(TEMP_DIR, f"chunk_{i:03d}.nc")
            
            # 使用过滤后的 clean_batch 进行合并
            with xr.open_mfdataset(
                clean_batch,
                engine='h5netcdf',
                concat_dim='time',
                combine='nested',
                preprocess=preprocess,
                parallel=True,
                coords='minimal',
                data_vars='minimal',
                compat='override'
            ) as ds_chunk:
                
                ds_chunk.to_netcdf(
                    temp_file_path,
                    mode='w',
                    format='NETCDF4',
                    engine='h5netcdf',
                    encoding={TARGET_VARIABLE: {'zlib': True, 'complevel': 5}}
                )
            
            temp_files_list.append(temp_file_path)
            
        except Exception as e:
            print(f"\n批次 {i} 处理虽然经过过滤但依然失败: {e}")
            # 继续尝试下一个批次，不要直接退出
            continue

    print("\n所有分批处理完成，正在进行最终合并...")

    # 4. 最终合并
    if not temp_files_list:
        print("没有产生任何临时文件，合并失败。")
        return

    try:
        with xr.open_mfdataset(temp_files_list, engine='h5netcdf', concat_dim='time', combine='nested') as final_ds:
            final_ds.to_netcdf(
                OUTPUT_MERGED_FILE,
                mode='w',
                format='NETCDF4',
                engine='h5netcdf',
                encoding={TARGET_VARIABLE: {'zlib': True, 'complevel': 5}}
            )
        print("✅ 全部完成！")
        shutil.rmtree(TEMP_DIR)

    except Exception as e:
        print(f"最终合并失败: {e}")

if __name__ == "__main__":
    main()