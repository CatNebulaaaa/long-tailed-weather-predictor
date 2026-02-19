import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
# 1. 你的链接文件路径
LINKS_FILE = 'links.txt' 

# 2. 存储路径（强烈建议使用 /root/autodl-tmp/ 下的文件夹，空间大）
SAVE_DIR = '/root/autodl-tmp/raw_data'

# 3. 并发线程数
# 下载 15 万个小文件，建议开启 10-15。如果报错太多就调小到 5。
MAX_WORKERS = 12 

# ==========================================

def download_one_file(url, session):
    """单个文件下载逻辑"""
    filename = url.split('/')[-1].split('?')[0]
    filepath = os.path.join(SAVE_DIR, filename)

    # 1. 断点续传逻辑：如果文件已存在且大小正常（>1KB），跳过
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
        return f"SKIP: {filename}"

    try:
        # 2. 发起请求
        # 注意：allow_redirects=True 必须开启，NASA 会重定向授权
        # proxies=None 表示直接使用系统环境变量中的学术加速
        with session.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            # 如果返回 401，通常是 .netrc 没配置对
            if r.status_code == 401:
                return f"AUTH_ERROR: 请检查 ~/.netrc 文件配置"
            
            r.raise_for_status()
            
            # 3. 写入文件
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=16384): # 16KB 缓存块
                    if chunk:
                        f.write(chunk)
        return f"DONE: {filename}"
    
    except Exception as e:
        # 下载失败则删除残余文件，防止下次被误判为已完成
        if os.path.exists(filepath):
            os.remove(filepath)
        return f"ERROR: {filename} -> {e}"

def main():
    # 确保保存目录存在
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"已创建目录: {SAVE_DIR}")

    # 检查 .netrc 权限文件是否存在
    if not os.path.exists(os.path.expanduser('~/.netrc')):
        print("警告: 未发现 ~/.netrc 文件。下载 NASA 数据需要此授权文件，请先配置。")
        return

    # 读取链接
    if not os.path.exists(LINKS_FILE):
        print(f"错误: 找不到链接文件 {LINKS_FILE}")
        return

    with open(LINKS_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and line.startswith('http')]

    total_files = len(urls)
    print(f"开始并发下载...")
    print(f"总文件数: {total_files} | 线程数: {MAX_WORKERS}")
    print(f"数据将保存至: {SAVE_DIR}")

    # 使用 Session 保持长连接，显著提升下载 15 万个小文件的效率
    with requests.Session() as session:
        # 设置线程池
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_url = {executor.submit(download_one_file, url, session): url for url in urls}
            
            # 使用 tqdm 渲染进度条
            with tqdm(total=total_files, desc="下载进度", unit="file") as pbar:
                for future in as_completed(future_to_url):
                    result = future.result()
                    # 如果需要监控失败的文件，可以取消下面这行的注释
                    # if "ERROR" in result: print(result)
                    pbar.update(1)

if __name__ == "__main__":
    main()