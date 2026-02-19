#!/bin/bash
source /etc/network_turbo

for file in links_chunk_*
do
    echo "------------------------------------------------------"
    echo "正在处理分片: $file"
    # 使用修改后的 aria2.conf
    aria2c --conf-path=aria2.conf -i "$file" -d /root/autodl-tmp/raw_data
    
    echo "分片完成，冷却 30 秒，防止被 NASA 封锁..."
    sleep 30
done
