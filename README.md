# Swin-UNetMoE 降水预报模型

本项目实现了一个基于 **Swin Transformer** 与 **混合专家系统 (MoE)** 的降水预报模型。通过解耦降水机制（背景、层状云、对流云），旨在提升极端降水的捕捉能力并抑制背景噪声。

## 1. 模型结构 (Model Structure)

模型采用 Encoder-Decoder 架构，核心流程如下：

*   **共享 Swin 编码器 (Shared Swin Encoder)**: 
    *   利用 Swin Transformer 块提取多尺度分层特征。
*   **双路径机制路由器 (Dual-Path Regime Router)**: 
    *   **输入**: 同时接收深层语义特征（4x4）与浅层空间特征（32x32）。
    *   **功能**: 预测每个像素属于三种降水机制（Regime）的概率，生成软门控权重。
*   **混合专家解码器 (MoE Decoder)**:
    *   **Expert 0 (背景)**: 静态专家，输出固定零值，专门吸收无雨区噪声。
    *   **Expert 1 (层状云)**: 动态专家，针对中低强度降水，优化整体 RMSE。
    *   **Expert 2 (对流云)**: 动态专家，通过 **+1.0 偏置初始化** 引导，专门负责捕捉高强度极端降水。
*   **背景抑制 (Background Suppressor)**: 
    *   通过独立的抑制器产生掩码（Mask），进一步清除 Router 在背景区域的误触发。

---

## 2. 文件结构 (File Structure)

```text
.
├── download&&preprocess/   # 数据下载和预处理
│   ├── preprocess.py       # 将原始 .nc 转换为 Zarr 并生成物理标签
│   └── convert_to_pt.py    # 将 Zarr 转换为高效加载的 .pt 张量文件
├── models/                 # 模型核心定义
│   ├── swin_unet_moe.py    # SwinUNetMoE 主网络架构
│   └── regime_module.py    # 路由器(Router)、注意力门控及标签生成逻辑
├── train_script/           # 训练与评估模块
│   ├── train_moe.py        # 主训练脚本（含专家冻结与偏置初始化逻辑）
│   └── dataset.py          # PyTorch数据加载器
└── README.md               # 项目说明文档
```

---

## 3. 训练策略 (Core Strategies)

1.  **专家冻结**: 训练初期冻结解码器参数，强制 Router 先学会如何分配降水机制。
2.  **偏置注入**: 手动初始化专家偏置，从物理上定义专家的“职责”，解决冷启动问题。
3.  **温度锐化**: Router 输出使用 T=0.5 的 Softmax，使降水落区边界更加分明。
4.  **只放了主训练脚本**：剩下的微调的脚本太杂乱了就没放上来

---