# Swin-UNet MoE: Extreme Precipitation Forecasting

基于 Swin-UNet 和混合专家系统（Mixture of Experts, MoE）的降水预测模型。该项目专为气象领域的**极端降水预测**设计，通过引入像素级的路由机制和针对不同降水形态（无雨、层状云降水、对流降水）的专属专家网络，旨在解决极端强降水低估和误报（FAR）的问题。

## 🌟 核心特性 (Key Features)

* **Swin-UNet Backbone**: 采用分层 Swin Transformer 构建的 U-Net 架构，提取强力的全局与局部空间特征。
* **Global Alignment**: 在 Decoder 中引入 `GlobalAlignBlock`（基于交叉注意力机制），增强时空特征对齐。
* **多尺度 FPN 路由器 (FPN Router)**: 整合多层特征图生成像素级的路由概率，智能分类“背景”、“层状云”和“强对流”区域。
* **高分辨率气象专家网络 (Regime-Specific Experts)**:
  * ☁️ **Background Expert**: 轻量级门控卷积，处理无雨背景。
  * 🌧️ **Stratiform Expert**: 深度可分离卷积，处理平滑的中小雨量（层状云）。
  * ⛈️ **Convective Expert**: 借鉴 ASPP 思想，采用多尺度空洞卷积（Dilation=2, 4）与全局池化，极大扩张感受野，专精捕捉极端强对流降水（30mm/50mm+）。

## 💡 训练策略 (Training Strategy)

本模型设计了专门的损失函数系统以平衡极端值与误报率：
* **专家各司其职 (Expert Loss)**: 背景与层状云使用 Huber Loss；对流区采用**不对称 MSE Loss**，对低估（Underestimation）施加 10 倍重罚。
* **听从指挥 (Router Loss)**: 加权交叉熵，降水强度越大的像素分配越高的权重，确保极值区被正确路由给对流专家。
* **交叉抑制 (Silence Loss)**: 强迫对流与层状云专家在背景区域输出 0，大幅压制误报率（False Alarm Rate）。
* **差分学习率**: 冻结/慢速更新基础特征提取器（Encoder/Router），高学习率激活新的对流专家网络。
