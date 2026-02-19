import torch
import torch.nn as nn
import torch.nn.functional as F

class RectifiedFlowMoE(nn.Module):
    """
    工业级流匹配包装器 (Rectified Flow + MoE)
    支持 Heun 高阶采样和物理感知加权损失
    """
    def __init__(self, backbone, sigma_min=1e-5):
        super().__init__()
        self.backbone = backbone  # 确保 backbone.in_chans == 5 (3帧历史 + 1帧xt + 1帧t)
        self.sigma_min = sigma_min

    def forward(self, x_1, condition, targets_raw=None):
        """
        x_1: 归一化后的真值 [B, 1, H, W]
        condition: 归一化后的过去帧 [B, 3, H, W]
        targets_raw: 未归一化的物理真值 (用于计算物理加权，可选)
        """
        B, _, H, W = x_1.shape
        device = x_1.device

        # 1. 采样时间步 t (使用 logit-normal 分布采样效果通常更好，这里简化为均匀采样)
        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1, 1)

        # 2. 构造直线轨迹
        # x_0 为标准高斯噪声
        x_0 = torch.randn_like(x_1)
        # x_t = t*x1 + (1-t)*x0
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        target_v = x_1 - x_0

        # 3. 构造 5 通道输入 [Cond, x_t, t_map]
        t_map = t_expand.expand(B, 1, H, W)
        model_input = torch.cat([condition, x_t, t_map], dim=1)

        # 4. 模型预测
        # pred_v: 预测的速度向量
        pred_v, router_logits, expert_preds = self.backbone(model_input)

        # 5. 物理感知加权损失 (Weighting Strategy)
        # 策略：t 越大时（越接近真实细节），权重越高；强降水区域权重更高
        weight = 1.0 + t_expand # 时间加权
        
        if targets_raw is not None:
            # 强降水区域 (例如 > 10mm/h) 给予 5 倍关注
            rain_mask = (targets_raw > 10.0).float()
            weight = weight * (1.0 + 4.0 * rain_mask)

        # 计算加权 MSE
        loss_flow = (F.mse_loss(pred_v, target_v, reduction='none') * weight).mean()

        return loss_flow, router_logits

    @torch.no_grad()
    def sample(self, condition, steps=20, solver='heun'):
        """
        采样函数
        steps: 采样步数，RF 通常 10-20 步即可
        solver: 'euler' 或 'heun'
        """
        self.backbone.eval()
        B, _, H, W = condition.shape
        device = condition.device
        dt = 1.0 / steps

        # 从纯噪声 x_0 开始
        x = torch.randn(B, 1, H, W, device=device)

        for i in range(steps):
            t_val = i / steps
            t_next = (i + 1) / steps
            
            # --- 步骤 1: 预估 (Euler Step) ---
            v_pred = self._get_v(x, condition, t_val)
            
            if solver == 'euler':
                x = x + v_pred * dt
            
            elif solver == 'heun':
                # --- 步骤 2: 校正 (Heun's 2nd order) ---
                x_next_euler = x + v_pred * dt
                v_pred_next = self._get_v(x_next_euler, condition, t_next)
                
                # 使用平均速度进行更精确的步进
                x = x + 0.5 * (v_pred + v_pred_next) * dt

        return torch.clamp(x, min=-1.0, max=10.0) # 根据你的归一化范围截断

    def _get_v(self, x, condition, t_val):
        """辅助函数：获取模型预测的速度"""
        B, _, H, W = x.shape
        t_tensor = torch.full((B, 1, H, W), t_val, device=x.device)
        model_input = torch.cat([condition, x, t_tensor], dim=1)
        # 推理时使用硬路由获得锐利边缘
        v, _, _ = self.backbone(model_input, hard_routing=True)
        return v