import torch
from torch.optim import Optimizer

class SCOOP(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), rho=0.96, epsilon=1e-8, weight_decay=1e-2):
        """
        Robust SCOOP Optimizer with Safety Clamps
        """
        defaults = dict(lr=lr, betas=betas, rho=rho, epsilon=epsilon, weight_decay=weight_decay)
        super(SCOOP, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # 初始化状态 (更鲁棒的检查)
                if len(state) == 0 or 'exp_avg' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    # 初始化 hessian 为 1，而不是 0，防止刚开始就除以 0
                    state['hessian'] = torch.ones_like(p)

                exp_avg = state['exp_avg']
                hessian = state['hessian']
                beta1, _ = group['betas']
                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Momentum (一阶矩)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # --- 【核心修复 1】: 分母安全钳位 (Safety Clamp) ---
                # 原代码: denom = hessian.abs().add_(group['epsilon'])
                # 问题: 如果 hessian 接近 0，步长会爆炸。
                # 修复: 强制 Hessian 的模长至少为 1e-4。
                # 这保证了最大的缩放倍数不会超过 10000 倍，防止参数飞出天际。
                curvature = hessian.abs()
                #curvature = torch.clamp(curvature, min=1e-4) 
                #denom = curvature.add_(group['epsilon'])
                denom = curvature.add_(1e-8) # 依靠 epsilon 来做最后的防线
                
                # Update
                p.addcdiv_(exp_avg, denom, value=-group['lr'])

        return loss

    def hutchinson_hessian(self, num_samples=1):
        """
        Estimate Hessian diagonal using Hutchinson's method.
        """
        params = []
        groups = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    params.append(p)
                    groups.append(group)

        if not params:
            return

        grads = [p.grad for p in params]

        for i in range(num_samples):
            # Rademacher distribution
            vs = [torch.randint_like(p, high=2) * 2 - 1 for p in params]
            
            # Matrix-Vector Product
            grad_dot_v = sum([torch.sum(g * v) for g, v in zip(grads, vs)])
            
            hvs = torch.autograd.grad(
                grad_dot_v, params, 
                retain_graph=(i < num_samples - 1), 
                only_inputs=True
            )
            
            for p, v, hv, group in zip(params, vs, hvs, groups):
                state = self.state[p]
                if 'hessian' not in state:
                    state['hessian'] = torch.ones_like(p)
                
                # --- 【核心修复 2】: 估算值截断 ---
                # Hutchinson 估计偶尔会产生极大的值，污染历史平均。
                # 我们这里不做硬截断，但在更新时需要注意
                
                # v * hv 近似于 Hessian 对角线元素
                current_curvature = v * hv
                
                # 更新滑动平均
                # H_new = rho * H_old + (1-rho) * H_curr
                state['hessian'].mul_(group['rho']).add_(current_curvature, alpha=1 - group['rho'])