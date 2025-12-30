import torch
from torch.optim import Optimizer

class SCOOP(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.5, 0.999), rho=0.96, epsilon=1e-5, weight_decay=1e-4):
        """
        Ultimate SCOOP: Low Inertia + Safety Clamp + Robust Hessian
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

                # 初始化
                if len(state) == 0 or 'exp_avg' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.ones_like(p) # 安全起见初始设为 1

                exp_avg = state['exp_avg']
                hessian = state['hessian']
                beta1, _ = group['betas']
                state['step'] += 1

                # 1. Weight Decay (防止躺平)
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # 2. Momentum (低惯性 0.5)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 3. Apply Curvature
                curvature = hessian.abs()
                
                # 🔥 安全钳位：防止旧服务器精度不足导致除零爆炸
                curvature = torch.clamp(curvature, min=1e-5)
                
                denom = curvature.add_(group['epsilon'])
                
                # Update
                p.addcdiv_(exp_avg, denom, value=-group['lr'])

        return loss

    def hutchinson_hessian(self, num_samples=1):
        """
        Hutchinson's method with robust filtering for detached gradients.
        """
        params = []
        groups = []
        for group in self.param_groups:
            for p in group['params']:
                # 🔥 严格过滤：必须有 grad 且 grad 必须有计算图
                if p.requires_grad and p.grad is not None and p.grad.requires_grad:
                    params.append(p)
                    groups.append(group)

        # 如果没有可求二阶导的参数，直接返回，防止 crash
        if not params:
            return

        grads = [p.grad for p in params]

        for i in range(num_samples):
            # Rademacher distribution
            vs = [torch.randint_like(p, high=2) * 2 - 1 for p in params]
            
            # Matrix-Vector Product
            grad_dot_v = sum([torch.sum(g * v) for g, v in zip(grads, vs)])
            
            # 🔥 允许未使用的梯度 (allow_unused=True)
            hvs = torch.autograd.grad(
                grad_dot_v, params, 
                retain_graph=(i < num_samples - 1), 
                only_inputs=True,
                allow_unused=True 
            )
            
            for p, v, hv, group in zip(params, vs, hvs, groups):
                if hv is None:
                    # 如果二阶导不存在，视为 0
                    current_curvature = torch.zeros_like(p)
                else:
                    current_curvature = v * hv
                
                state = self.state[p]
                if 'hessian' not in state:
                    state['hessian'] = torch.ones_like(p)
                
                # 平滑更新 Hessian
                state['hessian'].mul_(group['rho']).add_(current_curvature, alpha=1 - group['rho'])