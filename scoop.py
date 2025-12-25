import torch
from torch.optim import Optimizer

class SCOOP(Optimizer):
    def __init__(self, params, lr=1e-3, rho=0.9, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, rho=rho, weight_decay=weight_decay, betas=betas, eps=eps)
        super(SCOOP, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['curvature'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                curvature = state['curvature']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                curv_adaptive = curvature.abs().add_(denom)
                step_size = group['lr']
                p.addcdiv_(exp_avg, curv_adaptive, value=-step_size)
        return loss

    @torch.no_grad()
    def hutchinson_hessian(self):
        params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.requires_grad:
                    params.append(p)
        if len(params) == 0: return

        vs = [torch.randint_like(p, high=2) * 2 - 1 for p in params]
        try:
            grads = [p.grad for p in params]
            v_dot_grad = sum([torch.sum(v * g) for v, g in zip(vs, grads)])
            hv = torch.autograd.grad(v_dot_grad, params, retain_graph=False)
            
            for i, p in enumerate(params):
                state = self.state[p]
                if 'curvature' not in state: state['curvature'] = torch.zeros_like(p)
                estimate = vs[i] * hv[i]
                rho = self.param_groups[0]['rho']
                state['curvature'].mul_(rho).add_(estimate, alpha=1 - rho)
        except RuntimeError:
            pass