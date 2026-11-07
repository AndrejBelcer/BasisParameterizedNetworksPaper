"""Blocked polynomial MMD and its exact analytic feature gradient.

Real features are constants. FP64 reductions reduce cancellation; the matrix
products follow the input dtype and the caller's existing precision settings.
"""
import torch


def _validate(x, y, block_size):
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError('Features must have matching [N,D] shapes.')
    if len(x) < 2 or len(y) < 2 or block_size < 1:
        raise ValueError('KID needs at least two images in each set and a positive block size.')
    if x.device != y.device or x.dtype != y.dtype:
        raise ValueError('Feature device and dtype must match.')


@torch.no_grad()
def self_constant(y, block_size=512):
    _validate(y, y, block_size)
    n, d = y.shape
    total = torch.zeros((), dtype=torch.float64, device=y.device)
    for i in range(0, n, block_size):
        a = y[i:i+block_size]
        for j in range(i, n, block_size):
            b = y[j:j+block_size]
            q = 1 + (a @ b.T) / d
            if i == j:
                q.diagonal().zero_()
            total += q.pow(3).sum(dtype=torch.float64) * (1 if i == j else 2)
    return float(total / (n*(n-1)))


@torch.no_grad()
def kid_and_gradient(x, y, real_constant, block_size=512, loss_scale=1.0, gradient=True):
    _validate(x, y, block_size)
    m, d = x.shape
    n = len(y)
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    grad = torch.zeros_like(x) if gradient else None
    for i in range(0, m, block_size):
        a = x[i:i+block_size]
        g = torch.zeros_like(a) if gradient else None
        for j in range(0, m, block_size):
            b = x[j:j+block_size]
            q = 1 + (a @ b.T)/d
            if i == j:
                q.diagonal().zero_()
            total += q.pow(3).sum(dtype=torch.float64)/(m*(m-1))
            if gradient:
                g.add_(q.square() @ b, alpha=6.0/(d*m*(m-1)))
        for j in range(0, n, block_size):
            b = y[j:j+block_size]
            q = 1 + (a @ b.T)/d
            total -= 2*q.pow(3).sum(dtype=torch.float64)/(m*n)
            if gradient:
                g.add_(q.square() @ b, alpha=-6.0/(d*m*n))
        if gradient:
            grad[i:i+len(a)] = g * loss_scale
    value = float(total + real_constant)
    if not torch.isfinite(total) or (gradient and not bool(torch.isfinite(grad).all())):
        raise FloatingPointError('Nonfinite KID or feature gradients.')
    return value, grad
