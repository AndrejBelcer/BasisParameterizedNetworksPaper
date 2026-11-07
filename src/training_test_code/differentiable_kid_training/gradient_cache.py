"""Whole-bank gradients with small differentiable replay batches."""
import torch
from .sampling import rgb01


@torch.no_grad()
def collect_features(sampler, extractor, noises, device, batch_size):
    parts = []
    clipped = 0.0
    for i in range(0, len(noises), batch_size):
        raw = sampler(noises[i:i+batch_size].to(device))
        clipped += float((raw.abs() > 1).float().mean()) * len(raw)
        parts.append(extractor(rgb01(raw)).detach())
    features = torch.cat(parts)
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError('Nonfinite generated features.')
    return features, clipped/len(noises)


def replay_backward(sampler, extractor, noises, cached, feature_gradient,
                    device, batch_size, atol=2e-4, rtol=2e-4):
    largest = 0.0
    for i in range(0, len(noises), batch_size):
        raw = sampler(noises[i:i+batch_size].to(device))
        features = extractor(rgb01(raw))
        reference = cached[i:i+len(features)]
        error = float((features.detach()-reference).abs().max())
        largest = max(largest, error)
        if not torch.allclose(features.detach(), reference, atol=atol, rtol=rtol):
            raise RuntimeError(f'Feature replay mismatch (max error {error:g}); update was not applied.')
        # Already normalized for the full bank. Never divide by chunk count.
        features.backward(feature_gradient[i:i+len(features)])
        del features, raw
    return largest


def norm(grads):
    values = [g.detach().double().square().sum() for g in grads if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def snapshot_gradients(parameters):
    return [None if p.grad is None else p.grad.detach().clone() for p in parameters]


def term_gradient_norms(parameters, degree):
    if not isinstance(degree,int) or degree < 1:
        return []
    totals = None
    for p in parameters:
        if p.grad is None or p.ndim < 1 or p.shape[0] != degree:
            continue
        values = p.grad.detach().double().reshape(degree,-1).square().sum(dim=1)
        totals = values if totals is None else totals+values
    return [] if totals is None else totals.sqrt().cpu().tolist()


def combine_gradients(parameters, metric_grads, rgb_weight):
    rgb_grads = [p.grad for p in parameters]
    metric_norm, rgb_norm = norm(metric_grads), norm(rgb_grads)
    dot = sum(float((a.double()*b.double()).sum()) for a, b in zip(metric_grads, rgb_grads)
              if a is not None and b is not None)
    cosine = dot/(metric_norm*rgb_norm) if metric_norm and rgb_norm else 0.0
    for p, metric, rgb in zip(parameters, metric_grads, rgb_grads):
        if metric is None:
            p.grad = None if rgb is None else rgb.mul_(rgb_weight)
        else:
            p.grad = metric if rgb is None else metric.add_(rgb, alpha=rgb_weight)
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError('Nonfinite combined gradient; optimizer was not called.')
    return metric_norm, rgb_norm, cosine


def verify_cached_gradient(model, extractor, real_features, device, replay_velocity=None):
    """Small real-model integration check, without changing any weights."""
    from .sampling import EulerSampler, shifted_grid, noise_bank
    from .kid_loss import kid_and_gradient
    parameters = list(model.parameters())
    noises = noise_bank(2, 3, 32, 711)
    grid = shifted_grid(4, .001, .37)
    sampler = EulerSampler(model, grid, device, checkpoint_steps=False)
    y = real_features[:min(16,len(real_features))]
    model.zero_grad(set_to_none=True)
    x = extractor(rgb01(sampler(noises.to(device))))
    d = x.shape[1]
    q = (1+x@x.T/d).pow(3)
    dense = (q.sum()-q.diagonal().sum())/(len(x)*(len(x)-1)) - 2*(1+x@y.T/d).pow(3).mean()
    (dense*1000).backward()
    expected = snapshot_gradients(parameters)
    del x, q, dense
    model.zero_grad(set_to_none=True)
    replay_sampler = EulerSampler(replay_velocity or model, grid, device, checkpoint_steps=False)
    cached, _ = collect_features(replay_sampler, extractor, noises, device, 2)
    _, gradient = kid_and_gradient(cached, y, 0.0, block_size=8, loss_scale=1000)
    checkpointed = EulerSampler(replay_velocity or model, grid, device, checkpoint_steps=True)
    replay_backward(checkpointed, extractor, noises, cached, gradient, device, 2)
    diffs = [a-p.grad if p.grad is not None else a for a,p in zip(expected,parameters) if a is not None]
    relative = norm(diffs)/max(norm(expected),1e-20)
    if not relative < 5e-3 or norm(expected) == 0:
        model.zero_grad(set_to_none=True)
        raise RuntimeError(f'Cached/full graph gradient check failed: relative error {relative:g}.')
    model.zero_grad(set_to_none=True)
    print(f'Full-graph/cached gradient check passed (relative error {relative:.3g}).', flush=True)
    return relative
