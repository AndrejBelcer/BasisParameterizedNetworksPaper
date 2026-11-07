"""Batched validation using the selected library's official metric functions."""
import numpy as np
import torch
from .sampling import EulerSampler, rgb01, noise_bank


def clean_kid_statistics(features, reference, subsets, subset_size, seed=2020):
    """Official Clean-FID subset estimates and their population subset SD."""
    from cleanfid.fid import kernel_distance
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        scores = [kernel_distance(reference, features, num_subsets=1,
                                  max_subset_size=subset_size) for _ in range(subsets)]
    finally:
        np.random.set_state(state)
    return float(np.mean(scores)), float(np.std(scores, ddof=0))


def clean_frechet_distance(mu, sigma, reference_mu, reference_sigma):
    """Official Clean-FID, with its same formula for SciPy's removed disp API."""
    from cleanfid.fid import frechet_distance
    try:
        return float(frechet_distance(mu, sigma, reference_mu, reference_sigma))
    except TypeError as exc:
        if 'disp' not in str(exc):
            raise
    # Clean-FID 0.1.35 expects sqrtm(..., disp=False) to return (root, error).
    # SciPy >=1.18 removed disp and returns the root directly. Preserve the
    # official product, diagonal fallback and real-part treatment unchanged.
    from scipy.linalg import sqrtm
    root = sqrtm(sigma.dot(reference_sigma))
    if not np.isfinite(root).all():
        offset = np.eye(sigma.shape[0])*1e-6
        root = sqrtm((sigma+offset).dot(reference_sigma+offset))
    if np.iscomplexobj(root):
        root = root.real
    delta = mu-reference_mu
    return float(delta.dot(delta)+np.trace(sigma)+np.trace(reference_sigma)-2*np.trace(root))


@torch.no_grad()
def evaluate(velocity, extractor, real_features, real_statistics, grid, device,
             count=2048, batch_size=8, seed=41001, kid_subsets=100,
             kid_subset_size=1000, compute_fid=False, use_clean_fid=False):
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_isc import isc_features_to_metric
    sampler = EulerSampler(velocity, grid, device, checkpoint_steps=False)
    noises = noise_bank(count, 3, 32, seed)
    features, logits, pictures = [], [], []
    for i in range(0, count, batch_size):
        images = rgb01(sampler(noises[i:i+batch_size].to(device)))
        if use_clean_fid:
            f, l = extractor.evaluation(images, compute_logits=compute_fid)
        else:
            f, l = extractor.evaluation(images)
        features.append(f.cpu())
        if l is not None:
            logits.append(l.cpu())
        if i < 100:
            pictures.append(images[:min(len(images),100-i)].cpu())
    features = torch.cat(features)
    logits = torch.cat(logits) if logits else None
    if not torch.isfinite(features).all() or (logits is not None and not torch.isfinite(logits).all()):
        raise FloatingPointError('Nonfinite validation features.')
    subset_size = min(kid_subset_size, count, len(real_features))
    if use_clean_fid:
        value, deviation = clean_kid_statistics(features.numpy(), real_features.numpy(),
                                                kid_subsets, subset_size)
    else:
        from torch_fidelity.metric_kid import kid_features_to_metric
        metrics = kid_features_to_metric(features, real_features,
            kid_subsets=kid_subsets, kid_subset_size=subset_size,
            kid_kernel='poly', kid_kernel_poly_degree=3, kid_kernel_poly_gamma=None,
            kid_kernel_poly_coef0=1, rng_seed=2020, verbose=False)
        value = float(metrics['kernel_inception_distance_mean'])
        deviation = float(metrics['kernel_inception_distance_std'])
    result = {'kid': value, 'kid_subset_std': deviation, 'fid': None, 'is': None}
    if compute_fid:
        stats = fid_features_to_statistics(features)
        if use_clean_fid:
            result['fid'] = clean_frechet_distance(stats['mu'], stats['sigma'],
                                                  real_statistics['mu'], real_statistics['sigma'])
        else:
            result['fid'] = float(fid_statistics_to_metric(stats, real_statistics, verbose=False)['frechet_inception_distance'])
        if logits is None:
            raise RuntimeError('IS evaluation requires TensorFlow-compatible logits.')
        result['is'] = float(isc_features_to_metric(logits, splits=10, shuffle=True, rng_seed=2020)['inception_score_mean'])
    return result, torch.cat(pictures)
