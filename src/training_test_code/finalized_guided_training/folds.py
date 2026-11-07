"""Exact pooled KID and paired delete-fold fits from cached Inception features.

Folds are diagnostics/resampling replicates, never independent extra fit blocks.
All parameter locations and both feature pipelines use the same latent slices.
"""
from dataclasses import dataclass
import math
import time

import torch

from .kid_evaluator import KIDResult, paired_difference, component_paired_difference
from .parameter_space import require_finite
from .quadratic import FitBlock, fit_quadratic, design, coefficient_count


@dataclass
class PartitionScores:
    full: KIDResult
    folds: list
    left_out: list


@torch.no_grad()
def partition_scores(metric, entries, fold_count):
    """Pool banks at ONE model location, retaining every cross-fold kernel pair.

    Store only O(N*fold_count) kernel row sums. Real-reference cross rows are
    reused from the original scores. No image generation or Inception calls.
    Each global fold contains one equal slice from every participating bank.
    """
    if type(fold_count) is not int or fold_count < 3 or not entries:
        raise ValueError("Partitioning needs at least three folds and cached histories")
    entries = sorted(entries, key=lambda entry: entry[0].noise_seed)
    identities = [(history.noise_seed, count) for history, count in entries]
    if len(set(seed for seed, _ in identities)) != len(entries):
        raise ValueError("A noise bank was counted twice in pooled features")
    if any(count % fold_count or count//fold_count < 3 or history.count < count
           for history, count in entries):
        raise ValueError("Every bank must supply equally sized folds of at least three images")
    cache_key = (fold_count, tuple(identities))
    cache = entries[0][0].partitions
    if cache_key in cache:
        return cache[cache_key]
    original = [metric.score(history, count) for history, count in entries]
    metric.runtime.synchronize()
    started = time.monotonic()
    names = list(original[0].component_scores)
    if not names or any(set(row.component_scores) != set(names) for row in original):
        raise ValueError("Pooled histories must have matching component KIDs")
    intervals = [tuple((history.noise_seed, fold*(count//fold_count), (fold+1)*(count//fold_count))
                       for history, count in entries) for fold in range(fold_count)]
    fold_size = sum(count//fold_count for _, count in entries)
    component_rows, component_cross = {}, {}
    for name in names:
        features, cross = [], []
        for fold in range(fold_count):
            features.append(torch.cat([history.features[name][fold*(count//fold_count):
                                                               (fold+1)*(count//fold_count)].cpu()
                                       for history, count in entries]))
            cross.append(torch.cat([row.component_cross_rows[name][fold*(count//fold_count):
                                                                   (fold+1)*(count//fold_count)].double().cpu()
                                    for row, (_, count) in zip(original, entries)]))
        rows = [torch.zeros(fold_size, fold_count, dtype=torch.float64) for _ in range(fold_count)]
        for first in range(fold_count):
            rows[first][:, first] = metric._kernel_rows(features[first], features[first], same=True).cpu()
            for second in range(first+1, fold_count):
                left, right = metric.kernel_pair_rows(features[first], features[second])
                rows[first][:, second], rows[second][:, first] = left, right
        component_rows[name], component_cross[name] = rows, cross
        del features

    def subset(selected):
        count = len(selected)*fold_size
        scores, self_rows, cross_rows = {}, {}, {}
        for name in names:
            self_rows[name] = torch.cat([component_rows[name][fold][:, selected].sum(dim=1)
                                         for fold in selected])
            cross_rows[name] = torch.cat([component_cross[name][fold] for fold in selected])
            scores[name] = float(self_rows[name].sum()/(count*(count-1))
                                 -2*cross_rows[name].mean()+metric.reference_constants[name])
            if not math.isfinite(scores[name]):
                raise FloatingPointError("Nonfinite pooled/fold KID")
        ranges = tuple(interval for fold in selected for interval in intervals[fold])
        return KIDResult(sum(scores.values())/len(names), sum(self_rows.values())/len(names),
                         sum(cross_rows.values())/len(names), entries[0][0].noise_seed, 0, count,
                         scores, self_rows, cross_rows, ranges)

    all_folds = list(range(fold_count))
    result = PartitionScores(subset(all_folds), [subset([fold]) for fold in all_folds],
                             [subset([other for other in all_folds if other != fold]) for fold in all_folds])
    cache[cache_key] = result
    metric.runtime.synchronize()
    metric.kernel_seconds += time.monotonic()-started
    metric.score_evaluations += 1
    return result


def delete_fold_covariance(values):
    """Jackknife scaling for overlapping leave-one-fold-out replicates."""
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("Delete-fold covariance needs at least three coefficient vectors")
    require_finite(values, "delete-fold coefficient vectors")
    centered = values-values.mean(dim=0)
    covariance = (len(values)-1)/len(values)*(centered.T @ centered)
    require_finite(covariance, "delete-fold covariance")
    return covariance


class FoldLandscape:
    """Independent pooled blocks plus correlated delete-fold fits of those data."""
    def __init__(self, metric, pools, scales, fold_count, ridge, covariance_floor, diagonal=False,
                 cross_term_ridge_multiplier=10., fold_regression_penalty=1., fold_regression_z=1.):
        started = time.monotonic()
        self.metric, self.pools, self.fold_count = metric, pools, fold_count
        self.blocks, self.source_blocks = [], []
        self.fold_blocks = [[] for _ in range(fold_count)]
        self.training_blocks = [[] for _ in range(fold_count)]
        # Only banks with the SAME measured geometry may be merged in a fit.
        # A targeted bank or failed confirmation remains its own independent block.
        groups = {}
        for pool in pools:
            candidates = {tuple(candidate.coordinates.tolist()): candidate
                          for candidate in pool.candidates[1:] if candidate.history.count >= pool.count}
            if candidates:
                groups.setdefault(tuple(sorted(candidates)), []).append((pool, candidates))
        for keys, group in groups.items():
            parent = partition_scores(metric, [(pool.incumbent.history, pool.count) for pool, _ in group], fold_count)
            results = [partition_scores(metric, [(candidates[key].history, pool.count)
                                                 for pool, candidates in group], fold_count) for key in keys]
            coordinates = [torch.tensor(key, dtype=torch.float64) for key in keys]
            self.blocks.append(FitBlock(coordinates, parent.full, [row.full for row in results]))
            self.source_blocks.append(dict(
                source_ids=[f"noise_{pool.bank.seed}" for pool, _ in group],
                noise_seeds=[pool.bank.seed for pool, _ in group],
                roles=[getattr(pool, "role", "search") for pool, _ in group],
                image_counts=[pool.count for pool, _ in group], pooled_image_count=parent.full.count,
                coordinates=[point.tolist() for point in coordinates],
                fold_membership=[row.sample_ranges for row in parent.folds]))
            for fold in range(fold_count):
                self.fold_blocks[fold].append(FitBlock(coordinates, parent.folds[fold],
                                                       [row.folds[fold] for row in results]))
                self.training_blocks[fold].append(FitBlock(coordinates, parent.left_out[fold],
                                                           [row.left_out[fold] for row in results]))
        self.bundle = fit_quadratic(self.blocks, scales, ridge=ridge,
                                    covariance_floor=covariance_floor, diagonal=diagonal,
                                    cross_term_ridge_multiplier=cross_term_ridge_multiplier)
        self.training_fits = [fit_quadratic(blocks, scales, ridge=ridge,
                                            covariance_floor=covariance_floor, diagonal=diagonal,
                                            cross_term_ridge_multiplier=cross_term_ridge_multiplier)
                              for blocks in self.training_blocks]
        self.fold_fits = [fit_quadratic(blocks, scales, ridge=ridge,
                                        covariance_floor=covariance_floor, diagonal=diagonal,
                                        cross_term_ridge_multiplier=cross_term_ridge_multiplier)
                          for blocks in self.fold_blocks]
        self.diagonal_bundle = fit_quadratic(self.blocks, scales, ridge=ridge,
                                             covariance_floor=covariance_floor, diagonal=True,
                                             cross_term_ridge_multiplier=cross_term_ridge_multiplier)
        self.ridge, self.covariance_floor = ridge, covariance_floor
        self.cross_term_ridge_multiplier = cross_term_ridge_multiplier
        self.bundle.combined.fold_models = [fit.combined for fit in self.fold_fits]
        self.bundle.combined.fold_regression_penalty = fold_regression_penalty
        self.bundle.combined.fold_regression_z = fold_regression_z
        # Every portfolio optimizer uses the same disjoint-fold risk preference.
        for fit in self.training_fits:
            fit.combined.fold_models = self.bundle.combined.fold_models
            fit.combined.fold_regression_penalty = fold_regression_penalty
            fit.combined.fold_regression_z = fold_regression_z
        joint = torch.stack([torch.cat([fit.components[name].coefficients for name in self.bundle.component_names])
                             for fit in self.training_fits])
        fold_covariance = delete_fold_covariance(joint)
        width = len(self.bundle.combined.coefficients)
        count = len(self.bundle.component_names)
        average = torch.cat([torch.eye(width, dtype=torch.float64)/count for _ in range(count)], dim=1)
        self.bundle.combined.resampling_covariance = average @ fold_covariance @ average.T
        for index, name in enumerate(self.bundle.component_names):
            selection = slice(index*width, (index+1)*width)
            self.bundle.components[name].resampling_covariance = fold_covariance[selection, selection]
        # The maximum of paired and delete-fold variances is used pointwise.
        # Adding them would count the same sampling noise twice.
        self.bundle.observation_details.update(fold_count=fold_count, pooled_features=True,
                                               source_blocks=self.source_blocks,
                                               uncertainty_rule="max_paired_and_delete_fold",
                                               fit_analysis_seconds=time.monotonic()-started)

    def evidence(self, coordinates):
        parents, children, banks = [], [], []
        for pool in self.pools:
            candidate = next((item for item in pool.candidates[1:]
                              if torch.equal(item.coordinates, coordinates) and item.history.count >= pool.count), None)
            if candidate is None:
                continue
            parents.append((pool.incumbent.history, pool.count))
            children.append((candidate.history, pool.count))
            delta, se = paired_difference(self.metric.score(candidate.history, pool.count),
                                          self.metric.score(pool.incumbent.history, pool.count))
            banks.append(dict(delta=delta, se=se, count=pool.count, noise_seed=pool.bank.seed,
                              source_id=f"noise_{pool.bank.seed}", role=getattr(pool, "role", "search")))
        if not children:
            return None
        parent = partition_scores(self.metric, parents, self.fold_count)
        child = partition_scores(self.metric, children, self.fold_count)
        delta, paired_se = paired_difference(child.full, parent.full)
        held_out = torch.tensor([paired_difference(a, b)[0] for a, b in zip(child.left_out, parent.left_out)],
                                dtype=torch.float64).reshape(-1, 1)
        fold_se = float(delete_fold_covariance(held_out)[0, 0].clamp_min(0).sqrt())
        folds = [dict(fold_id=index, delta=paired_difference(a, b)[0], gain=-paired_difference(a, b)[0],
                      se=paired_difference(a, b)[1], count=a.count, sample_ranges=a.sample_ranges)
                 for index, (a, b) in enumerate(zip(child.folds, parent.folds))]
        components = {}
        for name in child.full.component_scores:
            change, error = component_paired_difference(child.full, parent.full, name)
            values = torch.tensor([component_paired_difference(a, b, name)[0]
                                   for a, b in zip(child.left_out, parent.left_out)], dtype=torch.float64).reshape(-1, 1)
            component_fold_se = float(delete_fold_covariance(values)[0, 0].clamp_min(0).sqrt())
            components[name] = dict(delta=change, se=max(error, component_fold_se),
                                     paired_se=error, fold_se=component_fold_se)
        return dict(delta=delta, se=max(paired_se, fold_se), paired_se=paired_se, fold_se=fold_se,
                    banks=banks, components=components, folds=folds,
                    support_count=sum(row["delta"] < 0 for row in banks), required_support=len(banks)//2+1,
                    supporting_folds=sum(row["delta"] < 0 for row in folds), pooled=True)

    def prediction(self, coordinates):
        fit = self.bundle.combined
        row = design(coordinates, fit.scales, fit.diagonal)
        paired = float((row @ fit.covariance @ row).clamp_min(0).sqrt())
        fold = float((row @ fit.resampling_covariance @ row).clamp_min(0).sqrt())
        return dict(predicted_reduction=fit.reduction(coordinates),
                    predicted_reduction_se=fit.reduction_se(coordinates),
                    paired_prediction_se=paired, fold_prediction_se=fold,
                    diagonal_predicted_reduction=self.diagonal_bundle.reduction(coordinates),
                    fold_downside=fit.fold_downside(coordinates),
                    fold_predicted_gains=[model.combined.reduction(coordinates) for model in self.fold_fits],
                    fold_predicted_ses=[model.combined.reduction_se(coordinates) for model in self.fold_fits],
                    leave_one_fold_out_predicted_gains=[model.combined.reduction(coordinates)
                                                       for model in self.training_fits])

    def spatial_diagnostics(self):
        """Leave one LOCATION out, retaining the same noises and anchor.

        Unlike noise folds this checks spatial interpolation. Responses are
        correlated, so discrepancy uses the sum of prediction and observation
        SEs. This is descriptive, not an independent goodness-of-fit test.
        """
        rows = []
        for block in self.blocks:
            for point, child in zip(block.coordinates, block.results):
                blocks = []
                for source in self.blocks:
                    kept = [j for j, old in enumerate(source.coordinates) if not torch.equal(old, point)]
                    if kept:
                        blocks.append(FitBlock([source.coordinates[j] for j in kept], source.incumbent,
                                               [source.results[j] for j in kept]))
                if not blocks:
                    continue
                x = torch.cat([design(torch.stack(item.coordinates), self.bundle.combined.scales,
                                       self.bundle.combined.diagonal) for item in blocks])
                if int(torch.linalg.matrix_rank(x)) < coefficient_count(self.bundle.dimension,
                                                                        self.bundle.combined.diagonal):
                    continue
                fit = fit_quadratic(blocks, self.bundle.combined.scales, ridge=self.ridge,
                                    covariance_floor=self.covariance_floor,
                                    diagonal=self.bundle.combined.diagonal,
                                    cross_term_ridge_multiplier=self.cross_term_ridge_multiplier)
                delta, error = paired_difference(child, block.incumbent)
                predicted = fit.reduction(point)
                rows.append(dict(coordinates=point.tolist(), measured_gain=-delta,
                                 predicted_gain=predicted, prediction_se=fit.reduction_se(point),
                                 paired_se=error, error=predicted+delta))
        return dict(locations=rows, rmse=math.sqrt(sum(row["error"]**2 for row in rows)/len(rows)) if rows else None,
                    note="Leave-location-out spatial check; conditional and correlated, not fresh validation")

    def diagnostics(self):
        gradients = torch.stack([fit.combined.gradient*fit.combined.scales for fit in self.fold_fits])
        positive = (gradients > 0).sum(dim=0)
        agreement = torch.maximum(positive, self.fold_count-positive)/self.fold_count
        errors = []
        for fit, blocks in zip(self.training_fits, self.fold_blocks):
            residuals = [fit.combined.reduction(point)+(child.score-block.incumbent.score)
                         for block in blocks for point, child in zip(block.coordinates, block.results)]
            errors.append(math.sqrt(sum(value*value for value in residuals)/len(residuals)))
        require_finite(gradients, "fold gradients")
        return dict(fold_count=self.fold_count, pooled_image_count=self.bundle.combined.image_count,
                    images_per_fold=self.bundle.combined.image_count//self.fold_count,
                    scaled_gradients=gradients.tolist(), slope_sign_agreement=agreement.tolist(),
                    leave_one_fold_out_scaled_gradients=[(fit.combined.gradient*fit.combined.scales).tolist()
                                                        for fit in self.training_fits],
                    held_out_noise_rmse=errors,
                    fold_image_counts=[fit.combined.image_count for fit in self.fold_fits],
                    leave_one_fold_out_image_counts=[fit.combined.image_count for fit in self.training_fits],
                    cross_term_ridge_multiplier=self.cross_term_ridge_multiplier,
                    note="Noise-fold diagnostic at measured/adaptively selected locations; not a spatial holdout or independent confirmation")
