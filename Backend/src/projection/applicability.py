import json
import re

import numpy as np
import xarray as xr


def _shift_name(feature_name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", feature_name).strip("_").lower()
    return f"shift_{sanitized}"


def assess_applicability(
    training: xr.Dataset,
    future: xr.Dataset,
    feature_names: list[str],
    threshold_percentile: float = 95.0,
    bootstrap_replicates: int = 200,
    random_seed: int = 42,
    threshold_percentiles: list[float] | None = None,
) -> xr.Dataset:
    """Score future feature-envelope shift against complete training samples.

    The score is the maximum absolute training-standardized feature distance.
    Its cutoff is the requested percentile of training scores. This is an
    interpretable univariate-envelope screen, not a multivariate AOA method.
    Additional percentile thresholds can be supplied to inspect sensitivity.
    """
    if not feature_names:
        raise ValueError("At least one validated model feature is required")
    if threshold_percentiles is None:
        threshold_percentiles = [threshold_percentile]
    else:
        threshold_percentiles = [float(value) for value in threshold_percentiles]
        if not threshold_percentiles:
            raise ValueError("threshold_percentiles must not be empty")
    for value in threshold_percentiles:
        if not 0 < value <= 100:
            raise ValueError("Each threshold percentile must be in (0, 100]")
    if bootstrap_replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    for name in feature_names:
        if name not in training or name not in future:
            raise ValueError(f"Feature {name!r} must exist in training and future datasets")

    train_values = np.stack(
        [np.asarray(training[name].values, dtype=float).ravel() for name in feature_names], axis=1
    )
    complete_training = train_values[np.isfinite(train_values).all(axis=1)]
    if complete_training.shape[0] < 2:
        raise ValueError("At least two complete training samples are required")

    means = complete_training.mean(axis=0)
    scales = complete_training.std(axis=0)
    scales[scales == 0] = 1.0
    training_score = np.max(np.abs((complete_training - means) / scales), axis=1)
    cutoff = float(np.percentile(training_score, threshold_percentile))
    cutoff_by_percentile = {
        float(value): float(np.percentile(training_score, value)) for value in threshold_percentiles
    }

    standardized = [
        (future[name] - float(mean)) / float(scale)
        for name, mean, scale in zip(feature_names, means, scales)
    ]
    stacked = xr.concat(standardized, dim=xr.IndexVariable("feature", feature_names))
    shift_score = np.abs(stacked).max("feature", skipna=False).rename("covariate_shift_score")
    feature_diagnostics = {}
    for name, mean, scale in zip(feature_names, means, scales):
        feature_shift = np.abs((future[name] - float(mean)) / float(scale)).rename(_shift_name(name))
        feature_diagnostics[_shift_name(name)] = feature_shift
    base_mask = (shift_score <= cutoff).astype("uint8").rename("applicability_mask")

    rng = np.random.default_rng(random_seed)
    applicable_replicates = np.zeros(shift_score.shape, dtype=np.float64)
    future_values = [np.asarray(future[name].values, dtype=float) for name in feature_names]
    for _ in range(bootstrap_replicates):
        sample = complete_training[rng.integers(0, complete_training.shape[0], complete_training.shape[0])]
        sample_means = sample.mean(axis=0)
        sample_scales = sample.std(axis=0)
        sample_scales[sample_scales == 0] = 1.0
        sample_score = np.max(np.abs((sample - sample_means) / sample_scales), axis=1)
        sample_cutoff = np.percentile(sample_score, threshold_percentile)
        future_score = np.maximum.reduce([
            np.abs((values - mean) / scale)
            for values, mean, scale in zip(future_values, sample_means, sample_scales)
        ])
        applicable_replicates += np.isfinite(future_score) & (future_score <= sample_cutoff)
    probability = (applicable_replicates / bootstrap_replicates).astype(float)
    uncertainty = np.sqrt(probability * (1.0 - probability))
    probability_da = xr.DataArray(
        probability, dims=shift_score.dims, coords=shift_score.coords,
        name="applicability_probability",
    )
    uncertainty_da = xr.DataArray(
        uncertainty, dims=shift_score.dims, coords=shift_score.coords,
        name="applicability_uncertainty",
    )

    threshold_sensitivity = {}
    for percentile in threshold_percentiles:
        threshold_value = cutoff_by_percentile[float(percentile)]
        threshold_mask = (shift_score <= threshold_value).astype("uint8")
        threshold_sensitivity[f"applicability_mask_{int(percentile)}"] = threshold_mask.rename(
            f"applicability_mask_{int(percentile)}"
        )
        threshold_sensitivity[f"applicability_percentage_{int(percentile)}"] = float(
            threshold_mask.mean().item()
        ) * 100.0

    result = xr.Dataset({
        "covariate_shift_score": shift_score,
        "applicability_mask": base_mask,
        "applicability_probability": probability_da,
        "applicability_uncertainty": uncertainty_da,
        **feature_diagnostics,
        **threshold_sensitivity,
    })
    fraction = float(base_mask.mean().item())
    result.attrs.update({
        "method": "maximum absolute standardized feature distance",
        "threshold_percentile": threshold_percentile,
        "threshold": cutoff,
        "threshold_percentiles": json.dumps([float(v) for v in threshold_percentiles]),
        "threshold_cutoffs": json.dumps({str(v): cutoff_by_percentile[float(v)] for v in threshold_percentiles}, sort_keys=True),
        "feature_names": json.dumps(feature_names),
        "applicability_fraction": fraction,
        "applicability_percentage": fraction * 100.0,
        "applicability_uncertainty_method": (
            "Bernoulli standard deviation of bootstrap applicability classifications"
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "random_seed": random_seed,
        "mask_convention": "1=within training feature envelope; 0=outside",
    })
    return result