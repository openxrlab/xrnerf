import math

import numpy as np
import torch


def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """Piecewise-Constant PDF sampling from sorted bins."""
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    device = weights.device
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.maximum(torch.tensor(0).to(device), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.minimum(
        torch.tensor(1).to(device), torch.cumsum(pdf[..., :-1], dim=-1))
    cdf = torch.cat([
        torch.zeros(list(cdf.shape[:-1]) + [1]).to(device), cdf,
        torch.ones(list(cdf.shape[:-1]) + [1]).to(device)
    ], -1)

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = torch.arange(num_samples) * s

        u = u + torch.rand(list(cdf.shape[:-1]) + [num_samples]) * (
            s - torch.finfo(torch.float32).eps)

        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.tensor(1. - torch.finfo(torch.float32).eps))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps,
                           num_samples)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
    u = u.to(device)
    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.

    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]),
                       -2)[0]
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]),
                       -2)[0]
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    device = d.device
    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = torch.maximum(
        torch.tensor(1e-10).to(device), torch.sum(d**2, dim=-1, keepdim=True))

    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1]).to(device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov)."""
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                          (3 * mu**2 + hw**2)**2)
        r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                                  (hw**4) / (3 * mu**2 + hw**2))
    else:
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov)."""
    t_mean = (t0 + t1) / 2
    r_var = radius**2 / 4
    t_var = (t1 - t0)**2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(z_vals, origins, directions, radii, ray_shape, diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it."""
    t0 = z_vals[..., :-1]
    t1 = z_vals[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


def sample_along_rays(data, ray_shape):
    """Stratified sampling along the rays."""
    origins = data['rays_o']
    directions = data['rays_d']
    radii = data['radii']
    z_vals = data['z_vals']

    means, covs = cast_rays(z_vals, origins, directions, radii, ray_shape)

    data['z_vals'] = z_vals
    data['samples'] = (means, covs)
    return data


def resample_along_rays(data, randomized, ray_shape, resample_padding):
    """Resampling."""
    origins = data['rays_o']
    directions = data['rays_d']
    radii = data['radii']
    z_vals = data['z_vals']
    weights = data['weights']

    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ], -1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    # Add in a constant (the sampling function will renormalize the PDF).
    weights = weights_blur + resample_padding

    new_z_vals = sorted_piecewise_constant_pdf(
        z_vals,
        weights,
        z_vals.shape[-1],
        randomized,
    )
    new_z_vals = new_z_vals.detach()
    means, covs = cast_rays(new_z_vals, origins, directions, radii, ray_shape)

    data['z_vals'] = new_z_vals
    data['samples'] = (means, covs)
    return data
