from typing import Optional

import numpy as np

from scipy import interpolate

import jax.numpy as jnp


def pi0est(p: np.ndarray, log, lam: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Estimate pi0 for qvalue calculation"""
    p = p[~np.isnan(p)]  # remove NA values
    m = len(p)

    lam = np.sort(lam)  # guard against user unsorted input
    ll = len(lam)

    # check length:
    assert p.min() >= 0 and p.max() <= 1, "p values not in valid range [0, 1]."
    if 1 < ll < 4:
        log.info("if length of lambda greater than 1, you need at least 4 values.")
    assert lam.min() >= 0 and lam.max() < 1, "ERROR: qvalue_lambda must be within [0, 1)."

    if p.max() < lam.max():
        lam = np.array([0.0])
        ll = 1
        log.info("Warning: maximum p-value is smaller than lambda range. Set lam=0, fall back BH method.")
    # assert p.max() >= lam.max(), (
    #     "ERROR: maximum p-value is smaller than lambda range. "
    #     "Change the range of lambda or use qvalue_truncp() for truncated p-values."
    # )

    # Determines pi0
    if ll == 1:
        pi0 = np.mean(p >= lam[0]) / (1 - lam[0])  # extract value from one element array
        pi0 = np.append(pi0, 1).min()
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        counts = np.array([(p > i).sum() for i in np.arange(0.05, 1.0, 0.05)])
        for val in range(len(lam)):
            pi0.append(counts[val] / (m * (1 - lam[val])))
        pi0 = np.array(pi0)

        # fit a smooth cubic spline between pi0 vs. lam (same length)
        tck = interpolate.splrep(lam, pi0, k=3, s=len(lam))  # add s=len() can smooth out the boundary
        pi0Smooth = interpolate.splev(lam[-1], tck)
        pi0 = np.append(pi0Smooth, 1).min()

        # Or try CubicSpline
        # tck2 = interpolate.CubicSpline(lam, pi0)
        # pi0Smooth = tck2(lam[-1])
        # pi0 = np.append(pi0Smooth, 1).min()

        if verbose:
            log.info(f"qvalues pi0={pi0}, estimated proportion of null features ")

    if pi0 <= 0:
        pi0 = 1.0
        log.info(
            "The estimated pi0 <= 0. Setting the pi0 estimate to be 1. "
            "Check that you have valid p-values or use a different range of lambda."
        )

    assert 0.0 <= pi0 <= 1.0, f"pi0 is not between 0 and 1: {pi0}"

    return pi0


def calculate_qval(
    p: np.ndarray,
    log,
    pi0: Optional[float] = None,
    lam: Optional[np.ndarray] = None,
    fdr_level: float = 0.05,
    lowmem=False,
) -> tuple[np.ndarray, float]:
    """Calculate q value"""

    # TODO: fix shape if remove NA values; need return the same shape as input p
    keep = ~np.isnan(p)
    p = p[keep]
    if np.min(p) < 0 or np.max(p) > 1:
        raise ValueError("P values must be in [0, 1].")
    if fdr_level < 0 or fdr_level > 1:
        raise ValueError("FDR must be in [0, 1].")

    if lam is not None:
        lam = np.sort(lam)  # guard against user unsorted input
    else:
        lam = np.arange(0.05, 1.0, 0.05)

    # estimate pi0
    if pi0 is None:
        pi0 = pi0est(p, log, lam)

    m = len(p)

    if lowmem:
        # low memory version, only uses 1 pv and 1 qv matrices
        qv = np.zeros((len(p),))
        last_pv = p.argmax()
        qv[last_pv] = (pi0 * p[last_pv] * m) / float(m)
        p[last_pv] = -np.inf
        prev_qv = last_pv
        for i in range(int(len(p)) - 2, -1, -1):
            cur_max = p.argmax()
            qv_i = pi0 * m * p[cur_max] / float(i + 1)
            p[cur_max] = -np.inf
            qv_i1 = prev_qv
            qv[cur_max] = jnp.array([qv_i, qv_i1]).min()
            prev_qv = qv[cur_max]

    else:
        p_ordered = np.argsort(p)
        p = p[p_ordered]
        qv = pi0 * m / len(p) * p
        qv[-1] = np.array([qv[-1], 1.0]).min()

        for i in range(len(p) - 2, -1, -1):
            qv[i] = np.array([pi0 * m * p[i] / (i + 1.0), qv[i + 1]]).min()

        # reorder qvalues
        qv_temp = qv.copy()
        qv = np.zeros_like(qv)
        qv[p_ordered] = qv_temp

    # reshape qvalues
    result = np.full_like(keep, fill_value=np.nan, dtype=float)
    result[keep] = qv

    return result, pi0


def estimate_sig_threshold(q_values, p_values, fdr: float):
    sig_mask = q_values <= fdr
    if not np.any(sig_mask):
        # no significant associations
        return np.nan

    # determine global min(p) significance threshold and calculate nominal p-value threshold for each gene
    lb = np.nanmax(p_values[sig_mask])
    non_sig = ~sig_mask
    if not np.any(non_sig):
        ub = np.nanmin(p_values[~sig_mask])
        pthreshold = 0.5 * (lb + ub)
    else:
        pthreshold = lb

    return pthreshold
