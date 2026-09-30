"""Hurst exponent of a log-price series, judged against its own shuffled null.

For a self-similar process x (here: log price) the spread of increments grows as

    std(x[t + lag] - x[t])  ~  lag ** H

so H is the slope of log(std) vs log(lag). This is the variance-of-increments
estimator (not R/S). H = 0.5: no memory (random walk); H > 0.5: persistent
(moves tend to continue); H < 0.5: anti-persistent (moves tend to reverse).

Finite samples bias H below 0.5 even for a pure random walk (about 0.42 at
N=400, 0.47 at N=1000, 0.49 at N=2900 with lags 2..99), and the scatter is
0.04-0.12. So each series is compared with its own shuffled null, not with 0.5.

Run `python hurst.py` for a self-check on synthetic series with known H.
"""
import numpy as np

# Lags in candles: the time scale is lag x interval (2-99 days for 1d candles).
LAGS = np.arange(2, 100)


def hurst_exponent(x, lags=LAGS):
    """H of log-price x. x may be 2-D (one series per row): returns one H per row."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] < 2 * lags[-1]:
        return np.nan
    spread = np.array([np.std(x[..., lag:] - x[..., :-lag], axis=-1) for lag in lags])
    return np.polyfit(np.log(lags), np.log(spread), 1)[0]


def null_test(x, n_shuffle=2000, seed=0):
    """Compare H of log-price x with H of the same returns in random order.

    Shuffling keeps N, the return distribution (fat tails) and the total drift,
    and destroys only time ordering: memory, and also volatility clustering,
    which in GARCH(1,1) simulations does not shift H.
    Returns (H, null_mean, null_sd, p) with a two-sided permutation p-value.
    """
    x = np.asarray(x, dtype=float)
    h = hurst_exponent(x)
    if np.isnan(h):
        return h, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.tile(np.diff(x), (n_shuffle, 1)), axis=1)
    null = hurst_exponent(np.cumsum(np.pad(shuffled, ((0, 0), (1, 0))), axis=1))
    k = min(np.sum(null <= h), np.sum(null >= h))
    p = min(1.0, 2 * (k + 1) / (n_shuffle + 1))
    return h, null.mean(), null.std(), p


def verdict(h, null_mean, p, alpha=0.05):
    if p < alpha:
        return "persistent" if h > null_mean else "anti-persistent"
    return "no memory detected"


def _fgn(n, H, rng):
    """Exact fractional Gaussian noise (Davies-Harte): increments of a process with Hurst H."""
    k = np.arange(n + 1)
    g = 0.5 * ((k + 1.0) ** (2 * H) - 2 * k ** (2.0 * H) + np.abs(k - 1.0) ** (2 * H))
    lam = np.fft.fft(np.r_[g, g[-2:0:-1]]).real
    w = rng.normal(size=2 * n) + 1j * rng.normal(size=2 * n)
    return np.fft.fft(np.sqrt(np.maximum(lam, 0) / (4 * n)) * w)[:n].real


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1. Known H is recovered (long series, where finite-N bias is small).
    for H in (0.3, 0.7):
        est = np.mean([hurst_exponent(np.cumsum(_fgn(2900, H, rng))) for _ in range(20)])
        print(f"fGn H={H}: mean estimate {est:.3f}")
        assert abs(est - H) < 0.05

    # 2. Random walk: the null test flags ~alpha of series, not more.
    flagged = np.mean([null_test(np.cumsum(rng.normal(size=1000)), 200, seed=i)[3] < 0.05
                       for i in range(100)])
    print(f"random walk N=1000: flagged fraction {flagged:.2f} (expect ~0.05)")
    assert flagged <= 0.12

    # 3. Real memory is detected, on the correct side.
    for H, side in ((0.3, "anti-persistent"), (0.7, "persistent")):
        hits = 0
        for i in range(20):
            h, null_mean, _, p = null_test(np.cumsum(_fgn(2000, H, rng)), 200, seed=i)
            hits += verdict(h, null_mean, p) == side
        print(f"fGn H={H} N=2000: detected as {side} in {hits}/20")
        assert hits >= 16

    print("hurst.py self-check OK")
