"""Run one correlation mode for the multicollinearity experiment (Table 5, Appendix B.6).

One invocation fits one deterministic model, writes one exact cell, and writes
all requested sampled-repeat cells. Completed outputs are reused on rerun.
The config budget is a permutation budget: 256 gives 128 forward/reverse pairs.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.conformal.cq import calibration_width_adjustment_from_models  # noqa: E402
from src.conformal.quantile import QuantileModelConfig, fit_quantile_models  # noqa: E402
from src.shapley.shapley_global import compute_global_shapley, masked_calib_matrix  # noqa: E402
from src.shapley.shapley_local import shapley_local_single  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
from sampler import antithetic_perm_global, antithetic_perm_local  # noqa: E402


def rho_label(rho: float) -> str:
    return f"cor_{int(round(100 * rho)):02d}"


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        cfg = json.load(f)
    required = {
        "d", "n_samples", "n_calib", "n_test", "data_seed", "alpha",
        "backend", "budget", "sampler_repeats", "correlations",
    }
    missing = required.difference(cfg)
    if missing:
        raise ValueError(f"Config missing keys: {sorted(missing)}")
    if cfg["budget"] <= 0 or cfg["budget"] % 2:
        raise ValueError("budget must be a positive even total-permutation count")
    return cfg


def generate_correlated_synthetic(d: int, n: int, rho_x: float, seed: int):
    """Runtime-sweep DGP (experiments/runtime) with Gaussian-copula dependence and uniform marginals."""
    if not 0.0 <= rho_x < 1.0:
        raise ValueError("rho_x must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    U = rng.random((n, d))
    rho_z = float(2.0 * np.sin(np.pi * rho_x / 6.0))
    if rho_x == 0.0:
        X = U
        covariance = np.eye(d)
    else:
        covariance = (1.0 - rho_z) * np.eye(d) + rho_z * np.ones((d, d))
        L = np.linalg.cholesky(covariance)
        eps = np.finfo(float).eps
        Z = ndtri(np.clip(U, eps, 1.0 - eps))
        X = ndtr(Z @ L.T)
    sigma = 0.05 + 0.95 * (X[:, 0] + X[:, d - 1]) / 2.0
    y = rng.normal(0.0, sigma, n)
    return X, y, sigma, rho_z, covariance


def split_indices(n: int, seed: int, n_calib: int, n_test: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = n - n_calib - n_test
    if n_train <= 0:
        raise ValueError("n_samples must exceed n_calib + n_test")
    return (
        idx[:n_train],
        idx[n_train:n_train + n_calib],
        idx[n_train + n_calib:n_train + n_calib + n_test],
    )


def sampler_seed(data_seed: int, repeat: int) -> int:
    # Repeat zero is exactly the runtime sweep's sampler seed. Later streams are deterministic and disjoint.
    return int(data_seed + 100_000 + repeat * 1_000_000)


def _local_value(x: np.ndarray, q_lo, q_hi) -> float:
    xx = np.asarray(x, dtype=float).reshape(1, -1)
    return float(q_hi.predict(xx)[0] - q_lo.predict(xx)[0])


def game_endpoints(X_ca, y_ca, X_te, baseline, q_lo, q_hi, alpha):
    d = X_ca.shape[1]
    vg_empty = float(calibration_width_adjustment_from_models(
        masked_calib_matrix(X_ca, baseline, ()), y_ca, q_lo, q_hi, alpha=alpha
    ))
    vg_full = float(calibration_width_adjustment_from_models(
        masked_calib_matrix(X_ca, baseline, tuple(range(d))),
        y_ca, q_lo, q_hi, alpha=alpha,
    ))
    vl_empty = _local_value(baseline, q_lo, q_hi)
    vl_delta = np.asarray(
        [_local_value(x, q_lo, q_hi) - vl_empty for x in X_te], dtype=float
    )
    return vg_full - vg_empty, vl_delta


def array_digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        value = np.ascontiguousarray(arr)
        h.update(str(value.shape).encode())
        h.update(value.dtype.str.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def atomic_savez(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as f:
        tmp = Path(f.name)
    try:
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def validate_cached_cell(path: Path, cfg: dict, rho_x: float) -> None:
    try:
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata_json"]))
    except Exception as exc:
        raise RuntimeError(f"Unreadable cached cell {path}: {exc}") from exc
    expected = json.dumps(cfg, sort_keys=True)
    if metadata.get("config_json") != expected:
        raise RuntimeError(
            f"Cached cell config mismatch at {path}; move it aside before rerunning"
        )
    if not np.isclose(float(metadata.get("rho_x", np.nan)), rho_x):
        raise RuntimeError(f"Cached cell rho mismatch at {path}")


def common_payload(cfg, rho_x, rho_z, covariance, empirical_corr, tr, ca, te,
                   baseline, prediction_digest):
    eig = np.linalg.eigvalsh(covariance)
    metadata = {
        "config_name": cfg.get("name", "unknown"),
        "d": int(cfg["d"]),
        "n_samples": int(cfg["n_samples"]),
        "n_train": int(len(tr)),
        "n_calib": int(len(ca)),
        "n_test": int(len(te)),
        "data_seed": int(cfg["data_seed"]),
        "alpha": float(cfg["alpha"]),
        "backend": str(cfg["backend"]),
        "budget": int(cfg["budget"]),
        "rho_x": float(rho_x),
        "rho_z": float(rho_z),
        "empirical_pearson": float(empirical_corr),
        "latent_condition_number": float(eig[-1] / eig[0]),
        "split_digest": array_digest(tr, ca, te),
        "baseline_digest": array_digest(baseline),
        "prediction_digest": prediction_digest,
        "config_json": json.dumps(cfg, sort_keys=True),
    }
    return {"metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))}


def run_mode(cfg: dict, rho_x: float, out_dir: Path) -> None:
    label = rho_label(rho_x)
    cells_dir = out_dir / "cells"
    exact_path = cells_dir / f"exact_{label}.npz"
    sample_paths = [
        cells_dir / f"sampled_{label}_repeat{r}.npz"
        for r in range(int(cfg["sampler_repeats"]))
    ]
    need_exact = not exact_path.is_file()
    needed_repeats = [r for r, p in enumerate(sample_paths) if not p.is_file()]
    for path in [exact_path, *sample_paths]:
        if path.is_file():
            validate_cached_cell(path, cfg, rho_x)
    if not need_exact and not needed_repeats:
        print(f"[{label}] all outputs exist; skipping", flush=True)
        return

    d = int(cfg["d"])
    seed = int(cfg["data_seed"])
    X, y, sigma, rho_z, covariance = generate_correlated_synthetic(
        d, int(cfg["n_samples"]), rho_x, seed
    )
    tr, ca, te = split_indices(
        len(X), seed, int(cfg["n_calib"]), int(cfg["n_test"])
    )
    X_tr, y_tr = X[tr], y[tr]
    X_ca, y_ca = X[ca], y[ca]
    X_te = X[te]
    baseline = np.nanmean(X_ca, axis=0)

    model_cfg = QuantileModelConfig(
        alpha_lower=float(cfg["alpha"]) / 2.0,
        alpha_upper=1.0 - float(cfg["alpha"]) / 2.0,
        random_state=seed,
        backend=str(cfg["backend"]),
    )
    print(f"[{label}] fitting {cfg['backend']} on {len(tr)} rows", flush=True)
    q_lo, q_hi = fit_quantile_models(X_tr, y_tr, model_cfg)
    prediction_digest = array_digest(q_lo.predict(X_te), q_hi.predict(X_te))
    off_delta, local_delta = game_endpoints(
        X_ca, y_ca, X_te, baseline, q_lo, q_hi, float(cfg["alpha"])
    )
    upper = np.triu_indices(d, 1)
    empirical_corr = float(np.corrcoef(X, rowvar=False)[upper].mean())
    common = common_payload(
        cfg, rho_x, rho_z, covariance, empirical_corr, tr, ca, te,
        baseline, prediction_digest,
    )

    if need_exact:
        print(f"[{label}] exact global", flush=True)
        start = time.perf_counter()
        phi_g, _ = compute_global_shapley(
            X_calib=X_ca, y_calib=y_ca, q_lo=q_lo, q_hi=q_hi,
            alpha=float(cfg["alpha"]), baseline=baseline, n_jobs=1,
            show_progress=False,
        )
        offline_sec = time.perf_counter() - start
        phi_l = np.zeros((len(X_te), d), dtype=float)
        local_secs = np.zeros(len(X_te), dtype=float)
        for i, x in enumerate(X_te):
            print(f"[{label}] exact local {i + 1}/{len(X_te)}", flush=True)
            start = time.perf_counter()
            phi_l[i] = shapley_local_single(
                x, baseline, q_lo, q_hi, impute_nan_x0=True
            )
            local_secs[i] = time.perf_counter() - start
        atomic_savez(
            exact_path,
            method=np.asarray("exact"), rho_x=rho_x, phi_global=phi_g,
            phi_local=phi_l, phi_total=phi_l + phi_g[None, :],
            offline_sec=offline_sec, online_query_sec=local_secs,
            online_total_sec=float(local_secs.sum()),
            eff_err_global=float(phi_g.sum() - off_delta),
            eff_err_local=phi_l.sum(axis=1) - local_delta,
            X_test=X_te, baseline=baseline, test_indices=te,
            sigma_test=sigma[te], **common,
        )
        print(f"[{label}] wrote {exact_path}", flush=True)
    else:
        with np.load(exact_path, allow_pickle=False) as z:
            if not np.array_equal(z["test_indices"], te):
                raise RuntimeError(f"{exact_path} has incompatible test rows")

    for repeat in needed_repeats:
        seed_s = sampler_seed(seed, repeat)
        rng = np.random.default_rng(seed_s)
        print(f"[{label}] sampled repeat={repeat} seed={seed_s}", flush=True)
        start = time.perf_counter()
        phi_g = antithetic_perm_global(
            X_calib=X_ca, y_calib=y_ca, q_lo=q_lo, q_hi=q_hi,
            alpha=float(cfg["alpha"]), baseline=baseline,
            budget=int(cfg["budget"]), rng=rng,
        )
        offline_sec = time.perf_counter() - start
        phi_l = np.zeros((len(X_te), d), dtype=float)
        local_secs = np.zeros(len(X_te), dtype=float)
        for i, x in enumerate(X_te):
            start = time.perf_counter()
            phi_l[i] = antithetic_perm_local(
                x, baseline, q_lo, q_hi, budget=int(cfg["budget"]), rng=rng
            )
            local_secs[i] = time.perf_counter() - start
        atomic_savez(
            sample_paths[repeat],
            method=np.asarray("sampled"), rho_x=rho_x,
            sampler_repeat=repeat, sampler_seed=seed_s,
            phi_global=phi_g, phi_local=phi_l,
            phi_total=phi_l + phi_g[None, :],
            offline_sec=offline_sec, online_query_sec=local_secs,
            online_total_sec=float(local_secs.sum()),
            eff_err_global=float(phi_g.sum() - off_delta),
            eff_err_local=phi_l.sum(axis=1) - local_delta,
            X_test=X_te, baseline=baseline, test_indices=te,
            sigma_test=sigma[te], **common,
        )
        print(f"[{label}] wrote {sample_paths[repeat]}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(EXP_DIR / "config_full.json"))
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--out-dir", default=str(EXP_DIR / "results"))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if not any(np.isclose(args.rho, value) for value in cfg["correlations"]):
        raise ValueError(f"rho={args.rho} is not in config correlations")
    run_mode(cfg, float(args.rho), Path(args.out_dir))


if __name__ == "__main__":
    main()
