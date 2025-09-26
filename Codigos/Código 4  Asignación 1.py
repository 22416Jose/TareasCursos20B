
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple

# ---------------------- Configuración ----------------------
BASE_DIR = Path(".")  # ajusta si los CSV están en otra carpeta
FILES = [
    "PM25_Acualago_Referencia.csv",
    "PM25_Caldas_Referencia.csv",
    "PM25_ColegioNormal_2018-11_a_2019-08_Calibrar.csv",
    "PM25_Giron_Referencia.csv",
    "PM25_Normal_Referencia.csv",
    "PM25_Pilar_Referencia.csv",
]
LOWCOST_NAME = "PM25_ColegioNormal_2018-11_a_2019-08_Calibrar.csv"
W_HOURS = 72
MIN_FRAC = 0.5
TARGET_COVERAGE = 95.0
TARGET_R2 = 0.95

def load_pm25_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = pd.to_datetime(df["Fecha"].astype(str) + " " + df["Hora"].astype(str),
                        errors="coerce", dayfirst=True)
    df = df.assign(datetime=dt).dropna(subset=["datetime"])
    df = df[["datetime", "PM2.5"]].rename(columns={"PM2.5": "PM25"})
    return df.sort_values("datetime").reset_index(drop=True)

def hourly_mean(df: pd.DataFrame) -> pd.Series:
    return df.set_index("datetime")["PM25"].resample("H").mean()

def rolling_center(series: pd.Series, w_hours: int = W_HOURS, min_frac: float = MIN_FRAC) -> pd.Series:
    minp = max(1, int(w_hours * min_frac))
    return series.rolling(f"{w_hours}H", center=True, min_periods=minp).mean()

def build_xy(lowcost_hourly: pd.Series, ref_hourly: pd.Series) -> pd.DataFrame:
    x = rolling_center(lowcost_hourly, W_HOURS, MIN_FRAC).rename("x")
    y = rolling_center(ref_hourly, W_HOURS, MIN_FRAC).rename("y")
    return pd.concat([y, x], axis=1).dropna()

def train_test_metrics(df_xy: pd.DataFrame, m_train: int) -> Optional[Dict[str, float]]:
    n = len(df_xy)
    if m_train < 2 or m_train >= n:
        return None
    train = df_xy.iloc[:m_train].copy()
    test  = df_xy.iloc[m_train:].copy()
    Sxy = float(np.sum(train["x"] * train["y"]))
    Sxx = float(np.sum(train["x"] ** 2))
    alpha = Sxy / Sxx if Sxx > 0 else np.nan
    yhat_te = alpha * test["x"]
    resid = test["y"] - yhat_te
    SSE_te = float(np.sum(resid**2))
    SST0_te = float(np.sum(test["y"]**2))
    R2_te = 1 - SSE_te / SST0_te if SST0_te > 0 else np.nan
    RMSE_te = float(np.sqrt(np.mean(resid**2)))
    tol = np.maximum(5.0, 0.20 * np.abs(test["y"].values))
    valid = np.abs(resid.values) <= tol
    coverage = float(valid.mean() * 100.0)
    x_valid = test["x"].values[valid]
    x_min = float(np.min(x_valid)) if x_valid.size else np.nan
    x_max = float(np.max(x_valid)) if x_valid.size else np.nan
    return {
        "alpha_train": float(alpha),
        "R2_test": float(R2_te),
        "RMSE_test": float(RMSE_te),
        "coverage_test_%": float(coverage),
        "x_min_valid": x_min,
        "x_max_valid": x_max,
    }

def scan_station(df_xy: pd.DataFrame,
                 target_cov: float = TARGET_COVERAGE,
                 target_r2: float = TARGET_R2) -> Tuple[pd.DataFrame, Dict[str, float]]:
    n = len(df_xy)
    m_start = max(100, int(0.05 * n))
    m_grid = list(range(m_start, max(m_start + 1, n - 100), 100))
    if (n // 2) not in m_grid:
        m_grid.append(n // 2)
    m_grid = sorted({m for m in m_grid if 2 <= m < n})

    rows = []
    m_min = None
    res_at_min = None
    best_width = -np.inf
    m_best = None
    res_at_best = None

    for m in m_grid:
        res = train_test_metrics(df_xy, m)
        if res is None:
            continue
        width = (res["x_max_valid"] - res["x_min_valid"]) if (np.isfinite(res["x_max_valid"]) and np.isfinite(res["x_min_valid"])) else np.nan
        rows.append({"m": m, **res, "x_width_valid": width})
        if m_min is None and (res["coverage_test_%"] >= target_cov) and (res["R2_test"] >= target_r2):
            m_min = m; res_at_min = res
        if (res["coverage_test_%"] >= target_cov) and (res["R2_test"] >= target_r2) and np.isfinite(width):
            if width > best_width:
                best_width = width; m_best = m; res_at_best = res

    hist = pd.DataFrame(rows).sort_values("m").reset_index(drop=True)
    summary = {
        "N_total": n,
        "m_min": m_min,
        "alpha@min": (None if res_at_min is None else res_at_min["alpha_train"]),
        "R2_test@min": (None if res_at_min is None else res_at_min["R2_test"]),
        "RMSE_test@min": (None if res_at_min is None else res_at_min["RMSE_test"]),
        "Cobertura@min_%": (None if res_at_min is None else res_at_min["coverage_test_%"]),
        "x_min@min": (None if res_at_min is None else res_at_min["x_min_valid"]),
        "x_max@min": (None if res_at_min is None else res_at_min["x_max_valid"]),
        "m_mejor_alcance": m_best,
        "x_min@mejor": (None if res_at_best is None else res_at_best["x_min_valid"]),
        "x_max@mejor": (None if res_at_best is None else res_at_best["x_max_valid"]),
    }
    return hist, summary

def plot_station(hist: pd.DataFrame, station: str, outdir: Path,
                 target_cov: float = TARGET_COVERAGE,
                 m_min: Optional[int] = None,
                 m_best: Optional[int] = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # 1) Cobertura vs m
    fig1 = plt.figure(figsize=(6.6, 4.5))
    plt.plot(hist["m"], hist["coverage_test_%"], marker="o")
    plt.axhline(target_cov, linestyle="--")
    if m_min is not None:
        plt.axvline(m_min, linestyle="--")
        plt.text(m_min, target_cov + 0.5, f"m_min={m_min}", rotation=90, va="bottom", fontsize=9)
    plt.xlabel("Tamaño de entrenamiento m (horas/puntos)")
    plt.ylabel("Cobertura en prueba (%)")
    plt.title(f"{station}: Cobertura vs m (tolerancia = max(5, 0.2·y))")
    plt.tight_layout()
    (outdir / f"{station}_scan_coverage_vs_m.png").write_bytes(fig1.canvas.tostring_png())
    plt.savefig(outdir / f"{station}_scan_coverage_vs_m.png", dpi=200)
    plt.close(fig1)
    # 2) Rango válido vs m
    fig2 = plt.figure(figsize=(6.6, 4.5))
    plt.plot(hist["m"], hist["x_min_valid"], marker="o", label="x_min válido (test)")
    plt.plot(hist["m"], hist["x_max_valid"], marker="o", label="x_max válido (test)")
    if m_best is not None:
        plt.axvline(m_best, linestyle="--")
        try:
            ymax = np.nanmax(hist["x_max_valid"].values)
        except Exception:
            ymax = 0.0
        plt.text(m_best, ymax, f"m_mejor={m_best}", rotation=90, va="bottom", fontsize=9)
    plt.xlabel("Tamaño de entrenamiento m (horas/puntos)")
    plt.ylabel("x válido en prueba (µg/m³)")
    plt.title(f"{station}: Rango válido de x en prueba vs m")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{station}_scan_range_vs_m.png", dpi=200)
    plt.close(fig2)

def main(base_dir: Path = BASE_DIR, out_dir: Path = Path(".")) -> None:
    data = {name: load_pm25_csv(base_dir / name) for name in FILES}
    x_hour = hourly_mean(data[LOWCOST_NAME])
    refs = [k for k in data.keys() if k != LOWCOST_NAME]
    refs_hour = {k: hourly_mean(data[k]) for k in refs}

    summaries = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in sorted(refs):
        station = key.replace("PM25_", "").replace("_Referencia.csv", "")
        df_xy = build_xy(x_hour, refs_hour[key])
        hist, summary = scan_station(df_xy, TARGET_COVERAGE, TARGET_R2)
        summaries.append({"Referencia": station, **summary})
        plot_station(hist, station, out_dir, TARGET_COVERAGE, summary["m_min"], summary["m_mejor_alcance"])
        print(f"[{station}] listo.")

    summary_df = pd.DataFrame(summaries).sort_values("Referencia").reset_index(drop=True)
    (out_dir / "min_set_y_max_alcance_resumen.csv").write_text(summary_df.to_csv(index=False), encoding="utf-8")
    print("OK. Archivos generados en:", out_dir.resolve())
