import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# -------------------------------
# 1) Archivos de entrada
# -------------------------------
# Ubica estos archivos en el mismo directorio que este script, o ajusta las rutas.
paths = [
    "PM25_Acualago_Referencia.csv",
    "PM25_Caldas_Referencia.csv",
    "PM25_ColegioNormal_2018-11_a_2019-08_Calibrar.csv",  # bajo costo (x)
    "PM25_Giron_Referencia.csv",
    "PM25_Normal_Referencia.csv",
    "PM25_Pilar_Referencia.csv",
]
LOWCOST = "PM25_ColegioNormal_2018-11_a_2019-08_Calibrar.csv"
W = 72  # horas
MINP = max(1, int(W * 0.5))  # >= 50% de datos en ventana

# -------------------------------
# 2) Utilidades
# -------------------------------
def load_pm25(path: str) -> pd.DataFrame:
    """Lee CSV con columnas ['Fecha','Hora','PM2.5'] y devuelve DataFrame
    con ['datetime','PM25'] ordenado cronológicamente."""
    df = pd.read_csv(path)
    dt = pd.to_datetime(df["Fecha"].astype(str) + " " + df["Hora"].astype(str),
                        errors="coerce", dayfirst=True)
    df = df.assign(datetime=dt).dropna(subset=["datetime"])
    df = (df[["datetime", "PM2.5"]]
          .rename(columns={"PM2.5": "PM25"})
          .sort_values("datetime")
          .reset_index(drop=True))
    return df

def hourly_mean(df: pd.DataFrame) -> pd.Series:
    """Promedio por hora de PM25 con índice datetime."""
    return df.set_index("datetime")["PM25"].resample("H").mean()

def rolling_center(series: pd.Series, w: int = W, minp: int = MINP) -> pd.Series:
    """Promedio móvil centrado temporal (ventana de w horas)."""
    return series.rolling(f"{w}H", center=True, min_periods=minp).mean()

def build_aligned_xy(lowcost_hourly: pd.Series, ref_hourly: pd.Series) -> pd.DataFrame:
    """Devuelve DataFrame con columnas ['y','x'] (ref, bajo costo) ya suavizadas y alineadas."""
    y = rolling_center(ref_hourly, W, MINP).rename("y")
    x = rolling_center(lowcost_hourly, W, MINP).rename("x")
    return pd.concat([y, x], axis=1).dropna()

def split_calibration_metrics(df_xy: pd.DataFrame) -> dict:
    """Divide por mitad temporal: entrena alpha (modelo sin intercepto) en 1ª mitad,
    evalúa métricas y tolerancia en 2ª mitad. Devuelve diccionario con resultados."""
    n = len(df_xy)
    if n < 2:
        return {
            'N_train': 0, 'N_test': 0, 'alpha_train': np.nan,
            'R2_train': np.nan, 'RMSE_train': np.nan,
            'R2_test': np.nan, 'RMSE_test': np.nan,
            'Cobertura_test_%': np.nan,
            'x_valido_min_test': np.nan, 'x_valido_max_test': np.nan,
            'Bloque_valido_test_h': 0.0,
            'Bloque_valido_test_inicio': None, 'Bloque_valido_test_fin': None
        }
    m = n // 2
    train = df_xy.iloc[:m].copy()
    test  = df_xy.iloc[m:].copy()

    # Ajuste forzado al origen en TRAIN
    Sxy = np.sum(train["x"] * train["y"])
    Sxx = np.sum(train["x"] * train["x"])
    alpha = Sxy / Sxx if Sxx > 0 else np.nan

    # Métricas en TRAIN
    yhat_tr = alpha * train["x"]
    resid_tr = train["y"] - yhat_tr
    SSE_tr = np.sum(resid_tr**2)
    SST0_tr = np.sum(train["y"]**2)
    R2_tr = 1 - SSE_tr / SST0_tr if SST0_tr > 0 else np.nan
    RMSE_tr = float(np.sqrt(np.mean(resid_tr**2)))

    # Métricas en TEST
    yhat_te = alpha * test["x"]
    resid_te = test["y"] - yhat_te
    SSE_te = np.sum(resid_te**2)
    SST0_te = np.sum(test["y"]**2)
    R2_te = 1 - SSE_te / SST0_te if SST0_te > 0 else np.nan
    RMSE_te = float(np.sqrt(np.mean(resid_te**2)))

    # Tolerancia y cobertura en TEST
    tol_te = np.maximum(5.0, 0.20 * np.abs(test["y"]))
    valid_te = np.abs(resid_te) <= tol_te
    coverage = valid_te.mean() * 100.0
    x_min = test.loc[valid_te, "x"].min() if valid_te.any() else np.nan
    x_max = test.loc[valid_te, "x"].max() if valid_te.any() else np.nan

    # Bloque válido continuo más largo (en TEST)
    runs = []
    start = None
    for t, ok in valid_te.items():
        if ok and start is None:
            start = t
        if (not ok) and start is not None:
            prev = valid_te.index[valid_te.index.get_loc(t) - 1]
            runs.append((start, prev))
            start = None
    if start is not None:
        runs.append((start, valid_te.index[-1]))

    if runs:
        longest = max(runs, key=lambda ab: (ab[1] - ab[0]).total_seconds())
        longest_h = (longest[1] - longest[0]).total_seconds() / 3600.0 + 1.0
        a, b = longest
    else:
        longest_h = 0.0
        a = b = None

    return {
        'N_train': len(train), 'N_test': len(test),
        'alpha_train': float(alpha),
        'R2_train': float(R2_tr), 'RMSE_train': RMSE_tr,
        'R2_test': float(R2_te), 'RMSE_test': RMSE_te,
        'Cobertura_test_%': float(coverage),
        'x_valido_min_test': float(x_min) if pd.notnull(x_min) else np.nan,
        'x_valido_max_test': float(x_max) if pd.notnull(x_max) else np.nan,
        'Bloque_valido_test_h': float(longest_h),
        'Bloque_valido_test_inicio': a, 'Bloque_valido_test_fin': b
    }

# -------------------------------
# 3) Cargar datos y construir pares x,y (72 h)
# -------------------------------
datasets = {Path(p).name: load_pm25(p) for p in paths}
x_hour = hourly_mean(datasets[LOWCOST])
refs = [k for k in datasets if k != LOWCOST]
refs_hour = {k: hourly_mean(datasets[k]) for k in refs}

# -------------------------------
# 4) Cálculo por estación (split mitad--mitad)
# -------------------------------
per_station = {}
rows = []
for ref_key in sorted(refs):
    df_xy = build_aligned_xy(x_hour, refs_hour[ref_key])
    metrics = split_calibration_metrics(df_xy)
    station = ref_key.replace("PM25_", "").replace("_Referencia.csv", "")
    per_station[station] = {'df': df_xy, **metrics}
    rows.append({'Referencia': station, **metrics})

split_df = pd.DataFrame(rows).sort_values("Referencia").reset_index(drop=True)


# -------------------------------
# 5) Figuras de PRUEBA (mitad 2) por estación
# -------------------------------

for station, info in sorted(per_station.items(), key=lambda kv: kv[0]):
    df = info['df']
    m = len(df)//2
    test = df.iloc[m:].copy()
    alpha = float(info['alpha_train'])
    R2_test = float(info['R2_test'])
    RMSE_test = float(info['RMSE_test'])
    coverage = float(info['Cobertura_test_%'])
    N_test = int(info['N_test'])

    x = test['x'].values
    y = test['y'].values
    yhat = alpha * x
    resid = y - yhat

    # 1) Scatter (test) con fit y 1:1
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    plt.figure(figsize=(6.5,6.2))
    plt.plot(x, y, '.', alpha=0.6, label="Datos (test, 72 h)")
    plt.plot(xx, alpha*xx, '-', label=f"Fit: y={alpha:.4f}x (entreno)")
    plt.plot(xx, xx, '--', label="Línea 1:1")
    plt.xlabel("x: bajo costo (µg/m³)")
    plt.ylabel("y: referencia (µg/m³)")
    plt.title(f"{station} — Prueba (mitad 2)")
    txt = (f"N_test={N_test}\nR²_test={R2_test:.4f}\nRMSE_test={RMSE_test:.3f}\nCobertura={coverage:.2f}%")
    plt.gcf().text(0.62, 0.20, txt, fontsize=9, bbox=dict(boxstyle="round", alpha=0.3))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{station}_TEST_1_scatter.png", dpi=200)
    all_pdf.savefig()
    plt.close()

    # 2) Residuales vs y con bandas de tolerancia
    y_grid = np.linspace(np.nanmin(y), np.nanmax(y), 200)
    tol_grid = np.maximum(5.0, 0.20*y_grid)
    plt.figure(figsize=(6.8,4.8))
    plt.plot(y, resid, '.', alpha=0.6, label="Residuales (test)")
    plt.plot(y_grid,  tol_grid, '-', label="+tolerancia")
    plt.plot(y_grid, -tol_grid, '-', label="-tolerancia")
    plt.xlabel("y: referencia (µg/m³)")
    plt.ylabel("Residuo (µg/m³)")
    plt.title(f"{station} — Residuales en prueba con bandas de tolerancia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{station}_TEST_2_residuals_tol.png", dpi=200)
    all_pdf.savefig()
    plt.close()

    # 3) Histograma de residuales (test)
    mu = float(np.nanmean(resid))
    sd = float(np.nanstd(resid, ddof=1))
    plt.figure(figsize=(6.8,4.6))
    plt.hist(resid, bins=40, alpha=0.9)
    plt.xlabel("Residuo (µg/m³)")
    plt.ylabel("Frecuencia")
    plt.title(f"{station} — Histograma de residuales (test) (μ={mu:.2f}, σ={sd:.2f})")
    plt.tight_layout()
    plt.savefig(f"{station}_TEST_3_hist.png", dpi=200)
    all_pdf.savefig()
    plt.close()
