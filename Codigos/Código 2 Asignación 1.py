import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# --- Entradas ---
paths = [
    "PM25_Acualago_Referencia.csv",
    "PM25_Caldas_Referencia.csv",
    "PM25_ColegioNormal_2018-11_a_2019-08_Calibrar.csv",  # bajo costo (x)
    "PM25_Giron_Referencia.csv",
    "PM25_Normal_Referencia.csv",
    "PM25_Pilar_Referencia.csv",
]
LOWCOST = "PM25_ColegioNormal_2018-11_a_2019-08_Calibrar.csv"
W = 72  # horas, ventana de promedio movil centrado
MINP = max(1, int(W*0.5))  # exigir >=50% de datos en la ventana

# --- Utilidades ---
def load_pm25(path):
    df = pd.read_csv(path)
    dt = pd.to_datetime(df["Fecha"].astype(str)+" "+df["Hora"].astype(str),
                        errors="coerce", dayfirst=True)
    df = df.assign(datetime=dt).dropna(subset=["datetime"])
    return (df[["datetime","PM2.5"]]
            .rename(columns={"PM2.5":"PM25"})
            .sort_values("datetime")
            .reset_index(drop=True))

def hourly(series_df):
    return (series_df.set_index("datetime")["PM25"].resample("H").mean())

def roll(series, w=W, minp=MINP):
    return series.rolling(f"{w}H", center=True, min_periods=minp).mean()

# --- Carga y remuestreo horario ---
data = {Path(p).name: load_pm25(p) for p in paths}
x_hour = hourly(data[LOWCOST])  # bajo costo
refs = [k for k in data if k != LOWCOST]
refs_hour = {k: hourly(data[k]) for k in refs}

# --- Calibracion y resultados ---
rows = []
detailed = {}
for key in sorted(refs):
    y_roll = roll(refs_hour[key])
    x_roll = roll(x_hour)
    df = pd.concat([y_roll.rename("y"), x_roll.rename("x")], axis=1).dropna()
    if df.empty:
        continue

    # Ajuste forzado al origen: alpha = sum(x*y)/sum(x^2)
    Sxy = np.sum(df["x"]*df["y"])
    Sxx = np.sum(df["x"]*df["x"])
    alpha = Sxy / Sxx
    yhat = alpha*df["x"]
    resid = df["y"] - yhat

    SSE = np.sum(resid**2)
    SST0 = np.sum(df["y"]**2)
    R2_0 = 1 - SSE/SST0

    # Error estandar e IC95% para alpha (modelo sin intercepto)
    n = len(df)
    sigma2 = SSE/(n-1)
    var_alpha = sigma2 / Sxx
    se_alpha = np.sqrt(var_alpha)
    ci_low, ci_high = alpha - 1.96*se_alpha, alpha + 1.96*se_alpha

    # Tolerancia y validez
    tol = np.maximum(5.0, 0.20*np.abs(df["y"]))
    valid = np.abs(resid) <= tol
    coverage = 100.0*valid.mean()
    x_valid_min = df.loc[valid, "x"].min()
    x_valid_max = df.loc[valid, "x"].max()

    # Bloque valido continuo mas largo (en horas, indice horario)
    runs, start = [], None
    for t, ok in valid.items():
        if ok and start is None: start = t
        if (not ok or t==valid.index[-1]) and start is not None:
            end = t if ok else valid.index[valid.index.get_loc(t)-1]
            runs.append((start, end)); start = None
    if runs:
        dur = [((b-a)+pd.Timedelta(hours=1)).total_seconds()/3600 for a,b in runs]
        k = int(np.argmax(dur)); longest_h = dur[k]; a,b = runs[k]
    else:
        longest_h, a, b = 0.0, None, None

    rows.append({
        "Referencia": key.replace("PM25_","").replace("_Referencia.csv",""),
        "N_pares": n, "alpha": alpha, "alpha_CI95_low": ci_low,
        "alpha_CI95_high": ci_high, "R2_origen": R2_0,
        "Cobertura_valida_%": coverage, "x_valido_min": x_valid_min,
        "x_valido_max": x_valid_max, "Bloque_valido_mas_largo_h": longest_h,
        "Bloque_valido_inicio": a, "Bloque_valido_fin": b
    })
    detailed[key] = df.assign(yhat=yhat, resid=resid, tol=tol, valid=valid)

calib = pd.DataFrame(rows).sort_values("Referencia")
calib.to_csv("calibracion_resultados_resumen.csv", index=False)

# --- Figuras completas por estacion (3 paginas) ---
with PdfPages("calibracion_figuras_completas_ALL.pdf") as allpdf:
    for _, r in calib.iterrows():
        ref = r["Referencia"]; alpha = r["alpha"]; ciL=r["alpha_CI95_low"]; ciH=r["alpha_CI95_high"]
        R2=r["R2_origen"]; N=r["N_pares"]; cover=r["Cobertura_valida_%"]
        df = detailed[f"PM25_{ref}_Referencia.csv"]; x=df["x"].values; y=df["y"].values
        resid=df["resid"].values; yhat=df["yhat"].values

        # 1) Scatter con 1:1 y recta y=alpha x
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        plt.figure(figsize=(6.5,6.2))
        plt.plot(x,y,'.',alpha=.5,label="Datos (72 h)")
        plt.plot(xx, alpha*xx, '-', label=f"Fit: y={alpha:.4f}x")
        plt.plot(xx, xx, '--', label="Línea 1:1")
        plt.xlabel("x: bajo costo (µg/m³)"); plt.ylabel("y: referencia (µg/m³)")
        plt.title(f"{ref} — Ajuste y=αx (72 h)")
        txt=(f"α={alpha:.4f} (IC95% {ciL:.4f}–{ciH:.4f})\nR²={R2:.4f}  N={N}\nCobertura={cover:.2f}%")
        plt.gcf().text(0.62,0.20,txt,fontsize=9,bbox=dict(boxstyle="round",alpha=.3))
        plt.legend(); plt.tight_layout(); allpdf.savefig(); plt.close()

        # 2) Residuales vs y con bandas ±max(5, 0.2 y)
        yg = np.linspace(np.nanmin(y), np.nanmax(y), 200); tol=np.maximum(5.0, 0.2*yg)
        plt.figure(figsize=(6.8,4.8))
        plt.plot(y, resid, '.', alpha=.5, label="Residuales")
        plt.plot(yg, tol,'-',label="+tolerancia"); plt.plot(yg,-tol,'-',label="-tolerancia")
        plt.xlabel("y (µg/m³)"); plt.ylabel("Residuo (µg/m³)")
        plt.title(f"{ref} — Residuales vs y (tolerancia)")
        plt.legend(); plt.tight_layout(); allpdf.savefig(); plt.close()

        # 3) Histograma de residuales
        mu, sd = float(np.nanmean(resid)), float(np.nanstd(resid, ddof=1))
        plt.figure(figsize=(6.8,4.6)); plt.hist(resid, bins=40, alpha=.9)
        plt.xlabel("Residuo (µg/m³)"); plt.ylabel("Frecuencia")
        plt.title(f"{ref} — Histograma (μ={mu:.2f}, σ={sd:.2f})")
        plt.tight_layout(); allpdf.savefig(); plt.close()
print("OK")