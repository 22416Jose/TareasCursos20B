import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# 1) Cargar datos y preparar
# ----------------------------

paths = [
    "/mnt/data/PM25_Acualago_Referencia.csv",
    "/mnt/data/PM25_Caldas_Referencia.csv",
    "/mnt/data/PM25_ColegioNormal_2018-11_a_2019-08_Calibrar.csv",
    "/mnt/data/PM25_Giron_Referencia.csv",
    "/mnt/data/PM25_Normal_Referencia.csv",
    "/mnt/data/PM25_Pilar_Referencia.csv",
]

def load_pm25(path: str) -> pd.DataFrame:
    """Lee CSV con columnas ['Fecha','Hora','PM2.5'] y devuelve DataFrame
    con columnas ['datetime','PM25'] ordenado cronológicamente."""
    df = pd.read_csv(path)
    dt = pd.to_datetime(
        df['Fecha'].astype(str) + ' ' + df['Hora'].astype(str),
        errors='coerce', dayfirst=True
    )
    df = df.assign(datetime=dt).dropna(subset=['datetime'])
    df = df[['datetime', 'PM2.5']].rename(columns={'PM2.5': 'PM25'})
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

datasets = {Path(p).name: load_pm25(p) for p in paths}

def hourly_mean(df: pd.DataFrame) -> pd.Series:
    """Promedio por hora de la columna PM25 con índice datetime."""
    return df.set_index('datetime')['PM25'].resample('H').mean()

lowcost_name = "PM25_ColegioNormal_2018-11_a_2019-08_Calibrar.csv"
lowcost_hourly = hourly_mean(datasets[lowcost_name])

ref_names = [n for n in datasets if n != lowcost_name]
refs_hourly = {name: hourly_mean(datasets[name]) for name in ref_names}

# ----------------------------
# 2) Promedios móviles y D/RMSE
# ----------------------------

windows = [3, 6, 12, 24, 48, 72]  # horas

def rolling_mean(series: pd.Series, w: int) -> pd.Series:
    """Promedio móvil centrado con ventana temporal de w horas.
    Exige al menos 50% de datos dentro de la ventana."""
    minp = max(1, int(w * 0.5))
    return series.rolling(f'{w}H', center=True, min_periods=minp).mean()

rows = []
for ref_name, ref_ser in refs_hourly.items():
    for w in windows:
        ref_roll = rolling_mean(ref_ser, w).rename('ref')
        lc_roll  = rolling_mean(lowcost_hourly, w).rename('lc')
        aligned = pd.concat([ref_roll, lc_roll], axis=1).dropna()
        if len(aligned) == 0:
            D = np.nan
            N = 0
            rmse = np.nan
        else:
            diff = aligned['ref'] - aligned['lc']
            D = float(np.sqrt(np.sum(np.square(diff.values))))
            N = int(len(aligned))
            rmse = float(D / np.sqrt(N))
        rows.append({
            'Referencia': ref_name.replace('PM25_','').replace('_Referencia.csv',''),
            'Ventana_horas': int(w),
            'N_puntos': N,
            'Distancia_Euclidea': D,
            'RMSE': rmse
        })

results_df = pd.DataFrame(rows).sort_values(['Referencia','Ventana_horas']).reset_index(drop=True)

# ----------------------------
# 3) Figuras
# ----------------------------

plt.figure(figsize=(7,5))
for ref_name in results_df['Referencia'].unique():
    subset = results_df[results_df['Referencia'] == ref_name]
    plt.plot(subset['Ventana_horas'], subset['RMSE'], marker='o', label=ref_name)
plt.xlabel('Ancho de ventana (horas)')
plt.ylabel('RMSE (µg/m³)')
plt.title('RMSE entre promedios móviles (ref vs. bajo costo)')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/data/rmse_vs_ventana.png', dpi=200)
plt.close()

plt.figure(figsize=(7,5))
for ref_name in results_df['Referencia'].unique():
    subset = results_df[results_df['Referencia'] == ref_name]
    plt.plot(subset['Ventana_horas'], subset['Distancia_Euclidea'], marker='o', label=ref_name)
plt.xlabel('Ancho de ventana (horas)')
plt.ylabel('Distancia euclídea')
plt.title('Distancia D vs. ancho de la ventana móvil')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/data/distancia_vs_ventana.png', dpi=200)
plt.close()
