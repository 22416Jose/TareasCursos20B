import pandas as pd
import glob
from pathlib import Path

# Patrón de archivos fuente (ajústalo si es necesario)
pattern = "/mnt/data/mediciones_clg_normalsup_pm25_a_*.csv"
files = sorted(glob.glob(pattern))

def load_and_fix(path: str) -> pd.DataFrame:
    """
    Normaliza un CSV (formato 'fecha_hora_med,"id_parametro","valor"')
    al formato: Fecha, Hora, PM2.5 (zona horaria America/Bogota).
    Maneja comillas y encabezados duplicados.
    """
    # Leer como CSV; si viene todo en una sola columna, separte manualmente
    raw = pd.read_csv(path, header=None, sep=",", engine="python", dtype=str)
    if raw.shape[1] != 3:
        parts = raw.iloc[:, 0].astype(str).str.split(",", n=2, expand=True)
    else:
        parts = raw.iloc[:, :3].astype(str)

    parts.columns = ["FechaHora_raw", "Parametro", "PM2.5"]

    # Quitar fila de encabezado si aparece dentro de los datos
    is_header = parts["FechaHora_raw"].str.contains("fecha_hora_med", case=False, na=False)
    parts = parts[~is_header].copy()

    # Limpiar comillas/espacios
    for c in parts.columns:
        parts[c] = parts[c].str.replace('"', '', regex=False).str.strip()

    # Parsear datetime (vienen en UTC con 'Z') y convertir a America/Bogota
    dt = pd.to_datetime(parts["FechaHora_raw"], errors="coerce", utc=True)
    dt = dt.dt.tz_convert("America/Bogota")

    # Construir DataFrame final
    df = pd.DataFrame({
        "Fecha": dt.dt.strftime("%Y-%m-%d"),
        "Hora": dt.dt.strftime("%H:%M:%S"),
        "PM2.5": pd.to_numeric(parts["PM2.5"], errors="coerce")
    })

    # Ordenar y limpiar
    df = df.dropna(subset=["PM2.5"]).sort_values(["Fecha", "Hora"]).reset_index(drop=True)
    return df

# Procesar todos los archivos
fixed_by_month = {}
for f in files:
    fixed_by_month[f] = load_and_fix(f)

# Consolidado
all_fixed = (
    pd.concat(fixed_by_month.values(), ignore_index=True)
      .sort_values(["Fecha", "Hora"])
      .reset_index(drop=True)
)

# Guardar por mes y consolidado
out_dir = "/mnt/data"
saved_paths = []

for f, df in fixed_by_month.items():
    if len(df) == 0:
        continue
    ym = pd.to_datetime(df["Fecha"].iloc[0]).strftime("%Y-%m")
    out_path = f"{out_dir}/PM25_ColegioNormal_{ym}.csv"
    df.to_csv(out_path, index=False)
    saved_paths.append(out_path)

out_all = f"{out_dir}/PM25_ColegioNormal_2018-11_a_2019-08.csv"
all_fixed.to_csv(out_all, index=False)
saved_paths.append(out_all)

print("Archivos generados:")
for p in saved_paths:
    print(p)
