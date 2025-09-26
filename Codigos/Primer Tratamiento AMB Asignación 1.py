import pandas as pd
import numpy as np
import re
from pathlib import Path

# === Configuración ===
xlsx_path = Path("Datos Estaciones AMB.xlsx")   # Cambia la ruta si es necesario
out_dir = Path("salida_pm25")                   # Carpeta de salida de CSVs
out_dir.mkdir(parents=True, exist_ok=True)

# Valores considerados como "no dato"
NA_STRINGS = [
    "nodata", "no data", "no_dato", "no dato", "sin dato",
    "n/a", "na", "nan", "null", "none", "-", "--", "", " "
]

def normalize(col: str) -> str:
    """Normaliza nombres de columnas (minúsculas, sin tildes, colapsa espacios)."""
    s = col.strip().lower()
    s = (s.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u").replace("ü","u"))
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def guess_pm25_column(cols_norm_map):
    """Encuentra la columna PM2.5 por patrones frecuentes."""
    patterns = [
        r"pm\s*2[.,]?\s*5",   # PM 2.5 / PM2.5 / PM 2,5
        r"\bpm25\b",
        r"pm_?2_?5",
    ]
    for norm, original in cols_norm_map.items():
        for pat in patterns:
            if re.search(pat, norm):
                return original
    return None

def guess_datetime_columns(cols_norm_map):
    """Detecta columnas de fecha/hora (combinada o separadas)."""
    combined_keys = [
        "fechahora", "fecha hora", "fecha_hora",
        "datetime", "date time", "date_time",
        "timestamp", "marca de tiempo"
    ]
    date_keys = ["fecha", "date", "day", "dia"]
    time_keys = ["hora", "time"]

    for norm, original in cols_norm_map.items():
        if any(k in norm for k in combined_keys):
            return {"combined": original, "date": None, "time": None}

    date_col, time_col = None, None
    for norm, original in cols_norm_map.items():
        if date_col is None and any(k == norm or norm.startswith(k+" ") for k in date_keys):
            date_col = original
        if time_col is None and any(k == norm or norm.startswith(k+" ") for k in time_keys):
            time_col = original
    return {"combined": None, "date": date_col, "time": time_col}

def coerce_datetime(series, dayfirst=True):
    """Convierte serie a datetime (maneja números de Excel y textos)."""
    if np.issubdtype(series.dtype, np.datetime64):
        return pd.to_datetime(series, errors="coerce")
    if np.issubdtype(series.dtype, np.number):
        # Serial Excel (días desde 1899-12-30) y fracciones como horas
        return pd.to_datetime(series, unit="d", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(series.astype(str), dayfirst=dayfirst, errors="coerce")

def export_pm25_by_sheet(xlsx_path: Path, out_dir: Path):
    xls = pd.ExcelFile(xlsx_path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(
            xlsx_path, sheet_name=sheet, dtype=object,
            keep_default_na=True, na_values=NA_STRINGS
        )
        if df is None or df.empty:
            continue

        # Mapa nombre_normalizado -> nombre_original
        cols_norm_map = {normalize(str(c)): c for c in df.columns}

        # 1) Encontrar PM2.5
        pm_col = guess_pm25_column(cols_norm_map)
        if pm_col is None:
            # Si no hay PM2.5, saltamos la hoja
            continue

        # 2) Encontrar fecha/hora
        dt_info = guess_datetime_columns(cols_norm_map)

        # 3) Construir DateTime
        if dt_info["combined"]:
            dt_series = coerce_datetime(df[dt_info["combined"]])
        else:
            if dt_info["date"] is not None and dt_info["time"] is not None:
                fecha_parsed = coerce_datetime(df[dt_info["date"]]).dt.date
                hora_series = df[dt_info["time"]]
                hora_parsed = pd.to_datetime(hora_series, errors="coerce").dt.time
                if hora_parsed.isna().mean() > 0.5:
                    # Intento alterno: serial Excel como fracción de día
                    try:
                        hora_parsed = pd.to_datetime(hora_series.astype(float), unit="d",
                                                     origin="1899-12-30", errors="coerce").dt.time
                    except Exception:
                        pass
                dt_series = pd.to_datetime(
                    pd.Series(fecha_parsed.astype(str)) + " " + pd.Series(hora_parsed.astype(str)),
                    errors="coerce"
                )
            elif dt_info["date"] is not None:
                dt_series = coerce_datetime(df[dt_info["date"]])
            else:
                # Último intento: probar columnas que contengan 'fecha/hora/time/date'
                possible = [c for c in df.columns
                            if re.search(r"fecha|hora|time|date|datetime|timestamp", str(c), flags=re.I)]
                dt_series = None
                for c in possible:
                    cand = coerce_datetime(df[c])
                    if cand.notna().sum() > len(cand) * 0.2:
                        dt_series = cand
                        break
                if dt_series is None:
                    dt_series = pd.to_datetime(pd.Series([pd.NaT]*len(df)))

        # 4) PM2.5 a numérico y filtrar “buenos”
        pm_series = pd.to_numeric(df[pm_col], errors="coerce")
        out = pd.DataFrame({"DateTime": dt_series, "PM2.5": pm_series})
        out = out.dropna(subset=["PM2.5"])
        if out["DateTime"].notna().any():
            out = out.dropna(subset=["DateTime"])

        # 5) Formatear Fecha y Hora y reordenar
        out["Fecha"] = out["DateTime"].dt.date
        out["Hora"]  = out["DateTime"].dt.strftime("%H:%M:%S")
        out = out[["Fecha", "Hora", "PM2.5"]]

        if out.empty:
            continue

        # 6) Guardar CSV por ventana (hoja)
        safe_sheet = re.sub(r"[^A-Za-z0-9_-]+", "_", sheet).strip("_")
        out.to_csv(out_dir / f"PM25_{safe_sheet}.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    export_pm25_by_sheet(xlsx_path, out_dir)
    print(f"Listo. Archivos en: {out_dir.resolve()}")
