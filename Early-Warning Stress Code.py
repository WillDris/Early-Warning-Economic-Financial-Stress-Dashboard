# File Name: Early-Warning Stress Code.py
# Author: William Driscoll

# ----------------------------
# Library Imports
# ----------------------------

import io
from io import StringIO
from pathlib import Path
import os
import pandas as pd
import requests
import zipfile
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Config
# ----------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# StatsCan "productId" / PID (table number with dashes removed)
PID_CPI = "18100004"       # 18-10-0004-01 CPI
PID_LFS = "14100287"       # 14-10-0287-01 Labour Force Survey
PID_GDP_IND = "36100434"   # 36-10-0434-01 GDP by industry (monthly)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
WDS_ENDPOINTS = ["https://www150.statcan.gc.ca/t1/wds/en/grp/wds/getFullTableDownloadCSV",]
DIRECT_ZIP_TPL = "https://www150.statcan.gc.ca/n1/tbl/csv/{pid}-{lang}.zip"


# ----------------------------
# Helpers
# ----------------------------
def _contains(s: pd.Series, pattern: str) -> pd.Series:
    return s.astype(str).str.contains(pattern, case=False, na=False)

def _pick_label(df: pd.DataFrame, col: str, preferred: list[str], pattern: str | None = None) -> str | None:
    if col not in df.columns:
        return None
    vals = df[col].dropna().astype(str)
    for p in preferred:
        if (vals == p).any():
            return p
    if pattern:
        matches = vals[vals.str.contains(pattern, case=False, na=False)]
        if not matches.empty:
            return matches.value_counts().idxmax()
    return None

def _parse_statcan_download_response(resp: requests.Response, headers: dict, timeout: int) -> pd.DataFrame:
    content_type = resp.headers.get("Content-Type", "").lower()
    content = resp.content

    looks_like_json = "application/json" in content_type or content.lstrip().startswith(b"{")
    if looks_like_json:
        payload = resp.json()
        if payload.get("status") != "SUCCESS":
            raise RuntimeError(f"WDS error: {payload}")
        download_url = payload.get("object") or payload.get("file")
        if not download_url:
            raise RuntimeError(f"WDS response missing download URL: {payload}")
        dl = requests.get(download_url, headers=headers, timeout=timeout)
        dl.raise_for_status()
        return _parse_statcan_download_response(dl, headers, timeout)

    looks_like_zip = "application/zip" in content_type or content.startswith(b"PK")
    if looks_like_zip:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_name:
                raise RuntimeError("Downloaded ZIP did not contain a CSV file.")
            return pd.read_csv(zf.open(csv_name[0]), low_memory=False)

    return pd.read_csv(io.BytesIO(content), low_memory=False)

def statcan_full_table_csv(pid: str, timeout: int = 120) -> pd.DataFrame:
    headers = {"User-Agent": UA}
    last_err = None

    for method in ("post", "get"):
        for url in WDS_ENDPOINTS:
            try:
                if method == "post":
                    r = requests.post(url, json={"productId": pid}, headers=headers, timeout=timeout)
                else:
                    r = requests.get(url, params={"productId": pid}, headers=headers, timeout=timeout)
                if r.status_code == 404:
                    last_err = (url, f"{r.status_code} {r.reason}")
                    continue
                r.raise_for_status()
                return _parse_statcan_download_response(r, headers, timeout)
            except Exception as e:
                last_err = (url, e)

    # Fallback: direct table CSV ZIP URL (often works even when WDS is blocked)
    for lang in ("eng", "fra"):
        url = DIRECT_ZIP_TPL.format(pid=pid, lang=lang)
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 404:
                last_err = (url, f"{r.status_code} {r.reason}")
                continue
            r.raise_for_status()
            return _parse_statcan_download_response(r, headers, timeout)
        except Exception as e:
            last_err = (url, e)

    raise RuntimeError(f"Failed to download PID {pid}. Last error from {last_err[0]}: {last_err[1]}")

def to_month_start(s: pd.Series) -> pd.Series:
    # StatsCan REF_DATE is like "1914-01" -> convert to datetime (month start)
    return pd.to_datetime(s, format="%Y-%m")

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def boc_group_csv(group: str, start_date: str = "1990-01-01", timeout: int = 120) -> pd.DataFrame:
    """
    Download a BoC Valet group as CSV.
    Returns a DataFrame where each series is a column and the date column is 'date'.
    """
    headers = {"User-Agent": UA}
    url = f"https://www.bankofcanada.ca/valet/observations/group/{group}/csv"
    r = requests.get(url, params={"start_date": start_date}, headers=headers, timeout=timeout)
    r.raise_for_status()

    text = r.text
    lines = text.splitlines()
    obs_idx = None
    for i, line in enumerate(lines):
        if line.strip().strip('"').upper() == "OBSERVATIONS":
            obs_idx = i
            break
    if obs_idx is not None and obs_idx + 1 < len(lines):
        csv_text = "\n".join(lines[obs_idx + 1 :])
        df = pd.read_csv(StringIO(csv_text))
    else:
        df = pd.read_csv(StringIO(text))
    # Valet CSV uses 'date' as the time column
    if "date" not in df.columns:
        raise ValueError(f"Unexpected BoC CSV format: missing 'date' column. Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    return df

# ----------------------------
# CPI: Canada all-items -> YoY inflation %
# ----------------------------
def build_cpi_yoy_canada(df_cpi: pd.DataFrame) -> pd.DataFrame:
    df = df_cpi.copy()
    df["date"] = to_month_start(df["REF_DATE"])
    df["VALUE"] = safe_numeric(df["VALUE"])

    # Filter to Canada + All-items CPI index series
    # We prefer the *non-terminated* "All-items" level series with a base like "2017=100"
    # Then compute YoY % change ourselves (more reliable than chasing a %change UOM).
    f = (
        (df["GEO"] == "Canada")
        & (df["Products and product groups"] == "All-items")
        & (df["UOM"].astype(str).str.contains(r"=100", regex=True))
        & (df["TERMINATED"].isna())  # keep current (not terminated) series
    )
    sub = df.loc[f, ["date", "UOM", "VALUE"]].dropna()

    if sub.empty:
        raise ValueError("Could not find a non-terminated Canada 'All-items' CPI index series in the table.")

    # If multiple bases exist (rare but possible), pick the one with the most recent data coverage.
    # We'll choose the UOM that has the maximum latest date.
    latest_by_uom = sub.groupby("UOM")["date"].max().sort_values(ascending=False)
    best_uom = latest_by_uom.index[0]
    sub = sub[sub["UOM"] == best_uom].sort_values("date")

    # YoY inflation: 100 * (level / level_lag12 - 1)
    sub["cpi_index"] = sub["VALUE"]
    sub["cpi_inflation_yoy_pct"] = 100.0 * (sub["cpi_index"] / sub["cpi_index"].shift(12) - 1.0)

    out = sub[["date", "cpi_inflation_yoy_pct"]].dropna().reset_index(drop=True)
    return out

# ----------------------------
# Unemployment: Canada, SA, 15+ (Both sexes)
# ----------------------------
def build_unemployment_rate_canada_sa(df_lfs: pd.DataFrame) -> pd.DataFrame:
    df = df_lfs.copy()
    df["date"] = to_month_start(df["REF_DATE"])
    df["VALUE"] = safe_numeric(df["VALUE"])

    sex_col = "Sex" if "Sex" in df.columns else "Gender" if "Gender" in df.columns else None
    if not sex_col:
        raise ValueError(f"LFS table is missing expected 'Sex'/'Gender' column. Columns are: {list(df.columns)}")

    # Some LFS tables put seasonal adjustment in a dedicated column; others encode it in "Statistics" or "Data type".
    season_col = None
    for cand in ["Seasonal adjustment", "Statistics", "Data type"]:
        if cand in df.columns and df[cand].astype(str).str.contains("Seasonally adjusted", na=False).any():
            season_col = cand
            break
    if not season_col:
        raise ValueError(
            "Could not find a column indicating seasonal adjustment (expected 'Seasonal adjustment', "
            "'Statistics', or 'Data type' to include 'Seasonally adjusted')."
        )

    base = df["GEO"].eq("Canada")
    lfc = _contains(df["Labour force characteristics"], "Unemployment rate")
    season = _contains(df[season_col], "Seasonally adjusted")

    candidates = df.loc[base & lfc & season].copy()
    labels = {
        sex_col: _pick_label(candidates, sex_col, ["Both sexes"], r"\bBoth sexes\b|Total"),
        "Age group": _pick_label(candidates, "Age group", ["15 years and over"], r"15 years"),
        "Statistics": _pick_label(candidates, "Statistics", ["Estimate", "Value"], r"\bEstimate\b|Value"),
        "Data type": _pick_label(candidates, "Data type", ["Unemployment rate"], r"Unemployment rate"),
        "UOM": _pick_label(candidates, "UOM", [], r"Percent|%"),
    }

    def apply_labels(mask: pd.Series, label_map: dict) -> pd.Series:
        for col, label in label_map.items():
            if label and col in df.columns:
                mask = mask & (df[col] == label)
        return mask

    f = apply_labels(base & lfc & season, labels)

    sub = df.loc[f, ["date", "VALUE"]].dropna().sort_values("date")
    if sub.empty:
        # Relax age filter slightly (some tables use "15 years and over, total")
        age_relaxed = _contains(df["Age group"], "15 years")
        f = apply_labels(base & lfc & season & age_relaxed, labels)
        sub = df.loc[f, ["date", "VALUE"]].dropna().sort_values("date")

    if sub.empty:
        # Last resort: if the table does not include seasonal adjustment dimension, drop it.
        f = apply_labels(base & lfc & age_relaxed, labels)
        sub = df.loc[f, ["date", "VALUE"]].dropna().sort_values("date")

    if sub.empty:
        raise ValueError("Could not find Canada unemployment rate (Both sexes, 15+). Check dimension labels in your table.")

    # Guard against any remaining duplicate dates by keeping the highest value per month.
    if sub["date"].duplicated().any():
        sub = sub.groupby("date", as_index=False)["VALUE"].max()

    out = sub.rename(columns={"VALUE": "unemployment_rate_sa_pct"}).reset_index(drop=True)
    return out

# ----------------------------
# Monthly real GDP (by industry): Canada, SA, all industries -> MoM growth %
# ----------------------------
def build_monthly_real_gdp_growth_canada_sa(df_gdp: pd.DataFrame) -> pd.DataFrame:
    df = df_gdp.copy()
    df["date"] = to_month_start(df["REF_DATE"])
    df["VALUE"] = safe_numeric(df["VALUE"])

    # GDP by industry tables usually include:
    # 'North American Industry Classification System (NAICS)', 'Seasonal adjustment', 'Prices'
    # Some vintages omit 'Estimates', so treat it as optional.
    expected = ["North American Industry Classification System (NAICS)", "Seasonal adjustment", "Prices"]
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"GDP table missing expected column '{col}'. Columns are: {list(df.columns)}")

    # Common “all industries” code text often looks like "All industries [T001]"
    f = (
        (df["GEO"] == "Canada")
        & _contains(df["North American Industry Classification System (NAICS)"], "All industries")
        & _contains(df["Seasonal adjustment"], "Seasonally adjusted")
        & _contains(df["Prices"], "Chained")  # real (volume) measure
    )
    if "Estimates" in df.columns:
        f = f & _contains(df["Estimates"], "Gross domestic product at basic prices")

    sub = df.loc[f, ["date", "VALUE"]].dropna().sort_values("date")
    if sub.empty:
        # If this happens, print unique labels to help you adjust filters quickly
        raise ValueError(
            "Could not match the common GDP-by-industry dimensions for Canada SA real GDP.\n"
            "Inspect unique values in columns: 'Seasonal adjustment', 'Estimates', 'Prices', 'NAICS'."
        )

    sub["real_gdp_level"] = sub["VALUE"]
    sub["real_gdp_growth_mom_pct"] = 100.0 * (sub["real_gdp_level"] / sub["real_gdp_level"].shift(1) - 1.0)

    out = sub[["date", "real_gdp_growth_mom_pct"]].dropna().reset_index(drop=True)
    return out

# ----------------------------
# OECD: Consumer Confidence (CCI) for Canada (monthly, standardised, amplitude adjusted)
# ----------------------------
def oecd_cli_cci_canada(timeout: int = 120, start_period: str = "1990-01") -> pd.DataFrame:
    """
    Pull OECD Consumer Confidence Indicator (CCI) for Canada from the OECD Data Explorer SDMX API.
    Returns: DataFrame with columns [date, consumer_confidence_index]
    """
    headers = {"User-Agent": UA}

    # OECD SDMX-JSON endpoint for CLI dataset (MEI_CLI). We'll filter to Canada CCI locally.
    url = f"https://stats.oecd.org/sdmx-json/data/MEI_CLI/?contentType=csv&startTime={start_period}"

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))

    # Filter to the single Canada CCI series. In MEI_CLI, CCI is LOCOSP with IX units.
    filters = {
        "REF_AREA": "CAN",
        "FREQ": "M",
        "MEASURE": "LOCOSP",
        "UNIT_MEASURE": "IX",
        "ACTIVITY": "_Z",
        "ADJUSTMENT": "NOR",
        "TRANSFORMATION": "IX",
        "TIME_HORIZ": "_Z",
        "METHODOLOGY": "H",
    }
    for col, val in filters.items():
        if col in df.columns:
            df = df[df[col] == val]

    # OECD CSV typically includes columns like:
    # TIME_PERIOD/TIME and OBS_VALUE/Value. We'll handle common variants.
    time_col = "TIME_PERIOD" if "TIME_PERIOD" in df.columns else "TIME" if "TIME" in df.columns else "Time period"
    val_col = "OBS_VALUE" if "OBS_VALUE" in df.columns else "Value" if "Value" in df.columns else "Observation value"

    out = df[[time_col, val_col]].copy()
    time_str = out[time_col].astype(str).str.strip()
    monthly_mask = time_str.str.match(r"^\d{4}-\d{2}$")
    out = out[monthly_mask].copy()
    out["date"] = pd.to_datetime(out[time_col], format="%Y-%m")
    out["consumer_confidence_index"] = pd.to_numeric(out[val_col], errors="coerce")
    out = out[["date", "consumer_confidence_index"]].dropna().sort_values("date").reset_index(drop=True)

    return out

# ----------------------------
# BoC: Policy rate (target for overnight) - monthly EOM
# ----------------------------
def build_policy_rate_monthly_canada(start_date: str = "1990-01-01") -> pd.DataFrame:
    """
    Uses BoC Valet group ATABLE_POLICY_INSTRUMENT.
    Prefer the end-of-month policy target series 'STATIC_ATABLE_V39079' if present,
    else fall back to daily 'V39079' and take end-of-month.
    """
    df = boc_group_csv("ATABLE_POLICY_INSTRUMENT", start_date=start_date)

    # Prefer end-of-month series (often present in this group)
    candidates = [c for c in df.columns if c in ("STATIC_ATABLE_V39079", "V39079")]
    if not candidates:
        raise ValueError(f"Could not find V39079 series in ATABLE_POLICY_INSTRUMENT. Columns: {list(df.columns)}")

    series_col = "STATIC_ATABLE_V39079" if "STATIC_ATABLE_V39079" in candidates else "V39079"

    tmp = df[["date", series_col]].copy()
    tmp[series_col] = pd.to_numeric(tmp[series_col], errors="coerce")
    tmp = tmp.dropna()

    # If daily, convert to end-of-month
    tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
    policy = tmp.sort_values("date").groupby("month", as_index=False)[series_col].last()
    policy = policy.rename(columns={"month": "date", series_col: "policy_rate_pct"})

    return policy.sort_values("date").reset_index(drop=True)

# ----------------------------
# BoC: Yield curve slope (10Y - 2Y) benchmark yields - monthly
# ----------------------------
def build_yield_curve_slope_10y_2y_monthly_canada(start_date: str = "1990-01-01") -> pd.DataFrame:
    """
    Uses BoC Valet group bond_yields_benchmark.
    Pull 2Y and 10Y benchmark yields and compute slope = 10Y - 2Y.
    """
    df = boc_group_csv("bond_yields_benchmark", start_date=start_date)

    # Series IDs from BoC benchmark bond yields group
    # 2Y: BD.CDN.2YR.DQ.YLD
    # 10Y: BD.CDN.10YR.DQ.YLD
    y2 = "BD.CDN.2YR.DQ.YLD"
    y10 = "BD.CDN.10YR.DQ.YLD"

    if y2 not in df.columns or y10 not in df.columns:
        raise ValueError(
            "Missing benchmark yield columns in bond_yields_benchmark group.\n"
            f"Need: {y2} and {y10}\n"
            f"Have: {list(df.columns)}"
        )

    tmp = df[["date", y2, y10]].copy()
    tmp[y2] = pd.to_numeric(tmp[y2], errors="coerce")
    tmp[y10] = pd.to_numeric(tmp[y10], errors="coerce")
    tmp = tmp.dropna()

    # Convert to monthly (end-of-month), then compute slope
    tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
    m = tmp.sort_values("date").groupby("month", as_index=False)[[y2, y10]].last()
    m = m.rename(columns={"month": "date", y2: "yld_2y_pct", y10: "yld_10y_pct"})
    m["yield_curve_slope_10y_2y_pct"] = m["yld_10y_pct"] - m["yld_2y_pct"]

    return m[["date", "yield_curve_slope_10y_2y_pct"]].sort_values("date").reset_index(drop=True)

# ----------------------------
# VIX (FRED) -> monthly volatility proxy
# ----------------------------
def build_vix_monthly_from_fred(how: str = "avg") -> pd.DataFrame:
    """
    Pull VIXCLS from FRED (daily close) without an API key.
    how:
      - 'avg' monthly average (recommended)
      - 'eom' end-of-month
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "vix"]
    df["date"] = pd.to_datetime(df["date"])
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.dropna().sort_values("date")

    s = df.set_index("date")["vix"]
    if how == "eom":
        m = s.resample("M").last()
    else:
        m = s.resample("M").mean()

    out = m.reset_index().rename(columns={"vix": "vix_monthly"})
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    print("Working directory:", os.getcwd())
    print("Saving outputs to:", OUTPUT_DIR)

    # --- CPI, Unemployment, GDP Growth (Statistics Canada) ---
    specs = [
        ("cpi", PID_CPI, "CPI table", "CPI YoY inflation (Canada, All-items)", build_cpi_yoy_canada, "cpi_inflation_yoy_canada.csv"),
        ("lfs", PID_LFS, "LFS table", "Unemployment rate (Canada, SA)", build_unemployment_rate_canada_sa, "unemployment_rate_sa_canada.csv"),
        ("gdp", PID_GDP_IND, "GDP-by-industry table", "Monthly real GDP growth MoM (Canada, SA, all industries)", build_monthly_real_gdp_growth_canada_sa, "real_gdp_growth_mom_canada.csv"),
    ]

    tables = {}
    for key, pid, label, _, _, _ in specs:
        print(f"Downloading {label}...")
        tables[key] = statcan_full_table_csv(pid)

    outputs = {}
    for key, _, _, build_label, builder, outfile in specs:
        print(f"Building {build_label}...")
        outputs[key] = builder(tables[key])
        out_path = OUTPUT_DIR / outfile
        outputs[key].to_csv(out_path, index=False)
        print("Saved:", out_path)

    # --- OECD (Consumer Confidence) ---
    print("Downloading OECD Consumer Confidence (Canada)...")
    df_cci = oecd_cli_cci_canada(start_period="1990-01")
    cci_path = OUTPUT_DIR / "consumer_confidence_oecd_canada.csv"
    df_cci.to_csv(cci_path, index=False)
    print("Saved:", cci_path)

    # --- Bank of Canada (Policy rate) ---
    print("Downloading BoC policy rate (Canada)...")
    df_policy = build_policy_rate_monthly_canada(start_date="1990-01-01")
    policy_path = OUTPUT_DIR / "policy_rate_canada.csv"
    df_policy.to_csv(policy_path, index=False)
    print("Saved:", policy_path)

    # --- Bank of Canada (Yield curve slope) ---
    print("Downloading BoC benchmark yields and building 10Y-2Y slope (Canada)...")
    df_slope = build_yield_curve_slope_10y_2y_monthly_canada(start_date="1990-01-01")
    slope_path = OUTPUT_DIR / "yield_curve_slope_10y_2y_canada.csv"
    df_slope.to_csv(slope_path, index=False)
    print("Saved:", slope_path)

    # --- FRED Equity Market Volatility (VIX)
    print("Downloading FRED VIX (monthly) ...")
    df_vix = build_vix_monthly_from_fred(how="avg")
    slope_path = OUTPUT_DIR / "yield_curve_slope_10y_2y_canada.csv"
    df_slope.to_csv(slope_path, index=False)
    print("Saved:", slope_path)

    # Merge into one dataset
    merged = (
        outputs["cpi"]
        .merge(outputs["lfs"], on="date", how="inner")
        .merge(outputs["gdp"], on="date", how="inner")
        .merge(df_cci, on="date", how="inner")
        .merge(df_policy, on="date", how="inner")
        .merge(df_slope, on="date", how="inner")
        .merge(df_vix, on="date", how="inner")
        .sort_values("date")
    )
    merged_path = OUTPUT_DIR / "merged_inputs_canada_monthly.csv"
    merged.to_csv(merged_path, index=False)
    print("Saved merged inputs:", merged_path)
    print("\nDone data acquisition.")

    df_stress = merged.copy()

    df_stress["cpi_stress"] = df_stress["cpi_inflation_yoy_pct"]
    df_stress["unemp_stress"] = df_stress["unemployment_rate_sa_pct"]
    df_stress["gdp_stress"] = -df_stress["real_gdp_growth_mom_pct"]
    df_stress["cci_stress"] = -df_stress["consumer_confidence_index"]
    
    df_stress["policy_stress"] = df_stress["policy_rate_pct"]
    df_stress["slope_stress"] = -df_stress["yield_curve_slope_10y_2y_pct"]
    df_stress["vix_stress"] = df_stress["vix_monthly"]

    # Standardize using z-scores so everything is on the same scale
    econ_cols = ["cpi_stress", "unemp_stress", "gdp_stress", "cci_stress"]
    z = (df_stress[econ_cols] - df_stress[econ_cols].mean()) / df_stress[econ_cols].std(ddof=0)
    z = z.add_prefix("z_")
    df_econ_z = pd.concat([df_stress[["date"]], z], axis=1).dropna()

    fin_cols = ["policy_stress", "slope_stress", "vix_stress"]
    z_fin = (
    df_stress[fin_cols]
    .subtract(df_stress[fin_cols].mean())
    .divide(df_stress[fin_cols].std(ddof=0))
    .add_prefix("z_"))
    df_fin_z = pd.concat([df_stress[["date"]], z_fin], axis=1).dropna()

    # Build macroeconomic stress sub-index 
    z_econ_cols = ["z_cpi_stress", "z_unemp_stress", "z_gdp_stress", "z_cci_stress"]
    df_econ_z["macro_stress_index"] = df_econ_z[z_econ_cols].mean(axis=1)

    # Build financial stress sub-index
    z_fin_cols = ["z_policy_stress", "z_slope_stress", "z_vix_stress"]
    df_fin_z["financial_stress_index"] = df_fin_z[z_fin_cols].mean(axis=1)

    # Combine to macro-weighted overall stress index
    df_all = df_econ_z.merge(df_fin_z, on="date", how="inner")
    df_all["overall_stress_index"] = (
    0.6 * df_all["macro_stress_index"] +
    0.4 * df_all["financial_stress_index"])
    
    recessions = [
    ("2001-03-01", "2002-01-01"),
    ("2008-10-01", "2009-06-01"),
    ("2020-03-01", "2020-04-01"),]

    # Create macro stress index plot
    plt.figure()
    plt.plot(df_econ_z["date"], df_econ_z["macro_stress_index"])
    for start, end in recessions:
        plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="grey",alpha=0.3)
    plt.title("Canada Macro Stress Index (standardized)")
    plt.xlabel("Date")
    plt.ylabel("Index (z-score)")
    plt.show()
    
    # Create financial stress index plot
    plt.figure()
    plt.plot(df_fin_z["date"], df_fin_z["financial_stress_index"])
    for start, end in recessions:
        plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="grey",alpha=0.3)
    plt.title("Canada Financial Stress Index (standardized)")
    plt.xlabel("Date")
    plt.ylabel("Index (z-score)")
    plt.show()

    # Create overall stress index plot
    plt.figure()
    plt.plot(df_all["date"], df_all["overall_stress_index"])
    for start, end in recessions:
        plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="grey",alpha=0.3)
    plt.title("Canada Overall Stress Index (standardized)")
    plt.xlabel("Date")
    plt.ylabel("Index (z-score)")
    plt.show()

    # Create stress index component contributions
    macro_components = ["z_cpi_stress","z_unemp_stress","z_gdp_stress","z_cci_stress"]
    plt.figure(figsize=(10,5))
    plt.stackplot(
        df_all["date"],
        [df_all[c] for c in macro_components],
        labels=["Inflation", "Unemployment", "GDP growth", "Confidence"],
        alpha=0.8)
    plt.title("Macro Stress Index — Component Contributions")
    plt.ylabel("Contribution (z-score)")
    plt.legend(loc="upper left")
    plt.show()

    #Perform lead-lag check
    lead_lag = df_stress.merge(df_all[["date", "overall_stress_index"]], on="date", how="inner")
    stress = lead_lag["overall_stress_index"]
    unemp = lead_lag["unemployment_rate_sa_pct"]
    max_lag = 24  # months
    lags = range(-max_lag, max_lag + 1)
    cors = []
    for lag in lags:
        cors.append(stress.corr(unemp.shift(-lag)))
    lead_lag_df = pd.DataFrame({
        "lag_months": lags,
        "correlation": cors})
    # Plot lead-lag check
    plt.figure(figsize=(8,4))
    plt.plot(lead_lag_df["lag_months"], lead_lag_df["correlation"])
    plt.axvline(0, color="black", linestyle="--")
    plt.title("Lead–Lag: Stress Index vs Unemployment")
    plt.xlabel("Stress leads ←  Lag (months)  → Unemployment leads")
    plt.ylabel("Correlation")
    plt.show()


    # Save outputs + components for PowerBI
    long = df_all.melt(
    id_vars="date",
    value_vars=["macro_stress_index", "financial_stress_index", "overall_stress_index",],
    var_name="series",
    value_name="value")
    long.to_csv("canada_stress_index_components.csv", index=False)


    print("\nDone.")

if __name__ == "__main__":
    main()
