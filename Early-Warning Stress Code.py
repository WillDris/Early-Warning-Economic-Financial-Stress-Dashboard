import io
import zipfile
import requests
import pandas as pd

BASE = "https://www150.statcan.gc.ca/t1/wds/rest"

def wds_get_json(url: str):
    r = requests.get(url, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()

def download_zip_to_dfs(zip_url: str):
    r = requests.get(zip_url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # StatsCan zips usually contain one CSV data file (sometimes plus a README)
    csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    dfs = {}
    for name in csv_names:
        with z.open(name) as f:
            dfs[name] = pd.read_csv(f)
    return dfs

# --- CPI: Table 18-10-0004-01 => productId 18100004 (WDS full table)
product_id = "18100004"
lang = "en"

meta = wds_get_json(f"{BASE}/getFullTableDownloadCSV/{product_id}/{lang}")
zip_url = meta["object"]   # per WDS guide this is a direct URL to the zip
print("ZIP URL:", zip_url)

dfs = download_zip_to_dfs(zip_url)

# Usually there is a single main data CSV; grab the largest one
main_name = max(dfs, key=lambda k: dfs[k].shape[0])
cpi = dfs[main_name]

print(main_name)
print(cpi.head())
print(cpi.columns)
