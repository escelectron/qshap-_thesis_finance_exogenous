#!/usr/bin/env python3
"""
Fetch and merge SPX + 5 exogenous features (20 years)
Author: Pranav K. Sanghadia
License: MIT
Description:
  - Stage 1: Download each dataset (if not already on disk)
  - Stage 2: Merge, clean, align, and save processed result
Folder structure:
  /src
  /spx_data
  /spx_results

1) SPX daily closing price (for target)
2) Gold daily close (via Yahoo Finance)
3) DXY (US Dollar Index) daily close (via Yahoo Finance or FRED, whichever is available)
4) Sunspot count (daily) from SILSO / LISIRD
5) Kp-index (planetary geomagnetic) historical daily values (or derived from 3-hr Kp/AP data)
6) Global earthquake daily count (using USGS event API)  
  
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "spx_data")
RESULTS_DIR = os.path.join(BASE_DIR, "spx_results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
MIN_MAGNITUDE = 4.0



def safe_read_yahoo_csv(path, name):
    """Read cached Yahoo CSV and normalize columns."""
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip metadata lines (like 'Price,SPX Close' or 'Ticker,^GSPC')
    data_start = 0
    for i, line in enumerate(lines):
        if line.lower().startswith("date"):
            data_start = i
            break

    df = pd.read_csv(path, skiprows=data_start)
    if "Date" in df.columns:
        df.set_index("Date", inplace=True)
    df.index = pd.to_datetime(df.index)

    # Rename first column to expected name
    if len(df.columns) == 1:
        df.columns = [name]
    elif "Close" in df.columns:
        df.rename(columns={"Close": name}, inplace=True)

    return df


# -------------------------------------------------------
# === Stage 1: Download individual datasets ===
# -------------------------------------------------------

def fetch_yahoo(ticker, name):
    """Download data from Yahoo Finance and save locally"""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if os.path.exists(path):
        print(f"📂 Found cached file: {path}")
        df = safe_read_yahoo_csv(path, name)
        if df is not None and not df.empty:
            return df
        else:
            print(f"⚠️ Cached file unreadable, re-downloading {name}...")

    print(f"⬇️  Downloading {name} ({ticker}) from Yahoo Finance...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1d")[["Close"]]
    df.rename(columns={"Close": name}, inplace=True)
    df.index = pd.to_datetime(df.index).normalize()
    df.to_csv(path)
    print(f"✅ Saved {name} → {path}")
    return df



def fetch_sunspots():
    path = os.path.join(DATA_DIR, "Sunspot.csv")
    if os.path.exists(path):
        print(f"📂 Found cached file: {path}")
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    print("⬇️  Fetching daily sunspot data from SILSO...")
    url = "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    data = []
    for line in resp.text.splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 4:
            y, m, d, val = int(p[0]), int(p[1]), int(p[2]), float(p[3])
            data.append([datetime(y, m, d), val])
    df = pd.DataFrame(data, columns=["Date", "Sunspot"]).set_index("Date")
    df = df.loc[START_DATE:END_DATE]
    df.to_csv(path)
    print(f"✅ Saved Sunspot → {path}")
    return df


def fetch_kp_index():
    path = os.path.join(DATA_DIR, "Kp.csv")
    if os.path.exists(path):
        print(f"📂 Found cached file: {path}")
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    print("⬇️  Fetching Kp-index data from NOAA SWPC...")
    url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    j = pd.DataFrame(r.json())
    j["time_tag"] = pd.to_datetime(j["time_tag"])
    j["Date"] = j["time_tag"].dt.floor("D")
    df = j.groupby("Date")["kp_index"].mean().rename("Kp").to_frame()
    df = df.loc[START_DATE:END_DATE]
    df.to_csv(path)
    print(f"✅ Saved Kp-index → {path}")
    return df


def fetch_earthquake_counts():
    path = os.path.join(DATA_DIR, "Earthquake_Counts.csv")
    if os.path.exists(path):
        print(f"📂 Found cached file: {path}")
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    print("⬇️  Fetching earthquake counts from USGS...")
    all_counts = []
    for year in range(2005, datetime.now().year + 1):
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "geojson",
            "starttime": f"{year}-01-01",
            "endtime": f"{year}-12-31",
            "minmagnitude": MIN_MAGNITUDE,
            "limit": 20000,
        }
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            events = r.json().get("features", [])
            if not events:
                continue
            dates = [datetime.utcfromtimestamp(e["properties"]["time"]/1000).date() for e in events]
            dfy = pd.DataFrame(dates, columns=["Date"])
            c = dfy.groupby("Date").size().rename("Quake_Count").to_frame()
            c.index = pd.to_datetime(c.index)
            all_counts.append(c)
        except Exception as e:
            print(f"⚠️ USGS year {year}: {e}")

    if not all_counts:
        print("⚠️ No earthquake data fetched.")
        return pd.DataFrame()

    df = pd.concat(all_counts).resample("D").sum().fillna(0)
    df = df.loc[START_DATE:END_DATE]
    df.to_csv(path)
    print(f"✅ Saved Earthquake counts → {path}")
    return df

# -------------------------------------------------------
# === Stage 2: Merge and align data ===
# -------------------------------------------------------

def merge_all():
    print("\n🔄 Merging all available datasets...")

    spx = fetch_yahoo("^GSPC", "SPX_Close")
    gold = fetch_yahoo("GC=F", "Gold_Close")
    dxy = fetch_yahoo("DX-Y.NYB", "DXY")
    sun = fetch_sunspots()
    kp = fetch_kp_index()
    quake = fetch_earthquake_counts()

    df = pd.concat([spx, gold, dxy, sun, kp, quake], axis=1, join="outer")

    # Fill missing values appropriately
    df["SPX_Close"] = df["SPX_Close"].ffill()
    df["Gold_Close"] = df["Gold_Close"].ffill()
    df["DXY"] = df["DXY"].ffill()
    df["Sunspot"] = df["Sunspot"].bfill().ffill()
    df["Kp"] = df["Kp"].bfill().ffill()
    df["Quake_Count"] = df["Quake_Count"].fillna(0)

    df = df.loc[START_DATE:END_DATE]
    df.index.name = "Date"

    merged_path = os.path.join(RESULTS_DIR, "spx_exogenous_features_merged.csv")
    df.to_csv(merged_path)
    print(f"✅ Final merged dataset saved → {merged_path}")
    print(f"📊 Shape: {df.shape}")
    return df

# -------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------

if __name__ == "__main__":
    print("=== SPX Exogenous Data Pipeline ===")
    df = merge_all()
    print(df.head(10))
