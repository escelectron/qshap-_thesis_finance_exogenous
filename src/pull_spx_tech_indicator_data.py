#!/usr/bin/env python3
"""
SPX Hourly Feature Extraction Script (Final Fixed)
Author: Pranav K. Sanghadia
License: MIT
Description:
    Fetches SPX (^GSPC) hourly data from Yahoo Finance and computes
    Volume, A/D Line, ADX, MACD, and RSI indicators.
    Generates binary target (1/0) based on next day's close proximity (±30 points).
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta  # pip install ta

# === Parameters ===
TICKER = "^GSPC"
TARGET_THRESHOLD = 30
OUTPUT_FILE = "spx_features.csv"

# === Step 1: Download SPX hourly data ===
print("Downloading SPX hourly data from Yahoo Finance...")
spx = yf.download(TICKER, interval="1h", period="730d")
spx.dropna(inplace=True)

# --- Flatten MultiIndex columns if present ---
if isinstance(spx.columns, pd.MultiIndex):
    spx.columns = [col[0] if isinstance(col, tuple) else col for col in spx.columns]

# --- Ensure columns are 1D Pandas Series ---
for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if c in spx.columns:
        if isinstance(spx[c], pd.DataFrame):
            spx[c] = spx[c].iloc[:, 0]
        spx[c] = pd.to_numeric(spx[c], errors='coerce')

# === Step 2: Compute indicators ===
print("Computing technical indicators...")

# Accumulation / Distribution Line (A/D)
spx['A/D'] = ((spx['Close'] - spx['Low']) - (spx['High'] - spx['Close'])) / \
             (spx['High'] - spx['Low']) * spx['Volume']
spx['A/D'] = spx['A/D'].cumsum()

# Average Directional Index (ADX)
adx = ta.trend.ADXIndicator(
    high=spx['High'],
    low=spx['Low'],
    close=spx['Close'],
    window=14
)
spx['ADX'] = adx.adx()

# MACD (12, 26, 9)
macd = ta.trend.MACD(spx['Close'])
spx['MACD'] = macd.macd()

# RSI (14)
rsi = ta.momentum.RSIIndicator(spx['Close'], window=14)
spx['RSI'] = rsi.rsi()

# Keep relevant features
spx = spx[['Close', 'Volume', 'A/D', 'ADX', 'MACD', 'RSI']].dropna()

# === Step 3: Create Target Column ===
print("Creating target column...")
spx['Next_Close'] = spx['Close'].shift(-24)
spx['Diff'] = spx['Next_Close'] - spx['Close']
spx['Target'] = np.where(spx['Diff'].abs() <= TARGET_THRESHOLD, 1, 0)
spx.dropna(inplace=True)

# === Step 4: Save to CSV ===
spx.to_csv(OUTPUT_FILE)
print(f"✅ Dataset saved to {OUTPUT_FILE}")
print(spx.head(10))
