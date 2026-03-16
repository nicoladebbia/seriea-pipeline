#!/usr/bin/env python3
"""Download historical Serie A data from Football-Data.co.uk.

This data includes:
- Match results (home/away goals, half-time scores)
- Match statistics (shots, shots on target, corners, fouls, cards)
- Betting odds from 6+ bookmakers (B365, BW, IW, PS, WH, VC)
- Over/under and Asian handicap lines
"""

import time
from pathlib import Path
import requests
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "data" / "external" / "football-data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seasons in Football-Data format (last 2 digits of each year)
SEASONS = [
    "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/I1.csv"

def download_season(season: str) -> pd.DataFrame | None:
    """Download a single season's data."""
    url = BASE_URL.format(season=season)
    output_path = OUTPUT_DIR / f"serie_a_{season}.csv"
    
    if output_path.exists():
        print(f"  {season}: already downloaded, loading from cache")
        return pd.read_csv(output_path)
    
    print(f"  {season}: downloading from {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save raw CSV
        output_path.write_bytes(response.content)
        
        # Return as DataFrame
        df = pd.read_csv(output_path)
        print(f"  {season}: {len(df)} matches downloaded")
        return df
    except Exception as e:
        print(f"  {season}: ERROR - {e}")
        return None

def main():
    print("Downloading Football-Data.co.uk Serie A data...")
    print()
    
    all_dfs = []
    for season in SEASONS:
        df = download_season(season)
        if df is not None and not df.empty:
            # Add season column
            season_fmt = f"20{season[:2]}-20{season[2:]}"
            df["season"] = season_fmt
            all_dfs.append(df)
        time.sleep(2)  # Be nice to the server
    
    if all_dfs:
        # Combine all seasons
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = OUTPUT_DIR / "serie_a_combined.csv"
        combined.to_csv(combined_path, index=False)
        print()
        print(f"Combined dataset: {len(combined)} matches saved to {combined_path}")
        print(f"Seasons: {combined['season'].unique().tolist()}")
        
        # Show column summary
        print(f"\nColumns ({len(combined.columns)}):")
        for col in combined.columns:
            non_null = combined[col].notna().sum()
            print(f"  {col}: {non_null}/{len(combined)} non-null")
    else:
        print("No data downloaded!")

if __name__ == "__main__":
    main()
