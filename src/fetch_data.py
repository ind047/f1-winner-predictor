"""Fetch F1 data from the Jolpica API (2010-2025) and save it into a csv file."""

import csv
import time
import requests


BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASONS = range(2010, 2026)
PAGE_SIZE = 100
OUTPUT_FILE = "data/races.csv"

COLUMNS = ["season","round","race_name","circuit","date",
           "driver_code","driver_id","constructor","grid",
           "position","position_order","points","laps","status",]


def fetch_season_results(season: int)-> list[dict]:
    """Fetch all race results for one season, paging through the API."""

    rows =[]
    offset = 0
    while True:
        url = f"{BASE_URL}/{season}/results.json?limit={PAGE_SIZE}&offset={offset}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        races = data["MRData"]["RaceTable"]["Races"]

        if not races:
            break

        for race in races:
            for result in race["Results"]:
                rows.append({
                    "season": season,
                    "round": race["round"],
                    "race_name": race["raceName"],
                    "circuit": race["Circuit"]["circuitName"],
                    "date": race["date"],
                    "driver_code": result["Driver"]["code"] if "code" in result["Driver"] else None,
                    "driver_id": result["Driver"]["driverId"],
                    "constructor": result["Constructor"]["name"],
                    "grid": result["grid"],
                    "position": result.get("position"),
                    "position_order": result.get("positionOrder"),
                    "points": result.get("points"),
                    "laps": result.get("laps"),
                    "status": result.get("status"),
                })

        total = int(data["MRData"]["total"])
        offset += PAGE_SIZE
        if offset >= total:
            break

        time.sleep(0.5)

    return rows


def main() -> None:
    all_rows = []
    for season in SEASONS:
        season_rows = fetch_season_results(season)
        print(f"Fetched {len(season_rows)} results for season {season}.")
        all_rows.extend(season_rows)
        time.sleep(0.5)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f,fieldnames= COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} rows to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
