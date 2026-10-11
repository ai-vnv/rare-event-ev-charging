# Surabaya case study

Four files describe one planning instance. To study another city, keep the column
names and replace the rows.

| File | Rows | Columns |
|---|---|---|
| `stations.csv` | 11 candidate stations | `station_id`, `station`, `lat`, `lon`, `daily_cost_idr` (h_j), `demand_mean`, `demand_sd`, `outage_threshold` (mu_j, sigma_j, z_j of Table I) |
| `demand.csv` | 31 sub-districts (demand nodes) | `subdistrict`, `lat`, `lon`, `population`, `demand_motorcycle`, `demand_car` (w_i^k, vehicles per day) |
| `travel_time.csv` | 341 demand-station pairs | `subdistrict`, `station`, `minutes_motorcycle`, `minutes_car` (d_ij^k) |
| `params.json` | model parameters | Table II of the paper, plus the connector limits (`connector_minutes_per_day`, `max_connectors_per_station`) and `existing_stations` |

## Provenance

- **Candidate stations** are customer service units of the local electricity utility in
  Surabaya. `daily_cost_idr` is the daily operating and investment cost of a station at
  that site.
- **Electricity demand parameters** (`demand_mean`, `demand_sd`, `outage_threshold`) are
  the masked and rescaled values the utility provided, as printed in Table I of the
  paper. A station has an outage on a day when its demand exceeds the threshold, so its
  reliability is `Phi((z - mu) / sigma)`, about 0.977 to 0.979.
- **Charging demand** is the estimated number of electric motorcycles and cars that
  charge per day in each sub-district, as used in the paper.
- **Travel times** are average travel times in minutes by motorcycle and by car,
  collected from Google Maps.
- **Parameters** in `params.json` are those of Table II. `existing_stations` lists the
  utility's main charging station, which is already in operation and stays open in every
  design.
- **Basemap** (`basemap/surabaya_canvas.jpg`) is the Carto *Positron* basemap, rendered
  once at the map window of the paper by `scripts/fetch_basemap.py` and committed so
  that the figures build offline. `basemap/surabaya_canvas.json` records the window and
  the attribution the figures print. Map data (c) OpenStreetMap contributors, available
  under the Open Database License; the Positron style is (c) Carto, licensed CC BY 3.0.
  Rebuilding it needs a free Carto API key; `--provider esri` fetches a key-free
  equivalent instead. The figures still render without a basemap.
