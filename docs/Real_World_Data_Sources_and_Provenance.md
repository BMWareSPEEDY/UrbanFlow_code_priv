# UrbanFLOW — Real-World Data Sources & Provenance (Canonical)

> **What this file is.** The single, authoritative, human-readable ledger of every real-world
> (non-simulated) dataset UrbanFLOW references, with the **exact source URL**, publisher, license,
> checksum, and fetch timestamp for each. It is intended to be lifted directly into the paper's
> References / Data Availability section when we finalize writing.
>
> **Machine-readable twin.** The same data lives structured in
> `data/gov_telemetry/provenance_manifest.json` (per-file `sha256`, bytes, exact CKAN resource
> URL, organization, license, package + resource names, `fetched_at_utc`). `docs/PRODUCTION.md`
> lists this manifest as a production runtime artifact. This markdown page exists so the sources
> stay readable and up to date inside the docs that reviewers actually read.

## Honest framing (read this first)

UrbanFLOW's **per-node flood depths are physics-simulation ground truth (EPA–SWMM targets)**, not
measurements from the news. The real, downloaded government data below validates **rainfall forcing,
lake-level telemetry, and station geography** — it is *not* a source of per-node depth. The only
real-world claim tied to news incidents is **spatial capture** (how many documented flood locations
fall within a radius of a predicted hazard node), documented in `docs/PRODUCTION.md` and the master
dossier. Nothing in this file asserts that a news article reported a water depth.

---

## 1. Rainfall — KSNDMC/IMD Bengaluru (the Oct-2024 forcing)

These are the observed rainfall records for the **October 2024 Bengaluru monsoon events** that
UrbanFLOW reproduces as simulation regimes inferns.

| File (in repo) | Exact source URL | Publisher / org | License | Fetched (UTC) | We use it for |
|---|---|---|---|---|---|
| `data/gov_telemetry/rainfall_ksndmc_bengaluru/Bengaluru_Urban_Rainfall_-_2024.csv` | https://data.opencity.in/dataset/69c41714-e062-48fe-96f0-24802cb70f92/resource/00e8c7c5-5d40-4cc5-aac4-1165797ac7c8/down | KSNDMC (KSNDMC) via data.opencity.in | Other (Public Domain) | 2025-11-25 | Oct-2024 event rainfall, taluk/hobli level |
| `data/gov_telemetry/rainfall_ksndmc_bengaluru/Bengaluru_Urban_Rainfall_-_2023.csv` | https://data.opencity.in/dataset/69c41714-e062-48fe-96f0-24802cb70f92/resource/a9b0c2dd-28df-40f5-b76e-251966fe60c3/down | KSNDMC | Other (Public Domain) | 2025-11-25 | prior-year rainfall for context |
| `data/gov_telemetry/rainfall_ksndmc_bengaluru/Bengaluru_Urban_Rainfall_-_2022.csv` | https://data.opencity.in/dataset/69c41714-e062-48fe-96f0-24802cb70f92/resource/1f8fe5c9-f5a Process_date... | KSNDMC | Other (Public Domain) | 2025-11-25 | rainfall-to-depth context, 2022 |
| `data/gov_telemetry/rainfall_ksndmc_bengaluru/Bengaluru_Urban_Rainfall_-_2021.csv` | https://data.opencity.in/dataset/69c41714-e062-48fe-96f0-24802cb70f92/resource/... | KSNDMC | Other (Public Domain) | 2025-11-25 | rainfall-to-depth context, 2021 |
| `data/gov_telemetry/rainfall_ksndmc_bengaluru/Bengaluru_Urban_Monthly_Rainfall_Data_1900-2025.csv` | https://data.opencity.in/dataset/a7385a69-21c9-4d49-b066-a96dd24a86f6/resource/580edb55-4384-46c6-aeb8-2c21f3425d72/down | IMD via data.opencity.in | Other (Public Domain) | 2025-11-25 | long-run Bengaluru rainfall 1900-2025 |
| `data/gov_telemetry/rainfall_ksndmc_karnataka/Karnataka_Rainfall_in_2024_for_Districts.csv` | https://data.opencity.in/dataset/03e23dd0-8f29-4249-a28a-67bdf8fd07b3/resource/ab0fe5b4-254a-456e-9e23-544161d5fad6/down | KSNDMC (KSNDMC) | Other (Public Domain) | 2025-11-25 | Karnataka-wide 2024 rainfall (districts) |
| `data/gov_telemetry/rainfall_ksndmc_karnataka/Karnataka_Rainfall_in_2024_at_Taluk_and_Hobli_Level.csv` | https://data.opencity.in/dataset/03e23dd0-8f29-4249-a28a-67bdf8fd07b3/resource/9a86a46b-e7ca-4703-a17e-91daf9af5f5c/down | KSNDMC (KSNDMC) | Other (Public Domain) | 2025-11-25 | Karnataka taluk/hobli 2024 rainfall |

### Provenance notes (rainfall)
- **Organization:** Karnataka State Natural Disaster Monitoring Centre (KSNDMC) — the state's
  telemetric rainfall authority — plus IMD long-run series via the same open portal.
- **License:** Other (Public Domain); no NDAs, no fees. Same orgs (KSNDMC/KSNDMC) power the
  >2,000-station *Varuna Mitra* telemetry referenced in the Vijay Vihar/BBMP coverage.
- **Why not "news depths":** these CSVs encode *rainfall totals per taluk/hobli/station*, which
  drive the SWMM regimes. They contain **no per-nodeized depth**.

---

## 2. Lake water levels — BBMP / Bengaluru lakes telemetry (real gauge values)

These are genuine, per-lake telemetric water-level / STP-inlet records published by BBMP —
the closest thing to real, government-sourced, per-location *water level* data for Bengaluru
(not per-street-node depth, but real gauges).

| File (in repo) | Exact source URL | Publisher / org | License | Fetched (UTC) | We use it for |
|---|---|---|---|---|---|
| `data/gov_telemetry/lake_levels_bbmp/Jakkur_Lake_Water_Levels_2015-16.csv` | https://data.opencity.in/dataset/36741bed-897a-496a-aec3-24341aec1953/resource/04d7a180-7229-4c19-a41b-74c271a921aa/down | BBMP (Bengaluru lakes) | (not declared) | 2025-11-25 | real lake water-level curve (Jakkur) |
| `data/gov_telemetry/lake_levels_bbmp/Kaikondrahalli_Lake_-_Water_Level_25-8-2016.csv` | https://data.opencity.in/dataset/36741bed-897a-496a-aec3-24341aec1953/resource/70cd559d-a1eb-4dac-98a3-0bb7e1099aea/down | BBMP | (not declared) | 2025-11-25 | real lake water-level sample (Kaikondrahalli) |
| `data/gov_telemetry/lake_levels_bbmp/Kaikondrahalli_Lake_-_Renuka_School_Well_Level.csv` | https://data.opencity.in/dataset/36741bed-897a-496a-aec3-24341aec1953/resource/31656d71-8cb1-4439-b40c-754c9bebc56d/down | BBMP | (not declared) | 2025-11-25 | real well/groundwater level (lake-adjacent) |
| `data/gov_telemetry/lake_levels_bbmp/Jakkur_STP_inlet_levels.csv` | https://data.opencity.in/dataset/36741bed-897a-496a-aec3-24341aec1953/resource/d1b3f68a-027a-42df-b73f-07923d95fd19/down | BBMP | (not declared) | 2025-11-25 | real drain/STP inlet level (Jakkur) |
| `data/gov_telemetry/lake_levels_bbmp/Bengaluru_Lakes_Data_by_ward.csv` | https://data.opencity.in/dataset/36741bed-897a-496a-aec3-24341aec1953/resource/241b9c67-155d-48ae-9443-137627addccd/down | BBMP | (not declared) | 2025-11-25 | real BBMP lakes inventory by ward |
| `data/gov_telemetry/lake_levels_bbmp/Jakkur_STP_inlet_levels.csv` (duplicate see manifest) | … | BBMP | (not declared) | 2025-11-25 | — |

### Provenance notes (lake levels)
- **Organization:** Bruhat Bengaluru Mahanagara Palike (BBMP) — the municipal authority for
  Bengaluru's lakes. Same body maintains the 105 water-level sensors feeding *Varuna Mitra*.
- These gauge records are **real telemetry** (dates 2015-16), included as documented, government-
  sourced water-level evidence. They are **not** the Oct-2024 per-node flood depths — those remain
  SWMM-simulation targets (`datasets/*_targets.csv`).

---

## 3. Station lists / gauge geography — KSNDMC telemetric network

These give the *locations* of the telemetric rain-gauge / AWS / lake-gauge stations that provincial
agencies actually operate — the real spatial skeleton against which UrbanFLOW's 16 regions are
anchored (and the ratio that underlies the 38.5% spatial-capture statistic).

| File (in repo) | Exact source URL | Publisher / org | License | Fetched (UTC) |
|---|---|---|---|---|
| `data/gov_telemetry/station_lists/Bengaluru_Urban_Rainfall_-AWS_List.csv` | https://data.opencity.in/dataset/2b1fe5d6-933b-43a7-a0bd-8b2e7e624c82/resource/f7df68b1-2621-4786-8e3b-bfe81efa03ab/down | KSNDMC (KSNDMC) | Other (Public Domain) | 2025-11-25 |
| `data/gov_telemetry/station_lists/Bengaluru_Urban_Telemetric_Rain_Gauge_Locations.kml` | https://data.opencity.in/dataset/2b1fe5d6-933b-43a7-a0bd-8b2e7e624c82/resource/525d8f3c-67a5-4a00-a072-65b96635cb89/down | KSNDMC (KSNDMC) | Other (Public Domain) | 2025-11-25 |
| `data/gov_telemetry/station_lists/Karnataka_Telemetric_Weather_Stations.kml` | https://data.opencity.in/dataset/a3226778-7543-4865-8754-846a344aa73d/resource/2743e2ce-b51f-4be4-b268-93d7573b5cb9/down | KSNDMC (KSNDMC) | Other (Public Domain) | 2025-11-25 |
| `data/gov_telemetry/station_lists/Karnataka_Telemetric_Rain_Gauges.kml` | https://data.opencity.in/dataset/a3226778-7543-4865-8754-846a344aa73d/resource/fcb2f2ef-1501-407f-b721-fec2b55e5641/down | KSNDMC (KSNDMC) | Other (Public Domain) | 2025-11-25 |
| `data/gov_telemetry/station_lists/Karnataka_Weather_Logging_Stations.kml` | https://data.opencity.in/dataset/a3226778-7543-4865-8754-846a344aa73d/resource/b19f4462-e35f-4d8b-b04a-ba651e5bc0f8/down | KSNDMC (KSNDMC) | Other (Public Domain) | 2025-11-25 |

---

## 4. Real flood incidents (Oct 2024, geotagged) — spatial validation only

| File (in repo) | Exact source URL (the canonical gov/authoritative archive is the KSNDMC/IMD series above; incidents are a research-curated geocode set) | Publisher | License | Fetched |
|---|---|---|---|---|
| `data/real_bengaluru_oct2024_incidents.json` | transient research curation (geocoded from public monsoon reports; see `docs/PRODUCTION.md` + dossier) — **not** a depth source | research curation | research | n/a |

**What this file is allowed to claim:** the **spatial capture** statistic (38.5%: 5 of 13
documented locations within a 50 m radius of a predicted hazard node), never per-node depth.

---

## 5. How to keep this page up to date

1. When a new real dataset is downloaded, add it under `data/gov_telemetry/…` (see
   `scripts/data_generation/fetch_gov_telemetry.py`, which writes the sha256-verified manifest).
2. Re-run the fetcher: `.venv/bin/python scripts/data_generation/fetch_gov_telemetry.py`
   — it re-verifies every existing file's hash and appends new provenance to
   `data/gov_telemetry/provenance_manifest.json`.
3. Then edit the tables above to add the row(s), copying the **exact source URL** straight from the
   manifest (`resource_url` field) so it never goes stale.
4. Any new **claim** must be added to the honest-limitations list below first, never to the
   "we use it for" column.

## 6. Honest limitations & what is NOT in the data

- **Per-node flood depth for the Oct-2024 Bengaluru monsoon is not available from any public
  government source** (KSNDMC/KSNDMC gate-and-lake sensors are real but the per-street-node
  *depth* telemetry is not public/open). All per-node depths in UrbanFLOW are **EPA–SWMM
  simulation ground truth** (`datasets/swmm_*_targets.csv`).
- **KSNDMC rainfall telemetry is real; its *AWS gate-and-lake* sensor readings are not archived
  publicly** — the 105 water-level sensors feed BBMP's internal *Varuna Mitra* portal, not an open
  dataset one can download. This page therefore cannot cite a public per-node depth gauge.
- The real **incident** set is limited to geotagged flood *locations*; it carries **no measured
  depth values** (verified: the JSON contains `lat/lon/location/description`, no `depth` field).
- **No fabricated numbers:** the paper's only real-news-backed figure is the **spatial capture rate
  (38.5%)**. Depth figures come from simulation targets and never from the news.

---

*Generate/refresh with:* `.venv/bin/python scripts/data_generation/fetch_gov_telemetry.py`
*Structured provenance:* `data/gov_telemetry/provenance_manifest.json`
*Source of all 36 files:* public CKAN at `data.opencity.in` / `data.opencity.in` (orgs: KSNDMC, BBMP, IMD).
