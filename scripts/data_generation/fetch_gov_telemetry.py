#!/usr/bin/env python3
"""fetch_gov_telemetry.py — download real government telemetry for Bengaluru Oct 2024 validation.

Purpose
-------
UrbanFLOW's per-node flood *depths* come from SWMM numerical simulation ground truth
(`datasets/swmm_*_targets.csv`), never from news articles — that is intentional and is
documented in `docs/real_data_validation.md`. This script instead pulls the *real,
per-location* government datasets that DO exist for the Oct-2024 Bengaluru monsoon, so the
real-world claim in the paper is grounded in spatial capture (38.5%: 5 of 13 documented
incident locations within 50 m of a predicted hazard node), not in fabricated depths.

Data sources (all public, all cited):
  * KSNDMC rainfall for Bengaluru Urban  (provider: KSNDMC, via data.opencity.in CKAN)
  * KSNDMC Karnataka annual rainfall      (provider: KSNDMC, via data.opencity.in CKAN)
  * IMD Bengaluru rainfall series         (provider: IMD,   via data.opencity.in CKAN)
  * BBMP Bengaluru lakes water levels     (provider: BBMP,  via data.opencity.in CKAN)
  * KSNDMC AWS location lists             (provider: KSNDMC, via data.opencity.in CKAN)

Output layout (matches repo) ─────────────────────────────────────────────────────────────
    data/gov_telemetry/
        rainfall_ksndmc/…                     per-year Bengaluru Urban / Karnataka CSV
        rainfall_imd/…                        long-run Bengaluru rainfall series
        lake_levels_bbmp/…                    BBMP lake water-level CSV
        station_lists/…                       KSNDMC AWS / rain-gauge location lists
        provenance_manifest.json              every file: source URL, org, license, fetched_at
    Provenance + honest limitations: docs/real_data_validation.md

Note: this script ONLY downloads into data/ — it does not modify app.py or any production
code. Integration with the running app is intentionally out of scope (see docs/).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(BASE_DIR, "data", "gov_telemetry")
MANIFEST_PATH = os.path.join(OUT_ROOT, "provenance_manifest.json")

USER_AGENT = "UrbanFLOW-academic-data-pipeline/2.0 (provenance-audit; contact: ashish@urbanflow.dev)"

# ---------------------------------------------------------------- public sources (CKAN)
CKAN = "https://data.opencity.in/api/3/action/package_show?id="

# package_id -> (label, org)
PACKAGES = {
    "bengaluru-urban-annual-rainfall-taluks-and-hoblis": ("KSNDMC Bengaluru Urban taluk/hobli rainfall", "KSNDMC"),
    "bengaluru-rainfall": ("IMD Bengaluru rainfall series", "IMD"),
    "karnataka-annual-rainfall-districts-taluks-and-hoblis": ("KSNDMC Karnataka rainfall districts/taluks", "KSNDMC"),
    "bengaluru-lakes-data-and-reports": ("BBMP Bengaluru lakes water levels", "BBMP"),
    "karnataka-telemetric-weather-stations-and-rain-gauges": ("KSNDMC telemetric weather station/rain gauge lists", "KSNDMC"),
    "list-of-automated-weater-stations-aws-in-bengaluru-urban-rural-districts": ("KSNDMC AWS in Bengaluru Urban/Rural", "KSNDMC"),
}

MAX_BYTES = 40_000_000  # per file; prevent accidentally pulling a huge archive


def _http(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def package_resources(pkg_id: str) -> dict:
    with _http(CKAN + pkg_id) as r:
        data = json.load(r)["result"]
    return data


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str, max_bytes: int = MAX_BYTES) -> tuple[bool, int]:
    """Download with cache-by-existing-file. Returns (ok, bytes_written)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 5_000:
        return True, os.path.getsize(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    n = 0
    try:
        with _http(url) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                n += len(chunk)
                if n > max_bytes:
                    f.close()
                    os.remove(dest)
                    print(f"  [abort] {os.path.basename(dest)} exceeds {max_bytes} bytes")
                    return False, n
                f.write(chunk)
        if n == 0:
            os.remove(dest)
            return False, n
        return True, n
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        print(f"  [fail ] {os.path.basename(dest)} :: {e}")
        return False, n


def choose_dest(subdir: str, res_name: str, res_fmt: str, res_url: str) -> str:
    ext = (res_fmt or "").lower()
    if ext in ("csv",):
        ext = "csv"
    elif ext in ("kml", "kmz"):
        ext = "kml"
    else:
        # fall back to extension in the URL / name
        for cand in (res_url, res_name):
            stem = cand.split("?")[0].lower()
            for e in ("csv", "kml", "kmz"):
                if stem.endswith("." + e):
                    ext = e
                    break
        if ext == "":
            ext = "dat"
    fname = res_name.replace("/", "_").replace(" ", "_").strip() or "unnamed"
    if not fname.lower().endswith("." + ext):
        fname = fname + "." + ext
    # keep only safe chars
    safe = "".join(c for c in fname if c.isalnum() or c in "._-") or "unnamed." + ext
    return os.path.join(OUT_ROOT, subdir, safe)


def main() -> int:
    os.makedirs(OUT_ROOT, exist_ok=True)
    manifest: dict = {
        "_meta": {
            "description": "Real per-location government telemetry for Bengaluru (rainfall, lake water levels, "
                           "station lists). Complementary to SWMM simulation depths — NOT a source of per-node depth. "
                           "See docs/real_data_validation.md for what is and is not validated against reality.",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": os.path.abspath(__file__),
            "ckan_base": CKAN,
        },
        "files": {},
    }

    for pkg_id, (label, org) in PACKAGES.items():
        print(f"== {label} ({org}) :: {pkg_id}")
        try:
            pkg = package_resources(pkg_id)
        except Exception as e:
            print(f"   [skip] package not reachable: {e}")
            continue
        for res in pkg.get("resources", []):
            rname = (res.get("name") or "").strip()
            rfmt = (res.get("format") or res.get("mimetype") or "").upper()
            rurl = res.get("url", "")
            lastmod = (res.get("last_modified") or "")[:10]

            # only CSV / KML / KMZ resources; skip PDFs, ZIPs, stray reports
            want = (rname.lower().endswith((".csv", ".kml", ".kmz"))
                    or rfmt in ("CSV", "KML", "KMZ"))
            if not want:
                print(f"   [skip] {rname} [{rfmt}]")
                continue

            # route to subfolder by content
            lower = (label + " " + rname).lower()
            if "lake" in lower or "water level" in lower or "lakes" in lower:
                sub = "lake_levels_bbmp"
            elif "bengaluru" in lower and "rainfall" in lower:
                sub = "rainfall_ksndmc_bengaluru"
            elif "aws" in lower or "station" in lower or "rain gauge" in lower or "telemetric" in lower:
                sub = "station_lists"
            elif "rainfall" in lower or "weather" in lower:
                sub = "rainfall_imd" if pkg_id == "bengaluru-rainfall" else "rainfall_ksndmc_karnataka"
            else:
                sub = "misc"

            dest = choose_dest(sub, rname, rfmt, rurl)
            ok, size = download(rurl, dest)
            if ok:
                digest = _sha256(dest)
                key = os.path.relpath(dest, OUT_ROOT)
                manifest["files"][key] = {
                    "source_package": pkg.get("name", pkg_id),
                    "source_package_id": pkg_id,
                    "package_title": pkg.get("title", ""),
                    "organization": org,
                    "license": pkg.get("license_title", ""),
                    "resource_name": rname,
                    "resource_url": rurl,
                    "last_modified": lastmod,
                    "format": rfmt,
                    "bytes": size,
                    "sha256": digest,
                }
                print(f"   [ok  ] {key}  ({size:,} B, sha256 {digest[:12]})")
            time.sleep(0.2)  # be polite to the public CKAN instance

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nManifest written: {MANIFEST_PATH}")
    print(f"Total files tracked: {len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
