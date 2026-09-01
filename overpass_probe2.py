"""Probe all known Overpass mirrors with a tiny query, then the big Macau query on the winners."""
import json
import time
import urllib.request

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    "https://overpass.monicz.dev/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.berl.freifunk.net/api/interpreter",
]
TINY = "[out:json];node(around:50,22.19,113.54);out 5;"
BIG = '[out:json][timeout:300];(way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"](around:3500,22.1987,113.5439);node(w););out body;>;out skel qt;'

for ep in MIRRORS:
    try:
        req = urllib.request.Request(ep, data=TINY.encode(), method="POST")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=45) as r:
            body = r.read()
        j = json.loads(body)
        print(f"{ep:55s} TINY OK ({len(body)}B, {round(time.time()-t0,1)}s)", flush=True)
    except Exception as e:
        print(f"{ep:55s} TINY FAIL {str(e)[:60]}", flush=True)
        continue
    try:
        req = urllib.request.Request(ep, data=BIG.encode(), method="POST")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
        j = json.loads(body)
        els = j.get("elements", [])
        n_nodes = sum(1 for e in els if e.get("type") == "node")
        n_ways = sum(1 for e in els if e.get("type") == "way")
        print(f"     BIG OK: {len(body)}B, nodes {n_nodes}, ways {n_ways}, {round(time.time()-t0,1)}s", flush=True)
    except Exception as e:
        print(f"     BIG FAIL {str(e)[:60]}", flush=True)