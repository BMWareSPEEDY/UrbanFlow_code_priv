"""Probe overpass-api.de with the full Macau query via urllib (bypassing requests stack)."""
import json
import time
import urllib.request

q = '[out:json][timeout:300];(way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"](around:3500,22.1987,113.5439);node(w););out body;>;out skel qt;'
req = urllib.request.Request(
    "https://overpass-api.de/api/interpreter",
    data=q.encode(),
    headers={"Content-Type": "application/x-www-form-urlencoded",
             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) urbanflow/1.0"},
    method="POST")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read()
    j = json.loads(body)
    print("OK", r.status, "bytes", len(body), "elements", len(j.get("elements", [])),
          "secs", round(time.time() - t0, 1))
    print("nodes:", sum(1 for e in j.get("elements", []) if e.get("type") == "node"))
    print("ways:", sum(1 for e in j.get("elements", []) if e.get("type") == "way"))
except Exception as e:
    print("FAIL", str(e)[:200])