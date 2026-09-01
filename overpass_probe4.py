"""Try berl.freifunk mirror for Macau with SSL verification disabled."""
import json
import ssl
import urllib.request

ctx = ssl._create_unverified_context()
BIG = '[out:json][timeout:300];(way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"](around:3500,22.1987,113.5439);node(w););out body;>;out skel qt;'
req = urllib.request.Request("https://overpass.berl.freifunk.net/api/interpreter", data=BIG.encode(), method="POST")
try:
    with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
        body = r.read()
    j = json.loads(body)
    els = j.get("elements", [])
    print("OK bytes", len(body), "nodes", sum(1 for e in els if e.get("type") == "node"),
          "ways", sum(1 for e in els if e.get("type") == "way"))
except Exception as e:
    print("FAIL", str(e)[:150])