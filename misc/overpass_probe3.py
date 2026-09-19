"""Inspect osm.ch big-query response body."""
import urllib.request

BIG = '[out:json][timeout:300];(way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"](around:3500,22.1987,113.5439);node(w););out body;>;out skel qt;'
req = urllib.request.Request("https://overpass.osm.ch/api/interpreter", data=BIG.encode(), method="POST")
with urllib.request.urlopen(req, timeout=300) as r:
    body = r.read()
print(body[:2000].decode())