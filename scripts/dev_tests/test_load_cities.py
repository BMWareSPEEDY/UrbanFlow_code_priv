"""Test loading international city graphs.
"""
import os
import osmnx as ox
from pyproj import Transformer

test_cities = {
    'hongkong': ('Hong Kong Urban Basin', 'city_hongkong_graph.graphml', 22.3193, 114.1694),
    'tokyo': ('Tokyo Metropolitan Catchment', 'city_tokyo_graph.graphml', 35.6762, 139.6503),
    'london': ('London Thames Catchment', 'city_london_graph.graphml', 51.5074, -0.1278),
    'paris': ('Paris Seine Basin', 'city_paris_graph.graphml', 48.8566, 2.3522),
    'singapore': ('Singapore Marina Catchment', 'city_singapore_graph.graphml', 1.3521, 103.8198),
    'berlin': ('Berlin Spree Basin', 'city_berlin_graph.graphml', 52.5200, 13.4050),
    'bangkok': ('Bangkok Chao Phraya Lowlands', 'city_bangkok_graph.graphml', 13.7563, 100.5018),
    'chicago': ('Chicago Waterfront Catchment', 'city_chicago_graph.graphml', 41.8781, -87.6298),
    'mumbai': ('Mumbai Coastal Floodplain', 'city_mumbai_graph.graphml', 19.0760, 72.8777),
    'delhi': ('Delhi Yamuna Floodplain', 'city_delhi_graph.graphml', 28.6139, 77.2090),
}

for k, (name, fname, lat, lng) in test_cities.items():
    if os.path.exists(fname):
        G = ox.load_graphml(fname)
        print(f"OK {k}: {name} ({len(G.nodes)} nodes, {len(G.edges)} edges)")
    else:
        print(f"MISSING {fname}")
