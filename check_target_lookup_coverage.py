"""Check target lookup coverage in app.py across all regions.
"""
import sys
from app import REGION_CACHE

for r_key, r_data in REGION_CACHE.items():
    nl = r_data['node_list']
    np_dict = r_data['node_pos']
    # Check how many nodes used the fallback formula vs real target
    # The fallback formula always produced round(..., 4) where values are continuous
    # But let's check if the node id exists in target_lookup
    from app import target_lookup
    matched = sum(1 for nid in nl if nid in target_lookup or str(nid) in target_lookup or f"J_{nid}" in target_lookup)
    print(f"Region {r_key:<15s}: {matched:5d} / {len(nl):5d} nodes matched in target_lookup ({matched/len(nl)*100:.1f}%)")
