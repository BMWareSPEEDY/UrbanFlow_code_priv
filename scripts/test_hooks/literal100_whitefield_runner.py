import os, sys, time, json
os.chdir("/Users/ashishpaliwal/PycharmProjects/UrbanFLOW")
sys.path.insert(0, os.path.join(os.getcwd(), "scripts/data_generation"))

from generate_parallel_swmm import (
    region_files,
    STORM_SCENARIOS,
    generate_swmm_inp,
    run_single_swmm_task,
)

def main():
    t0 = time.time()
    # literal-100 cloudburst scenario by index
    idx = next(i for i, s in enumerate(STORM_SCENARIOS) if abs(s["intensity"] - 100.0) < 1e-6)
    scen = STORM_SCENARIOS[idx]
    reg = "whitefield"
    path = region_files[reg]
    if not os.path.exists(path):
        path = os.path.join("graphs", path)
    print(f"[{time.time()-t0:5.1f}s] running literal-100 cloudburst on {reg}: {path} (scen{idx} {scen})", flush=True)

    rows = run_single_swmm_task((reg, path, idx, scen))
    dur = time.time() - t0
    print(f"[{time.time()-t0:5.1f}s] engine returned {len(rows)} nodes in {dur:.1f}s", flush=True)

    out = {
        "regime": "literal_100_mmhr_stress",
        "rainfall_mmhr": 100.0,
        "duration_min": 60.0,
        "profile": "cloudburst",
        "region": reg,
        "engine": "generate_parallel_swmm.run_single_swmm_task -> pyswmm.Simulation (identical to shipped 16-city engine)",
        "node_count": len(rows),
        "flooded_gt_30cm": sum(1 for r in rows if r["max_water_depth_m"] > 0.30),
        "wall_s": round(dur, 1),
        "rows": rows,
    }
    with open("data/literal100_whitefield_genuine_swmm.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"[{time.time()-t0:5.1f}s] WROTE data/literal100_whitefield_genuine_swmm.json", flush=True)

if __name__ == "__main__":
    main()