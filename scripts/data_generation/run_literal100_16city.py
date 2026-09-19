import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_parallel_swmm import run_single_swmm_task

CATCHMENTS = {
    'hsr': 'graphs/bengaluru_complete_graph.graphml',
    'bellandur': 'graphs/bengaluru_bellandur_graph.graphml',
    'whitefield': 'graphs/bengaluru_whitefield_graph.graphml',
    'ecity': 'graphs/bengaluru_ecity_graph.graphml',
    'koramangala': 'graphs/bengaluru_koramangala_graph.graphml',
    'tokyo': 'graphs/city_tokyo_graph.graphml',
    'hongkong': 'graphs/city_hongkong_graph.graphml',
    'singapore': 'graphs/city_singapore_graph.graphml',
    'london': 'graphs/city_london_graph.graphml',
    'paris': 'graphs/city_paris_graph.graphml',
    'nyc': 'graphs/city_nyc_graph.graphml',
    'chicago': 'graphs/city_chicago_graph.graphml',
    'berlin': 'graphs/city_berlin_graph.graphml',
    'bangkok': 'graphs/city_bangkok_graph.graphml',
    'mumbai': 'graphs/city_mumbai_graph.graphml',
    'delhi': 'graphs/city_delhi_graph.graphml',
}

SCEN = {'intensity': 100.0, 'duration': 60, 'profile': 'cloudburst'}
OUT = 'data/regime100_literal_swmm_rows_16city.json'

def main():
    print("CITY => running genuine literal-100 (100 mm/hr / 60 min) EPA SWMM via pyswmm")
    all_rows = {}
    total_t0 = time.time()
    for reg, path in CATCHMENTS.items():
        t0 = time.time()
        rows = run_single_swmm_task((reg, path, 0, SCEN))
        dt = time.time() - t0
        all_rows[reg] = rows
        print(f"  {reg:12s}: {len(rows):5d} rows in {dt:.1f}s", flush=True)
    json.dump(all_rows, open(OUT, 'w'), default=str, indent=1)
    print(f"\nsaved {OUT} ({sum(len(v) for v in all_rows.values())} total node-rows) in {time.time()-total_t0:.1f}s", flush=True)

if __name__ == '__main__':
    main()