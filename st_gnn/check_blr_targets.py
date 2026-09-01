import pandas as pd

for f in ['swmm_city_targets.csv', 'swmm_city_b4_targets.csv']:
    df = pd.read_csv(r"D:\CODES\PYTHON_CODES\UrbanFLOW" + "\\" + f)
    print(f, df['region'].value_counts().to_dict())
    sub = df[df['region'].str.lower() == 'bangalore']
    if len(sub):
        for I in [150.0, 300.0]:
            s = sub[abs(sub['intensity_mmhr'] - I) < 1e-6]
            y = s['max_water_depth_m'].values
            print(f"  bangalore I={I:5.0f} n={len(s):6d} mean {y.mean():.3f} zero% {(y==0).mean()*100:5.1f} dry% {(y<0.15).mean()*100:5.1f}")