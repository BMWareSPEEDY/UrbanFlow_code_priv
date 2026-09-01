import pandas as pd

df = pd.read_csv(r"D:\CODES\PYTHON_CODES\UrbanFLOW\swmm_multi_scenario_targets.csv")
print(df.columns.tolist())
print(df['region'].value_counts().to_dict())
sub = df[df['region'].str.lower().isin(['bangalore', 'blr', 'bellandur', 'hsr'])]
print(sub['region'].value_counts().to_dict())
for rg, s in sub.groupby('region'):
    for I in [150.0, 300.0]:
        ss = s[abs(s['intensity_mmhr'] - I) < 1e-6]
        if len(ss) == 0:
            continue
        y = ss['max_water_depth_m'].values
        print(f"{rg:<12} I={I:5.0f} n={len(ss):6d} mean {y.mean():.3f} zero% {(y==0).mean()*100:5.1f} dry% {(y<0.15).mean()*100:5.1f}")