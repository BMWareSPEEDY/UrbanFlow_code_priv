import re
with open('static/js/main.js', 'r', encoding='utf-8-sig') as f:
    content = f.read()

endpoints = re.findall(r"fetch\(['\"]([^'\"]+)['\"]", content)
print('JS calls these endpoints:')
for e in endpoints:
    print(f'  {e}')

# Also check what the old main.js had for runSimBtn
if 'run-simulation' in content:
    print('\nWARNING: JS still references /api/run-simulation')
if '/api/predict' in content:
    print('OK: JS references /api/predict')
if '/api/mitigation' in content:
    print('OK: JS references /api/mitigation')
if '/api/storm-playback' in content:
    print('OK: JS references /api/storm-playback')
if '/api/swmm-compare' in content:
    print('OK: JS references /api/swmm-compare')
if '/api/historical-validation' in content:
    print('OK: JS references /api/historical-validation')
