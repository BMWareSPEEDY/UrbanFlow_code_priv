"""Inspect regional_residual_calibrations.pt.
"""
import torch

try:
    cal = torch.load("regional_residual_calibrations.pt", map_location='cpu', weights_only=False)
    print("Type of regional_residual_calibrations:", type(cal))
    if isinstance(cal, dict):
        print("Keys:", list(cal.keys())[:20])
        for k in list(cal.keys())[:5]:
            print(f"Sample {k}:", type(cal[k]), getattr(cal[k], 'shape', None))
except Exception as e:
    print("Error loading:", e)
