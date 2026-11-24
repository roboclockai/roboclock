"""Recording-only third-person overview camera per task (1280x960, 36 mm aperture).

Not a policy observation. Poses follow each task's earlier review camera, widened where
the rig layout moved fixtures (44, 46, 47, 48, 49).
"""

OVERVIEW = {
    41: dict(eye=[0.90, -0.60, 1.90], target=[-0.04, 0.43, 0.97], focal_m=0.030),
    42: dict(eye=[-0.30, -1.00, 2.20], target=[-0.22, 0.42, 1.02], focal_m=0.026),
    43: dict(eye=[-0.16, -0.84, 1.98], target=[-0.04, 0.43, 1.00], focal_m=0.026),
    44: dict(eye=[0.70, -1.05, 1.95], target=[0.15, 0.50, 1.00], focal_m=0.024),
    45: dict(eye=[0.95, -1.25, 1.90], target=[-0.40, 0.55, 1.05], focal_m=0.024),
    46: dict(eye=[2.10, -0.55, 1.85], target=[0.90, 0.75, 1.00], focal_m=0.024),
    47: dict(eye=[0.40, -1.30, 2.45], target=[0.00, 0.45, 1.00], focal_m=0.022),
    48: dict(eye=[1.55, -0.10, 1.75], target=[0.12, 0.95, 0.65], focal_m=0.022),
    49: dict(eye=[0.75, -0.90, 1.85], target=[0.00, 0.58, 1.05], focal_m=0.022),
    50: dict(eye=[0.30, -1.10, 1.95], target=[0.00, 0.27, 1.02], focal_m=0.026),
}
