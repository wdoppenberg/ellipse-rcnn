from pathlib import Path

"""
Project constants
"""
# Physical
RMOON = 1737.1  # Body radius (moon) [km]


# Camera
CAMERA_FOV = 45  # Camera field-of-view (degrees)
CAMERA_RESOLUTION = (256, 256)


# Dataset generation
DIAMLIMS = (4, 100)  # Limit dataset to craters with diameter between 4 and 30 km
MAX_ELLIPTICITY = 1.3  # Limit dataset to craters with an ellipticity <= 1.1]
ARC_LIMS = 0.0
AXIS_THRESHOLD = (5, 100)
MIN_SOL_INCIDENCE = 10
MAX_SOL_INCIDENCE = 80
FILLED = True
INSTANCING = True
RANDOMIZED_ORIENTATION = True
MASK_THICKNESS = 1
SAVE_CRATERS = True

GENERATION_KWARGS = dict(
    axis_threshold=AXIS_THRESHOLD,
    resolution=CAMERA_RESOLUTION,
    fov=CAMERA_FOV,
    min_sol_incidence=MIN_SOL_INCIDENCE,
    max_sol_incidence=MAX_SOL_INCIDENCE,
    filled=FILLED,
    ellipse_limit=MAX_ELLIPTICITY,
    arc_lims=ARC_LIMS,
    diamlims=DIAMLIMS,
    instancing=INSTANCING,
    randomized_orientation=RANDOMIZED_ORIENTATION,
    mask_thickness=MASK_THICKNESS,
    save_craters=SAVE_CRATERS
)


# Database generation
TRIAD_RADIUS = 200  # Create triangles with every crater within this TRIAD_RADIUS [km]
DB_CAM_ALTITUDE = 300


# SPICE
SPICE_BASE_URL = 'https://naif.jpl.nasa.gov/pub/naif/'
KERNEL_ROOT = Path('data/spice_kernels')
