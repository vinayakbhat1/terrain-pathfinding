import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import cv2
import numpy as np
from sklearn.cluster import KMeans

# TERRAIN DEFINITIONS (UI Colors & Costs)
TERRAIN = {
    "water":      {"cost": 1000, "color": [40, 120, 255]},
    "snow":       {"cost": 70,   "color": [245, 245, 245]},
    "forest":     {"cost": 80,   "color": [0, 90, 0]},
    "vegetation": {"cost": 50,   "color": [120, 220, 0]},
    "rocky":      {"cost": 40,   "color": [170, 130, 90]},
    "urban":      {"cost": 5,    "color": [180, 180, 180]},
    "open_land":  {"cost": 10,   "color": [255, 180, 40]}
}

# REFERENCE PROFILES 
_REFERENCE_PROFILES_RGB = {
    "water": [
        np.array([15, 20, 25]),
        np.array([30, 45, 60]),
        np.array([10, 80, 200]),
        np.array([170, 215, 255])
    ],
    "snow": [
        np.array([235, 245, 255])
    ],
    "forest": [
        np.array([25, 35, 25]),
        np.array([45, 65, 45]),
        np.array([0, 70, 30])
    ],
    "vegetation": [
        np.array([85, 115, 65]),
        np.array([110, 140, 75]),
        np.array([120, 180, 100])
    ],
    "rocky": [
        np.array([110, 95, 80]),
        np.array([150, 130, 110])
    ],
    "urban": [
        np.array([145, 145, 145]),
        np.array([230, 230, 230]),
        np.array([255, 255, 255])
    ],
    "open_land": [
        np.array([170, 150, 115]),
        np.array([200, 180, 140]),
        np.array([255, 230, 150])
    ]
}


def _rgb_to_hsv(rgb_array):
    """Converts a 1D RGB array to a 1D HSV array (OpenCV scale)."""
    rgb_clipped = np.clip(rgb_array, 0, 255).astype(np.uint8)
    pixel = np.array([[rgb_clipped]], dtype=np.uint8)
    return cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0].astype(np.float32)


# Convert all reference profiles to HSV once at load time
REFERENCE_PROFILES_HSV = {
    name: [_rgb_to_hsv(p) for p in profiles]
    for name, profiles in _REFERENCE_PROFILES_RGB.items()
}


def hsv_distance(a, b):

    # Normalize to 0-1
    ah, ash, av = a[0] / 179.0, a[1] / 255.0, a[2] / 255.0
    bh, bsh, bv = b[0] / 179.0, b[1] / 255.0, b[2] / 255.0

    # Hue wrap correction on normalized scale
    dh  = abs(ah - bh)
    dh  = min(dh, 1.0 - dh)

    ds  = abs(ash - bsh)
    dv  = abs(av - bv)

    # Weight: H x2 (most important), S x1, V x1
    return np.sqrt((2.0 * dh)**2 + ds**2 + dv**2)


# MAIN FUNCTION
def build_grid(rgb_image: np.ndarray, k: int = 7):
    H, W, _ = rgb_image.shape

    # 1. Convert RGB → HSV, run KMeans in HSV space
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    pixels    = hsv_image.reshape(-1, 3).astype(np.float32)

    km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
    labels    = km.fit_predict(pixels)
    centroids = km.cluster_centers_   # centroids in HSV space

    # 2. Match each HSV centroid to closest terrain using normalized HSV distance
    cluster_to_terrain = {}
    for i, centroid in enumerate(centroids):
        min_dist   = float('inf')
        best_match = "open_land"

        for name, hsv_profiles in REFERENCE_PROFILES_HSV.items():
            for hsv_ref in hsv_profiles:
                dist = hsv_distance(centroid, hsv_ref)
                if dist < min_dist:
                    min_dist   = dist
                    best_match = name

        cluster_to_terrain[i] = best_match

    # 3. Reconstruct cost grid and colour map
    grid   = np.zeros(H * W, dtype=np.float32)
    colors = np.zeros((H * W, 3), dtype=np.uint8)

    for i in range(k):
        mask         = (labels == i)
        terrain_name = cluster_to_terrain[i]
        grid[mask]   = TERRAIN[terrain_name]["cost"]
        colors[mask] = TERRAIN[terrain_name]["color"]

    return grid.reshape(H, W), colors.reshape(H, W, 3)