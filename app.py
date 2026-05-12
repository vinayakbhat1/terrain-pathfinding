import streamlit as st
import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Path Planner", layout="wide")
st.title("Optimal Path Planner")

# ── TERRAIN COSTS ─────────────────────────────────────────────────────────────
TERRAIN = {
    "water":      {"cost": 9999, "color": [41,  120, 200]},
    "forest":     {"cost": 80,   "color": [27,  94,  32]},
    "vegetation": {"cost": 50,   "color": [76,  153, 0]},
    "rocky":      {"cost": 40,   "color": [141, 110, 99]},
    "urban":      {"cost": 5,    "color": [150, 150, 150]},
    "open_land":  {"cost": 10,   "color": [239, 159, 39]},
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader("Upload Satellite Image", type=["png","jpg","jpeg"])
k        = st.sidebar.slider("Terrain Clusters", 5, 10, 7)
run      = st.sidebar.button("Run", use_container_width=True)

# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def get_terrain_hsv(h, s, v):
    """
    Correctly classifies terrain from OpenCV HSV values:
      H: 0–179  (OpenCV halves the standard 0–360 range)
      S: 0–255
      V: 0–255
    """
    # --- Water ---
    # Deep water: dark pixels (very low V)
    if v < 50:
        return "water"
    # Blue/cyan hue range in OpenCV HSV: roughly H 90–130
    if 90 <= h <= 130 and s > 60:
        return "water"

    # --- Forest ---
    # Dark green: H 35–85, high saturation, medium-low brightness
    if 35 <= h <= 85 and s > 80 and v < 140:
        return "forest"

    # --- Vegetation ---
    # Lighter green: H 35–85, moderate saturation
    if 35 <= h <= 85 and s > 40:
        return "vegetation"

    # --- Urban ---
    # Grey/white: low saturation, high brightness
    if s < 40 and v >= 150:
        return "urban"

    # --- Rocky ---
    # Low saturation, mid brightness (bare earth, rocks)
    if s < 80 and 60 <= v < 150:
        return "rocky"

    # --- Open Land / Default ---
    return "open_land"


def build_grid(hsv_image, k):
    """
    Runs KMeans on the HSV image, classifies each cluster center,
    then assigns cost and color to every pixel.
    """
    H, W, _ = hsv_image.shape
    pixels   = hsv_image.reshape(-1, 3).astype(np.float32)
    km       = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pixels)

    grid      = np.zeros((H, W), np.float32)
    colors    = np.zeros((H, W, 3), np.uint8)
    label_map = km.labels_.reshape(H, W)

    for i, center in enumerate(km.cluster_centers_):
        # center is already in HSV space — pass directly
        h, s, v  = float(center[0]), float(center[1]), float(center[2])
        name     = get_terrain_hsv(h, s, v)
        mask     = label_map == i
        grid[mask]   = TERRAIN[name]["cost"]
        colors[mask] = TERRAIN[name]["color"]

    return grid, colors


def find_start_end(grid):
    """Finds nearest non-water pixel to top-left and bottom-right corners."""
    H, W = grid.shape

    def nearest_passable(r, c):
        for d in range(max(H, W)):
            for dr in range(-d, d + 1):
                for dc in range(-d, d + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] < 9999:
                        return (nr, nc)
        return (r, c)  # fallback (shouldn't happen)

    return nearest_passable(10, 10), nearest_passable(H - 10, W - 10)


def astar(grid, start, end):
    """
    A* pathfinding that treats water (cost >= 9999) as completely impassable walls.
    Uses Euclidean distance heuristic scaled by minimum terrain cost (5) for admissibility.
    """
    H, W = grid.shape
    INF  = float('inf')
    MIN_COST = 5  # urban — admissible heuristic scale

    heap = [(0.0, start)]
    came = {}
    g    = {start: 0.0}

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1,-1), (-1, 1), (1, -1), (1,  1)]

    def heuristic(a, b):
        return MIN_COST * np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    while heap:
        _, cur = heapq.heappop(heap)

        if cur == end:
            path = []
            while cur in came:
                path.append(cur)
                cur = came[cur]
            path.append(start)
            return path[::-1]

        for dr, dc in dirs:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < H and 0 <= nb[1] < W):
                continue

            nb_cost = grid[nb[0], nb[1]]

            # Hard wall — skip water entirely
            if nb_cost >= 9999:
                continue

            move_cost = nb_cost * (1.414 if dr and dc else 1.0)
            tg = g[cur] + move_cost

            if tg < g.get(nb, INF):
                came[nb] = cur
                g[nb]    = tg
                f        = tg + heuristic(nb, end)
                heapq.heappush(heap, (f, nb))

    return []  # no path found


def draw_path_thick(image, path, thickness=3):
    """Draws the path as a thick line so it's actually visible."""
    overlay = image.copy()
    if len(path) < 2:
        return overlay
    pts = np.array([[c, r] for r, c in path], dtype=np.int32)
    cv2.polylines(overlay, [pts], isClosed=False,
                  color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    return overlay


# ── MAIN ──────────────────────────────────────────────────────────────────────
if not uploaded:
    st.info("Upload a satellite image and click Run.")
    st.stop()

# Load image immediately on upload (before Run)
raw = np.frombuffer(uploaded.read(), np.uint8)
bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# ── STEP 1: Show original image always ────────────────────────────────────────
st.subheader("Step 1 — Input Image")
st.caption("This is the satellite image that will be analysed.")
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(rgb); ax.axis('off')
st.pyplot(fig); plt.close()

if not run:
    st.info("Click **Run** in the sidebar to start the analysis.")
    st.stop()

# ── STEP 2: Preprocess ────────────────────────────────────────────────────────
st.subheader("Step 2 — Preprocessing")
st.caption("Resize to 800×400, remove noise, convert to HSV color space for better color detection.")

bgr_resized = cv2.resize(bgr, (800, 400))
bgr_clean   = cv2.fastNlMeansDenoisingColored(bgr_resized, None, 10, 10, 7, 21)
hsv         = cv2.cvtColor(bgr_clean, cv2.COLOR_BGR2HSV)
rgb_clean   = cv2.cvtColor(bgr_clean, cv2.COLOR_BGR2RGB)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(rgb_clean); ax.set_title("Cleaned Image"); ax.axis('off')
    st.pyplot(fig); plt.close()
with col2:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(hsv); ax.set_title("HSV Color Space"); ax.axis('off')
    st.pyplot(fig); plt.close()

# ── STEP 3: Terrain Classification ───────────────────────────────────────────
st.subheader("Step 3 — Terrain Classification")
st.caption("KMeans groups similar pixels together. Each group is labelled as a terrain type with a cost.")

with st.spinner("Classifying terrain..."):
    grid, colors = build_grid(hsv, k)

# Show how many pixels are water for debugging
water_pct = (grid >= 9999).sum() / grid.size * 100
st.caption(f"Water pixels detected: {water_pct:.1f}% of image")

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(colors); ax.axis('off'); ax.set_title("Terrain Map")
st.pyplot(fig); plt.close()

# Legend
cols_leg = st.columns(len(TERRAIN))
for col, (name, val) in zip(cols_leg, TERRAIN.items()):
    r, g2, b = val["color"]
    hex_color = f"#{r:02x}{g2:02x}{b:02x}"
    col.markdown(
        f"<div style='background:{hex_color};padding:6px;border-radius:4px;"
        f"text-align:center;color:white;font-size:12px'>"
        f"<b>{name}</b><br>cost={val['cost']}</div>",
        unsafe_allow_html=True
    )

# ── STEP 4: Auto Start/End ────────────────────────────────────────────────────
st.subheader("Step 4 — Select Start & End Points")
st.caption("Automatically picks top-left and bottom-right corners, snapping away from water.")

start, end = find_start_end(grid)
st.success(f"Start: {start}   End: {end}")

# ── STEP 5: A* Pathfinding ────────────────────────────────────────────────────
st.subheader("Step 5 — A* Pathfinding")
st.caption("A* finds the lowest-cost route across the terrain grid, treating water as impassable walls.")

with st.spinner("Finding optimal path..."):
    path = astar(grid, start, end)

if not path:
    st.error("No path found. The start and end points may be completely surrounded by water. "
             "Try adjusting the Terrain Clusters slider or use a different image.")
    st.stop()

st.success(f"Path found: {len(path)} steps")

# Draw thick visible path
overlay = draw_path_thick(colors, path, thickness=3)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(overlay); ax.axis('off')
    ax.scatter([start[1], end[1]], [start[0], end[0]],
               c=['lime', 'red'], s=200, zorder=5, edgecolors='white', lw=2)
    ax.set_title("Path on Terrain Map (white line)")
    st.pyplot(fig); plt.close()

with col2:
    rows_p, cols_p = zip(*path)
    fig, ax = plt.subplots(figsize=(8, 4))
    # Cap cost grid display at 100 so water (9999) doesn't wash out colors
    display_grid = np.clip(grid, 0, 100)
    ax.imshow(display_grid, cmap='RdYlGn_r', vmin=0, vmax=100, alpha=0.85)
    ax.plot(cols_p, rows_p, color='dodgerblue', lw=2)
    ax.scatter([start[1], end[1]], [start[0], end[0]],
               c=['lime', 'red'], s=200, zorder=5, edgecolors='white', lw=2)
    ax.axis('off'); ax.set_title("Path on Cost Grid")
    st.pyplot(fig); plt.close()

# Terrain breakdown along path
cost_to_name = {v["cost"]: k for k, v in TERRAIN.items()}
counts = {}
for r, c in path:
    name = cost_to_name.get(int(grid[r, c]), "unknown")
    counts[name] = counts.get(name, 0) + 1

total_cost = sum(grid[r, c] for r, c in path)

col1, col2 = st.columns(2)
with col1:
    labels = list(counts.keys())
    sizes  = list(counts.values())
    clrs   = [np.array(TERRAIN[n]["color"]) / 255 if n in TERRAIN else [0.5, 0.5, 0.5]
              for n in labels]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sizes, labels=labels, colors=clrs, autopct='%1.1f%%', startangle=90)
    ax.set_title("Terrain Along Path")
    st.pyplot(fig); plt.close()

with col2:
    st.markdown("### Cost Breakdown")
    for name, steps in sorted(counts.items(), key=lambda x: -x[1]):
        pct      = steps / len(path) * 100
        cost     = TERRAIN.get(name, {}).get("cost", 0)
        subtotal = steps * cost
        r2, g2, b2 = TERRAIN.get(name, {}).get("color", [100, 100, 100])
        hex_col  = f"#{r2:02x}{g2:02x}{b2:02x}"
        st.markdown(
            f"<div style='background:{hex_col};padding:8px 12px;border-radius:6px;"
            f"margin-bottom:6px;color:white'>"
            f"<b>{name}</b> — {steps} steps ({pct:.1f}%) × cost {cost} = <b>{subtotal}</b></div>",
            unsafe_allow_html=True
        )

# ── FINAL OUTPUT ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Final Output")
col1, col2, col3 = st.columns(3)
col1.metric("Path Length", f"{len(path)} steps")
col2.metric("Total Cost",  f"{total_cost:.0f}")
col3.metric("Terrains Crossed", len(counts))

fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(overlay)
ax.scatter([start[1], end[1]], [start[0], end[0]],
           c=['lime', 'red'], s=250, zorder=5, edgecolors='white', lw=2)
ax.set_title(f"Optimal Path | {len(path)} steps | Total Cost: {total_cost:.0f}", fontsize=14)
ax.axis('off')
st.pyplot(fig); plt.close()