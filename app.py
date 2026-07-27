import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

from phase3_terrain import TERRAIN, build_grid
from phase4_pathfinding import find_start_end, astar, draw_path_thick

st.set_page_config(page_title="Path Planner", layout="wide")
st.title("Optimal Path Planner")

#  SIDEBAR 
uploaded = st.sidebar.file_uploader("Upload Satellite Image", type=["png","jpg","jpeg"])
k        = st.sidebar.slider("Terrain Clusters", 5, 10, 7)
run      = st.sidebar.button("Run", use_container_width=True)

#  MAIN 
if not uploaded:
    st.info("Upload a satellite image and click Run.")
    st.stop()

raw = np.frombuffer(uploaded.read(), np.uint8)
bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

#  STEP 1: Input Image 
st.subheader("Step 1 — Input Image")
st.caption("This is the satellite image that will be analysed.")
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(rgb); ax.axis('off')
st.pyplot(fig); plt.close()

if not run:
    st.info("Click **Run** in the sidebar to start the analysis.")
    st.stop()

#  STEP 2: Preprocessing 
st.subheader("Step 2 — Preprocessing")
st.caption("Resize to 800×400, remove noise, convert to HSV.")

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

#  STEP 3: Terrain Classification 
st.subheader("Step 3 — Terrain Classification")
st.caption("KMeans groups similar pixels. Each cluster is labelled as a terrain type with a cost.")

with st.spinner("Classifying terrain..."):
    grid, colors = build_grid(rgb_clean, k)

water_pct = (grid >= 1000).sum() / grid.size * 100
st.caption(f"Water pixels detected: {water_pct:.1f}% of image")

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(colors); ax.axis('off'); ax.set_title("Terrain Map")
st.pyplot(fig); plt.close()

# Legend
cols_leg = st.columns(len(TERRAIN))
for col, (name, val) in zip(cols_leg, TERRAIN.items()):
    r, g2, b = val["color"]
    hex_color  = f"#{r:02x}{g2:02x}{b:02x}"
    text_color = "black" if name == "snow" else "white"
    col.markdown(
        f"<div style='background:{hex_color};padding:6px;border-radius:4px;"
        f"text-align:center;color:{text_color};font-size:12px'>"
        f"<b>{name}</b><br>cost={val['cost']}</div>",
        unsafe_allow_html=True
    )

#  STEP 4: Start & End 
st.subheader("Step 4 — Select Start & End Points")
st.caption("Auto-picks top-left and bottom-right corners, snapping away from water.")

start, end = find_start_end(grid)
st.success(f"Start: {start}   End: {end}")

#  STEP 5: A* Pathfinding 
st.subheader("Step 5 — A* Pathfinding")
st.caption("A* finds the lowest-cost route, treating water as impassable.")

with st.spinner("Finding optimal path..."):
    path = astar(grid, start, end)

if not path:
    st.error("No path found. Try adjusting the Terrain Clusters slider or use a different image.")
    st.stop()

st.success(f"Path found: {len(path)} steps")

overlay = draw_path_thick(colors, path, thickness=3)

# Display the Path Overlay
fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(overlay); ax.axis('off')
ax.scatter([start[1], end[1]], [start[0], end[0]],
           c=['lime', 'red'], s=200, zorder=5, edgecolors='white', lw=2)
ax.set_title("Path on Terrain Map (white line)")
st.pyplot(fig); plt.close()

#  Final Output Metrics Calculation 
cost_to_name = {v["cost"]: k for k, v in TERRAIN.items()}
counts = {}
for r, c in path:
    name = cost_to_name.get(int(grid[r, c]), "unknown")
    counts[name] = counts.get(name, 0) + 1

total_cost = sum(grid[r, c] for r, c in path)

#  Final Output 
st.divider()
st.subheader("Final Output")
col1, col2, col3 = st.columns(3)
col1.metric("Path Length",      f"{len(path)} steps")
col2.metric("Total Cost",       f"{total_cost:.0f}")
col3.metric("Terrains Crossed", len(counts))

fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(overlay)
ax.scatter([start[1], end[1]], [start[0], end[0]],
           c=['lime', 'red'], s=250, zorder=5, edgecolors='white', lw=2)
ax.set_title(f"Optimal Path | {len(path)} steps | Total Cost: {total_cost:.0f}", fontsize=14)
ax.axis('off')
st.pyplot(fig); plt.close()