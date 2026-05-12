# Terrain Classification & A* Pathfinding

An end-to-end satellite image analysis tool that classifies terrain types using computer vision and finds the optimal traversal path using the A* algorithm — deployed as an interactive Streamlit web app.

---

## What it does

Upload any satellite image → the app automatically:
1. Preprocesses the image (denoise, HSV conversion)
2. Clusters pixels using **KMeans** into terrain types
3. Assigns traversal costs to each terrain
4. Finds the **lowest-cost path** from top-left to bottom-right using **A\***
5. Visualizes the path with a full cost breakdown

---

## Terrain Types & Costs

| Terrain | Cost | Color |
|---------|------|-------|
| Urban | 5 | Grey |
| Open Land | 10 | Orange |
| Rocky | 40 | Brown |
| Vegetation | 50 | Light Green |
| Forest | 80 | Dark Green |
| Water | 9999 (impassable) | Blue |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Image Processing | OpenCV, NumPy |
| Machine Learning | Scikit-learn (KMeans) |
| Pathfinding | A* (custom implementation with heapq) |
| Visualization | Matplotlib |
| Web App | Streamlit |

---

## Project Structure

```
terrain-pathfinding/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Dependencies
├── sample_images/                # Sample satellite images to test
│   └── sample.jpg
├── notebooks/
│   ├── phase1_input.ipynb        # Image loading & exploration
│   ├── phase2_preprocessing.ipynb # Denoising, resizing, HSV conversion
│   ├── phase3_terrain.ipynb      # KMeans clustering & terrain classification
│   └── phase4_pathfinding.ipynb  # A* implementation & path visualization
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/vinayakbhat1/terrain-pathfinding.git
cd terrain-pathfinding

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## How it works

### Phase 1 — Input
Load a satellite image (PNG/JPG). The app accepts any resolution and resizes it to 800×400 for consistent processing.

### Phase 2 — Preprocessing
- Resize to 800×400
- Apply Non-Local Means Denoising (`cv2.fastNlMeansDenoisingColored`) to reduce noise
- Convert BGR → HSV color space for more accurate color-based classification

### Phase 3 — Terrain Classification
- Flatten image pixels and run **KMeans clustering** (configurable: 5–10 clusters)
- Each cluster center is classified by its HSV values into one of 6 terrain types
- Each terrain type is assigned a traversal cost

### Phase 4 — A* Pathfinding
- Build a cost grid from the terrain map
- Auto-detect start (top-left) and end (bottom-right) points, snapping away from water
- Run **A\*** with Euclidean distance heuristic (scaled by minimum cost = 5 for admissibility)
- Diagonal movement supported (cost × 1.414)
- Water pixels treated as hard walls (completely impassable)

---

## Sample Output

The app displays:
- Original vs cleaned image
- HSV color space visualization
- Terrain classification map with legend
- Optimal path overlaid on terrain map
- Cost grid heatmap with path
- Pie chart of terrain breakdown along path
- Per-terrain cost breakdown (steps × cost = subtotal)
- Final metrics: path length, total cost, terrains crossed

---

## Requirements

```
streamlit
opencv-python
numpy
scikit-learn
matplotlib
```

---

## Author

**Vinayak Bhat**
[LinkedIn](https://linkedin.com/in/vinayak-bhat) | [GitHub](https://github.com/vinayakbhat1)
