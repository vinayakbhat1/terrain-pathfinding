import numpy as np
import heapq
import cv2


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
        return (r, c)

    return nearest_passable(10, 10), nearest_passable(H - 10, W - 10)


def astar(grid, start, end):
    """
    A* pathfinding. Water (cost >= 9999) = impassable.
    Heuristic scaled by MIN_COST for admissibility.
    """
    H, W     = grid.shape
    INF      = float('inf')
    MIN_COST = 5

    def heuristic(a, b):
        return MIN_COST * np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    heap = [(0.0, start)]
    came = {}
    g    = {start: 0.0}

    dirs = [(-1,0),(1,0),(0,-1),(0,1),
            (-1,-1),(-1,1),(1,-1),(1,1)]

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
            nb = (cur[0]+dr, cur[1]+dc)
            if not (0 <= nb[0] < H and 0 <= nb[1] < W):
                continue
            nb_cost = grid[nb[0], nb[1]]
            if nb_cost >= 9999:
                continue
            move_cost = nb_cost * (1.414 if dr and dc else 1.0)
            tg = g[cur] + move_cost
            if tg < g.get(nb, INF):
                came[nb] = cur
                g[nb]    = tg
                heapq.heappush(heap, (tg + heuristic(nb, end), nb))

    return []


def draw_path_thick(image, path, thickness=3):
    """Draws path as a thick white line on the image."""
    overlay = image.copy()
    if len(path) < 2:
        return overlay
    pts = np.array([[c, r] for r, c in path], dtype=np.int32)
    cv2.polylines(overlay, [pts], isClosed=False,
                  color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    return overlay