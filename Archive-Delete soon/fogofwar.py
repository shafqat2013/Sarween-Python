import numpy as np
import cv2
from shapely.geometry import (
    Polygon, MultiPolygon, GeometryCollection,
    Point, MultiPoint, box, LineString
)
from shapely.ops import unary_union

# ————————— Configuration —————————
GRID_COLS, GRID_ROWS = 40, 40
OBSTACLES = ['q2', 'v10', 'x15', 'r8',
            'c30','d30','e30','f30','g30',
            'c38','d38','e38','f38','g38',
            'c31','c32','c33','c34','c35','c36','c37',
            'g31','g32','g33','g35','g36','g37'] # reorderable without effect
CELL_SIZE = 40  # pixels

# Toggles
USE_LINE_SEGMENT   = True   # v2: treat each light as a line segment
GENERATE_CORNERS   = False  # default: single FOW image
SHOW_PENUMBRA      = True   # draw penumbra regions
SHOW_UMBRA         = True   # draw umbra regions
SHOW_CELL_LABELS   = True  # draw grid cell labels

# ————————— Helpers —————————
def index_to_column_label(n: int) -> str:
    label = ""
    while n >= 0:
        n, rem = divmod(n, 26)
        label = chr(65 + rem) + label
        n -= 1
    return label

def coord_to_indices(coord: str) -> tuple[int,int]:
    c, r = coord[0], coord[1:]
    return ord(c.upper()) - ord('A'), int(r) - 1

# Build infinite shadow wedge, then clip to world

def compute_shadow_poly(origin: tuple[float,float],
                        obs: Polygon,
                        boundary: tuple[float,float,float,float],
                        return_data: bool = False):
    ox, oy = origin
    world = box(*boundary)
    corners = list(obs.exterior.coords)[:-1]
    angs = [np.arctan2(y-oy, x-ox) % (2*np.pi) for x,y in corners]
    i0, i1 = int(np.argmin(angs)), int(np.argmax(angs))
    sil = [corners[i0], corners[i1]]
    maxd = (boundary[2]**2 + boundary[3]**2) * 10
    extents = []
    for sx, sy in sil:
        dx, dy = sx - ox, sy - oy
        if dx == 0 and dy == 0:
            extents.append((ox, oy))
        else:
            extents.append((ox + dx * maxd, oy + dy * maxd))
    if len(extents) == 2:
        wedge = Polygon([sil[0], extents[0], extents[1], sil[1]])
        poly = wedge.intersection(world)
    else:
        poly = Polygon()
    if return_data:
        return poly, sil, extents
    return poly

# Flatten shapely geometries
def _extract_polygons(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []

# ————————— Fog-of-War Renderer —————————
def fog_of_war(light_coords,
               cell_size: int = CELL_SIZE,
               show_penumbra: bool = SHOW_PENUMBRA,
               show_umbra:  bool = SHOW_UMBRA,
               show_labels: bool = SHOW_CELL_LABELS) -> np.ndarray:
    if isinstance(light_coords, str):
        light_coords = [light_coords]
    cols, rows = GRID_COLS, GRID_ROWS
    boundary = (0.0, 0.0, float(cols), float(rows))

    # Precompute obstacle geometries and centers
    obs_coords = [coord_to_indices(o) for o in OBSTACLES]
    obstacles = [box(x, y, x+1, y+1) for x, y in obs_coords]
    obs_centers = [((x+0.5), (y+0.5)) for x, y in obs_coords]

    per_light_unions, per_light_umbras = [], []
    for lc_str in light_coords:
        lc, lr = coord_to_indices(lc_str)
        light_center = (lc + 0.5, lr + 0.5)

        # determine sample origins
        if USE_LINE_SEGMENT:
            # pick obstacle closest to light
            dists = [((cx - light_center[0])**2 + (cy - light_center[1])**2)
                     for cx, cy in obs_centers]
            idx_closest = int(np.argmin(dists))
            ox0, oy0 = obs_coords[idx_closest]
            dx, dy = ox0 - lc, oy0 - lr
            # choose edge opposite closest obstacle
            if abs(dx) > abs(dy):
                # vertical edge
                if dx > 0:
                    origins = [(lc, lr), (lc, lr+1)]
                else:
                    origins = [(lc+1, lr), (lc+1, lr+1)]
            else:
                # horizontal edge
                if dy > 0:
                    origins = [(lc, lr), (lc+1, lr)]
                else:
                    origins = [(lc, lr+1), (lc+1, lr+1)]
        else:
            origins = [(lc + dx, lr + dy) for dx in (0,1) for dy in (0,1)]

        # compute shadows from each origin
        corner_shadows = []
        for origin in origins:
            polys = [compute_shadow_poly(origin, obs, boundary) for obs in obstacles]
            corner_shadows.append(unary_union(polys))

        # umbra per light
        umbra_i = corner_shadows[0]
        for s in corner_shadows[1:]:
            umbra_i = umbra_i.intersection(s)
        # shadow union per light
        union_i = unary_union(corner_shadows)
        per_light_umbras.append(umbra_i)
        per_light_unions.append(union_i)

    # global umbra & penumbra
    global_umbra = per_light_umbras[0]
    for u in per_light_umbras[1:]:
        global_umbra = global_umbra.intersection(u)
    global_union = unary_union(per_light_unions)
    global_penumbra = global_union.difference(global_umbra)

    # draw grid
    H, W = rows * cell_size, cols * cell_size
    img = np.full((H, W, 3), 255, np.uint8)
    for c in range(cols+1): cv2.line(img, (c*cell_size,0), (c*cell_size,H), (0,0,0),1)
    for r in range(rows+1): cv2.line(img, (0,r*cell_size), (W,r*cell_size), (0,0,0),1)

    # render shadows
    if show_penumbra:
        for poly in _extract_polygons(global_penumbra):
            pts = np.array([[int(x*cell_size), int(y*cell_size)] for x,y in poly.exterior.coords], np.int32)
            cv2.fillPoly(img, [pts], (128,128,128))
    if show_umbra:
        for poly in _extract_polygons(global_umbra):
            pts = np.array([[int(x*cell_size), int(y*cell_size)] for x,y in poly.exterior.coords], np.int32)
            cv2.fillPoly(img, [pts], (0,0,0))

    # draw lights & obstacles
    for lc_str in light_coords:
        lc, lr = coord_to_indices(lc_str)
        cv2.rectangle(img, (lc*cell_size, lr*cell_size), ((lc+1)*cell_size,(lr+1)*cell_size), (0,255,255),-1)
    for x, y in obs_coords:
        cv2.rectangle(img, (x*cell_size, y*cell_size), ((x+1)*cell_size,(y+1)*cell_size), (0,0,255),-1)

    # optional cell labels
    if show_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        for c in range(cols):
            label = index_to_column_label(c)
            for r in range(rows):
                cv2.putText(img, f"{label}{r+1}", (c*cell_size+4, r*cell_size+int(cell_size*0.3)), font, 0.3, (255,0,0),1)

    return img

# ————————— Main Entry Point —————————
if __name__ == "__main__":
    light = ["r3"]
    if not GENERATE_CORNERS:
        img = fog_of_war(light)
        cv2.imwrite("fow_v2.png", img)
        cv2.imshow("Fog of War V2", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        letters = ['1','2'] if USE_LINE_SEGMENT else ['A','B','C','D']
        for idx, letter in enumerate(letters):
            img = fog_of_war(light)
            # label origins
            lc, lr = coord_to_indices(light[0])
            origins = [(lc, lr), (lc, lr+1)] if USE_LINE_SEGMENT else [(lc+dx, lr+dy) for dx in (0,1) for dy in (0,1)]
            for i,(cx,cy) in enumerate(origins, start=1):
                cv2.putText(img, str(i), (int(cx*CELL_SIZE)+5,int(cy*CELL_SIZE)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
            # draw rays
            for origin in origins:
                _, sil, extents = compute_shadow_poly(origin, box(*obs_coords[0], obs_coords[0][0]+1, obs_coords[0][1]+1), (0,0,float(GRID_COLS),float(GRID_ROWS)), True)
                ox, oy = origin
                oxp, oyp = int(ox*CELL_SIZE), int(oy*CELL_SIZE)
                for ex, ey in extents:
                    cv2.line(img, (oxp,oyp), (int(ex*CELL_SIZE),int(ey*CELL_SIZE)), (0,0,255),2)
            cv2.imwrite(f"fow_v2_corner_{letter}.png", img)
        print("Generated fow_v2.png or fow_v2_corner_*.png")
