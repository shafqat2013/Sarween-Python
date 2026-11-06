"""Conservative circular-base geometry around an already accepted color blob.

This helper does not identify minis. It refines an existing color detection only
when a local circular boundary has strong radial edge support and a clear lead
over alternative centers. Uncertain images retain the caller's original point.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class BaseCircle:
    cx: float
    cy: float
    radius: float
    confidence: float
    angular_coverage: float
    support: int


def estimate_base_circle(image, center, bbox, grid_px, *, color_mask=None, excluded_mask=None):
    """Return a well-supported local base circle, or ``None`` to keep the blob.

    Coordinates and grid spacing are in the supplied full-resolution image.
    A caption overlapping the search region makes its pixels unavailable; no
    image values are removed, filled or reconstructed.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3):
        return None
    try:
        point = np.asarray(center, dtype=float)
        bounds = np.asarray(bbox, dtype=float)
        grid = float(grid_px)
    except (TypeError, ValueError):
        return None
    if (point.shape != (2,) or bounds.shape != (4,) or not np.all(np.isfinite(point))
            or not np.all(np.isfinite(bounds)) or not math.isfinite(grid) or grid < 6
            or min(bounds[2:]) <= 0):
        return None
    height, width = image.shape[:2]
    if (not isinstance(color_mask, np.ndarray) or color_mask.dtype != np.uint8
            or color_mask.ndim != 2 or color_mask.shape != (height, width)):
        return None
    if not (0 <= point[0] < width and 0 <= point[1] < height):
        return None
    if excluded_mask is not None and (not isinstance(excluded_mask, np.ndarray)
            or excluded_mask.ndim != 2 or excluded_mask.shape != (height, width)):
        return None

    x0, y0 = max(0, int(point[0] - grid * 1.6)), max(0, int(point[1] - grid * 1.6))
    x1 = min(width, int(point[0] + grid * 1.6 + 1))
    y1 = min(height, int(point[1] + grid * 1.6 + 1))
    if min(x1 - x0, y1 - y0) < 8:
        return None
    local_color = np.where(color_mask[y0:y1, x0:x1] > 0, 255, 0).astype(np.uint8)
    # Count the whole associated component, including projecting parts outside
    # the edge-search crop. Pixels outside this crop contribute no ownership.
    total_color = cv2.countNonZero(color_mask)
    color_y, color_x = np.where(local_color > 0)
    if total_color < 15 or len(color_x) < 15:
        return None
    color_points = np.column_stack((color_x + x0, color_y + y0)).astype(float)
    gray = cv2.GaussianBlur(cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, 25, 65)
    if excluded_mask is not None:
        invalid = np.where(excluded_mask[y0:y1, x0:x1] > 0, 255, 0).astype(np.uint8)
        invalid = cv2.dilate(invalid, np.ones((5, 5), np.uint8))
        edges[invalid > 0] = 0
    ys, xs = np.where(edges > 0)
    if len(xs) < 22:
        return None
    xy = np.column_stack((xs + x0, ys + y0)).astype(float)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gradient = np.column_stack((gx[ys, xs], gy[ys, xs]))
    gradient /= np.maximum(np.linalg.norm(gradient, axis=1, keepdims=True), 1e-6)

    # Fixed seed makes replay deterministic. Vectorized trials avoid a Python
    # loop over every possible circle on every tracking frame.
    triples = xy[np.random.default_rng(0).integers(0, len(xy), (512, 3))]
    a = triples[:, 0]
    b, c = triples[:, 1] - a, triples[:, 2] - a
    determinant = 2 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    usable = np.abs(determinant) > 1e-5
    a, b, c, determinant = a[usable], b[usable], c[usable], determinant[usable]
    bb, cc = np.sum(b * b, axis=1), np.sum(c * c, axis=1)
    offset = np.column_stack(((bb * c[:, 1] - cc * b[:, 1]) / determinant,
                              (b[:, 0] * cc - c[:, 0] * bb) / determinant))
    centers = a + offset
    radii = np.linalg.norm(offset, axis=1)
    shifts = np.linalg.norm(centers - point, axis=1)
    usable = ((radii >= grid * .45) & (radii <= grid) & (shifts <= radii * .8)
              & (centers[:, 0] >= bounds[0]) & (centers[:, 0] <= bounds[0] + bounds[2] - 1)
              & (centers[:, 1] >= bounds[1]) & (centers[:, 1] <= bounds[1] + bounds[3] - 1))
    centers, radii = centers[usable], radii[usable]
    if not len(centers):
        return None
    # A nearby map circle cannot own only a projecting arm/tail. Most of the
    # already accepted colored component must belong inside the proposed base.
    color_distance_squared = np.sum((color_points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    associated_mass = np.sum(color_distance_squared <= (radii[None, :] + 1) ** 2, axis=0) / total_color
    associated = associated_mass >= .60
    centers, radii = centers[associated], radii[associated]
    if not len(centers):
        return None

    vectors = xy[:, None, :] - centers[None, :, :]
    distance = np.maximum(np.linalg.norm(vectors, axis=2), 1e-6)
    radial_agreement = np.abs(np.sum(vectors / distance[:, :, None] * gradient[:, None, :], axis=2))
    inliers = (np.abs(distance - radii[None, :]) <= 1.5) & (radial_agreement >= .7)
    support = np.sum(inliers, axis=0)
    bins = (np.floor((np.arctan2(vectors[:, :, 1], vectors[:, :, 0]) + np.pi)
                     * 36 / (2 * np.pi)).astype(int) % 36)
    ids = bins + np.arange(len(centers))[None, :] * 36
    histogram = np.bincount(ids[inliers], minlength=len(centers) * 36).reshape(-1, 36)
    coverage = np.sum(histogram > 0, axis=1) / 36
    density = np.minimum(1, support / (2 * np.pi * radii))
    scores = .6 * coverage + .4 * density
    scores[support < 22] = 0
    # The supported boundary itself must also be associated with that color.
    # Mass containment alone would accept a large gray circle surrounding the
    # miniature. Allow a small gap for reflected/blurred base rims, but require
    # both a useful fraction and an arc of color-associated radial edge support.
    color_distance = cv2.distanceTransform(cv2.bitwise_not(local_color), cv2.DIST_L2, 3)
    near_color = color_distance[ys, xs] <= max(2.0, grid * .20)
    colored_inliers = inliers & near_color[:, None]
    colored_fraction = np.sum(colored_inliers, axis=0) / np.maximum(support, 1)
    colored_histogram = np.bincount(ids[colored_inliers], minlength=len(centers) * 36).reshape(-1, 36)
    colored_bins = np.sum(colored_histogram > 0, axis=1)
    scores[(colored_fraction < .35) | (colored_bins < 8)] = 0
    best = int(np.argmax(scores))
    if scores[best] < .62 or coverage[best] < .6:
        return None
    distinct = np.linalg.norm(centers - centers[best], axis=1) > grid * .15
    competitor = float(np.max(scores[distinct])) if np.any(distinct) else 0.0
    if scores[best] - competitor <= .08:
        return None
    return BaseCircle(float(centers[best, 0]), float(centers[best, 1]), float(radii[best]),
                      float(scores[best]), float(coverage[best]), int(support[best]))
