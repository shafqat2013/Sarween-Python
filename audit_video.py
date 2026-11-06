"""Saved-video inspection without tracking, camera devices or native windows."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np

from scripts.review_tracking_video import geometry_at, grid_overlay, scene_matrix


class AuditVideo:
    """Read a saved video and independently render its registered board.

    Frames are already the pixels saved by Sarween; no lens correction or
    miniature detector is loaded. ``cell_center`` returns coordinates in the
    most recent display image, for the caller's optional zoom focus.
    """

    def __init__(self, video: Path, timeline: dict | None = None):
        if not isinstance(video, (str, Path)):
            raise ValueError("Choose a saved video file.")
        try:
            path = Path(video).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("The saved video file is unavailable.") from exc
        if not path.is_file():
            raise ValueError("Choose a saved video file.")
        self.path = path
        self.timeline = timeline if isinstance(timeline, dict) else {}
        self._width = self._dimension("warpWidth", 1280)
        self._height = self._dimension("warpHeight", 720)
        mode = self.timeline.get("markerMode")
        self._mode = mode if mode in ("viewport", "legacy") else None
        self._anchors = {}
        self._recorded_registrations = {}
        if isinstance(self.timeline.get("markerRegistration"), dict):
            self._recorded_registrations[0] = self.timeline["markerRegistration"]
        self._printed_registration_error = ""
        self._h = None
        self._held = False
        self._frame = -1
        self._image = None
        self._next_read = 0
        self._closed = False
        self._display_board = True
        self._renders = {}
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
        self._load_checkpoint_anchor()
        # The only capture input is an existing, resolved full file path.
        self._capture = cv2.VideoCapture(str(path))
        try:
            if not self._capture.isOpened():
                raise ValueError("This video could not be opened.")
            fps = float(self._capture.get(cv2.CAP_PROP_FPS))
            count = float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
                raise ValueError("This video has no readable frames or timing.")
            self._fps, self._count = fps, int(count)
            if not math.isfinite(self._count / self._fps):
                raise ValueError("This video has invalid timing.")
            self.seek(0)
        except Exception:
            self.close()
            raise

    def _dimension(self, key, default):
        value = self.timeline.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            value = int(value)
            if 2 <= value <= 4096:
                return value
        return default

    @property
    def fps(self):
        return self._fps

    @property
    def count(self):
        return self._count

    @property
    def duration(self):
        return self._count / self._fps

    @property
    def frame(self):
        return self._frame

    @property
    def image(self):
        return self._image

    def _load_checkpoint_anchor(self):
        checkpoints = self.timeline.get("checkpoints") or []
        if not isinstance(checkpoints, list) or not checkpoints:
            return
        entry = checkpoints[0]
        if not isinstance(entry, dict):
            return
        try:
            relative = Path(entry["path"])
            if relative.is_absolute():
                return
            path = (self.path.parent / relative).resolve(strict=True)
            if not path.is_relative_to(self.path.parent) or not path.is_file() or path.stat().st_size > 100_000_000:
                return
            expected = entry.get("sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
                return
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected.lower():
                return
            with zipfile.ZipFile(path) as archive:
                if sum(item.file_size for item in archive.infolist()) > 200_000_000:
                    return
            with np.load(path, allow_pickle=False) as archive:
                # Deliberately never read tracking state or mini predictions.
                raw = archive["H_saved"]
                if raw.shape != (3, 3) or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
                    return
                homography = np.asarray(raw, dtype=float).copy()
                registration = None
                if "metadata" in archive:
                    metadata = json.loads(archive["metadata"].tobytes())
                    registration = metadata.get("markerRegistration")
            index = entry.get("frame", 0)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                return
            if homography.shape == (3, 3) and np.isfinite(homography).all():
                self._anchors[index] = homography
                if isinstance(registration, dict):
                    self._recorded_registrations[index] = registration
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
            return

    def _registration_at(self, frame):
        result = None
        eligible = [key for key in self._recorded_registrations if key <= frame]
        if eligible:
            best_frame = max(eligible)
            result = self._recorded_registrations[best_frame]
        else:
            best_frame = -1
        for event in self.timeline.get("events", []):
            if not isinstance(event, dict) or not isinstance(event.get("frame"), int) or event["frame"] > frame:
                continue
            registration = (event.get("cameraLock") or {}).get("markerRegistration")
            if isinstance(registration, dict) and event["frame"] >= best_frame:
                result = registration
                best_frame = event["frame"]
        return result

    @staticmethod
    def _registration_identity(registration):
        if not registration or registration.get("mode") != "printed":
            return "screen"
        from printed_markers import layout_fingerprint
        return layout_fingerprint(registration.get("printedMarkerLayout"))

    def _valid_h(self, homography):
        if homography is None or self._image is None:
            return False
        try:
            value = np.asarray(homography, float)
            if value.shape != (3, 3) or not np.isfinite(value).all() or np.linalg.cond(value) > 1e12:
                return False
            inverse = np.linalg.inv(value)
            rectangle = np.float64([[0, 0, 1], [self._width - 1, 0, 1],
                [self._width - 1, self._height - 1, 1], [0, self._height - 1, 1]])
            projected = rectangle @ inverse.T
            divisor = projected[:, 2]
            if np.any(np.abs(divisor) < 1e-10) or not (np.all(divisor > 0) or np.all(divisor < 0)):
                return False
            points = projected[:, :2] / divisor[:, None]
            height, width = self._image.shape[:2]
            if (not np.isfinite(points).all() or np.any(points < -2)
                    or np.any(points[:, 0] > width + 1) or np.any(points[:, 1] > height + 1)):
                return False
            contour = points.astype(np.float32).reshape(4, 1, 2)
            return bool(cv2.isContourConvex(contour) and abs(cv2.contourArea(contour)) >= 16)
        except (ValueError, TypeError, np.linalg.LinAlgError, cv2.error):
            return False

    def _register(self):
        self._h, self._held = None, False
        self._printed_registration_error = ""
        corners, ids, _rejected = self._detector.detectMarkers(cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY))
        markers = {}
        duplicate = set()
        if ids is not None:
            for marker_id, points in zip(ids.ravel(), corners):
                key = int(marker_id)
                if key in markers:
                    duplicate.add(key)
                markers[key] = np.asarray(points, np.float32).reshape(4, 2)
        groups = [("viewport", (10, 11, 12, 13)), ("legacy", (0, 1, 2, 3))]
        registration = self._registration_at(self._frame)
        if registration and registration.get("mode") == "printed":
            groups = []  # Never substitute an on-screen/legacy layout for paper.
            try:
                from printed_markers import solve_homography
                _scene, recorded_view = geometry_at(self.timeline, self._frame)
                homography = solve_homography(corners, ids, registration.get("printedMarkerLayout"),
                                              recorded_view, (self._width, self._height))
                if self._valid_h(homography):
                    self._h = homography
                    self._mode = "viewport"
                    self._anchors[self._frame] = homography.copy()
                    return
            except (ValueError, TypeError, cv2.error) as exc:
                self._printed_registration_error = str(exc)
        if self._mode in {"viewport", "legacy"}:
            groups = [group for group in groups if group[0] == self._mode]
        for mode, required in groups:
            if not all(key in markers and key not in duplicate for key in required):
                continue
            source = np.float32([markers[required[0]][0], markers[required[1]][1],
                                 markers[required[2]][2], markers[required[3]][3]])
            destination = np.float32([[0, 0], [self._width - 1, 0],
                [self._width - 1, self._height - 1], [0, self._height - 1]])
            homography = cv2.getPerspectiveTransform(source, destination)
            if self._valid_h(homography):
                self._h = homography
                self._mode = mode
                self._anchors[self._frame] = homography.copy()
                # Keep the original anchor and a bounded set of recent views.
                if len(self._anchors) > 128:
                    keys = sorted(self._anchors)
                    del self._anchors[keys[1]]
                return
        # Never use a future-frame registration when scrubbing backwards.
        for index in sorted((key for key in self._anchors if key <= self._frame), reverse=True):
            try:
                if self._registration_identity(registration) != self._registration_identity(self._registration_at(index)):
                    continue
            except (ValueError, TypeError):
                continue
            if self._valid_h(self._anchors[index]):
                self._h = self._anchors[index].copy()
                self._held = True
                return

    def seek(self, frame):
        """Read an integer frame, clamped to this saved video's bounds."""
        if self._closed:
            raise ValueError("This video is closed.")
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
            raise ValueError("Choose a whole video frame.")
        index = max(0, min(self._count - 1, int(frame)))
        if index == self._frame:
            return self._image
        if self._next_read != index:
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError(f"Cannot seek to frame {index}.")
        ok, image = self._capture.read()
        self._next_read = None
        position = float(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        if (not ok or not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2
                or not math.isfinite(position) or round(position) != index + 1):
            raise ValueError(f"Cannot read frame {index}.")
        if self._image is not None and self._image.shape != image.shape:
            self._anchors.clear()
        self._frame, self._image, self._next_read = index, image, index + 1
        self._renders.clear()
        self._register()
        return self._image

    def _legacy_grid(self):
        values = [self.timeline.get("gridCols"), self.timeline.get("gridRows")]
        if self._mode != "legacy" or any(isinstance(value, bool) or not isinstance(value, int)
            or not 1 <= value <= 500 for value in values):
            return None
        return values

    def _grid(self, image):
        if self._mode == "viewport":
            scene, view = geometry_at(self.timeline, self._frame)
            if scene.get("gridType", 1) != 1:
                raise ValueError("Square grid geometry is unavailable")
            return grid_overlay(image, scene, view)
        dimensions = self._legacy_grid()
        if dimensions is None:
            raise ValueError("Recorded grid geometry is unavailable")
        columns, rows = dimensions
        result = image.copy()
        for column in range(columns + 1):
            x = min(self._width - 1, round(column * self._width / columns))
            cv2.line(result, (x, 0), (x, self._height - 1), (90, 90, 90), 1)
        for row in range(rows + 1):
            y = min(self._height - 1, round(row * self._height / rows))
            cv2.line(result, (0, y), (self._width - 1, y), (90, 90, 90), 1)
        for row in range(rows):
            for column in range(columns):
                label, number = "", column + 1
                while number:
                    number, remainder = divmod(number - 1, 26)
                    label = chr(65 + remainder) + label
                cv2.putText(result, f"{label}{row + 1}",
                    (int((column + .08) * self._width / columns), int((row + .4) * self._height / rows)),
                    cv2.FONT_HERSHEY_PLAIN, .55, (180, 180, 180), 1)
        return result

    def display(self, board=True, grid=True):
        if self._closed or self._image is None:
            raise ValueError("This video is closed.")
        self._display_board = bool(board and self._h is not None)
        key = (bool(board), bool(grid))
        if key in self._renders:
            image, hint = self._renders[key]
            return image.copy(), hint
        if self._display_board:
            image = cv2.warpPerspective(self._image, self._h, (self._width, self._height))
            hint = "Corners covered · using last board view" if self._held else "Board view"
        else:
            image = self._image.copy()
            hint = "Full camera view"
            if self._printed_registration_error:
                hint += " · printed registration unavailable"
        if grid and self._h is not None:
            try:
                if self._display_board:
                    painted = self._grid(image)
                    image = cv2.addWeighted(image, .65, painted, .35, 0)
                else:
                    layer = self._grid(np.zeros((self._height, self._width, 3), np.uint8))
                    projected = cv2.warpPerspective(layer, np.linalg.inv(self._h),
                                                   (image.shape[1], image.shape[0]))
                    changed = np.any(projected != 0, axis=2)
                    blended = cv2.addWeighted(image, .65, projected, .35, 0)
                    image[changed] = blended[changed]
                    if self._held:
                        hint = "Full camera view · using last board grid"
            except (ValueError, KeyError, TypeError, OverflowError, np.linalg.LinAlgError, cv2.error):
                hint += " · grid unavailable"
        self._renders[key] = (image.copy(), hint)
        return image, hint

    def cell_center(self, cell):
        if self._closed or self._h is None or not isinstance(cell, str):
            return None
        match = re.fullmatch(r"([A-Za-z]{1,4})([1-9][0-9]{0,5})", cell.strip())
        if match is None:
            return None
        column = 0
        for char in match[1].upper():
            column = column * 26 + ord(char) - ord("A") + 1
        column -= 1
        row = int(match[2]) - 1
        try:
            if self._mode == "viewport":
                scene, view = geometry_at(self.timeline, self._frame)
                if scene.get("gridType", 1) != 1:
                    return None
                size = float(view["gridSize"])
                if not math.isfinite(size) or size <= 0:
                    return None
                x = view.get("gridOriginX", 0) + (column + .5) * size
                y = view.get("gridOriginY", 0) + (row + .5) * size
                if not (0 <= x < scene["width"] and 0 <= y < scene["height"]):
                    return None
                point = np.linalg.inv(scene_matrix(view, self._width, self._height)) @ [x, y, 1]
                point = point[:2] / point[2]
            else:
                dimensions = self._legacy_grid()
                if dimensions is None or column >= dimensions[0] or row >= dimensions[1]:
                    return None
                point = np.array([(column + .5) * self._width / dimensions[0],
                                  (row + .5) * self._height / dimensions[1]])
            if not np.isfinite(point).all() or not (0 <= point[0] < self._width and 0 <= point[1] < self._height):
                return None
            if not self._display_board:
                point = cv2.perspectiveTransform(np.float64([[point]]), np.linalg.inv(self._h))[0, 0]
                if not (0 <= point[0] < self._image.shape[1] and 0 <= point[1] < self._image.shape[0]):
                    return None
            return float(point[0]), float(point[1])
        except (ValueError, KeyError, TypeError, OverflowError, np.linalg.LinAlgError, cv2.error):
            return None

    def close(self):
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()
            self._capture = None
        self._closed = True
        self._image = None
        self._renders.clear()
