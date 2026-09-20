import unittest

import cv2
import numpy as np

import cv_core as core


class MarkerLockTest(unittest.TestCase):
    corner_ids = {"TL": 10, "TR": 11, "BR": 12, "BL": 13}

    @staticmethod
    def marker(marker_id, outer_corner, point):
        points = np.zeros((1, 4, 2), dtype=np.float32)
        points[0, outer_corner] = point
        return marker_id, points

    def marker_data(self, shift=(0, 0), include=(10, 11, 12)):
        dx, dy = shift
        specs = {
            10: self.marker(10, 0, (0 + dx, 0 + dy)),
            11: self.marker(11, 1, (1279 + dx, 0 + dy)),
            12: self.marker(12, 2, (1279 + dx, 719 + dy)),
            13: self.marker(13, 3, (0 + dx, 719 + dy)),
        }
        selected = [specs[marker_id] for marker_id in include]
        ids = np.asarray([[marker_id] for marker_id, _ in selected], dtype=np.int32)
        corners = [points for _, points in selected]
        return corners, ids

    def test_three_matching_markers_hold_saved_homography(self):
        corners, ids = self.marker_data(shift=(2, -1))
        self.assertTrue(
            core.markers_agree_with_homography(
                corners,
                ids,
                np.eye(3, dtype=np.float32),
                1280,
                720,
                self.corner_ids,
            )
        )

    def test_camera_shift_does_not_hold_saved_homography(self):
        corners, ids = self.marker_data(shift=(15, 0))
        self.assertFalse(
            core.markers_agree_with_homography(
                corners,
                ids,
                np.eye(3, dtype=np.float32),
                1280,
                720,
                self.corner_ids,
            )
        )

    def test_two_markers_are_not_enough_to_hold_lock(self):
        corners, ids = self.marker_data(include=(11, 13))
        self.assertFalse(
            core.markers_agree_with_homography(
                corners,
                ids,
                np.eye(3, dtype=np.float32),
                1280,
                720,
                self.corner_ids,
            )
        )

    def test_perspective_transform_is_respected(self):
        source = np.asarray(
            [[20, 30], [1260, 15], [1250, 700], [25, 710]], dtype=np.float32
        )
        destination = np.asarray(
            [[0, 0], [1279, 0], [1279, 719], [0, 719]], dtype=np.float32
        )
        homography = cv2.getPerspectiveTransform(source, destination)
        corners = []
        ids = []
        for marker_id, outer_corner, point in (
            (10, 0, source[0]),
            (11, 1, source[1]),
            (13, 3, source[3]),
        ):
            _, marker_corners = self.marker(marker_id, outer_corner, point)
            corners.append(marker_corners)
            ids.append([marker_id])
        self.assertTrue(
            core.markers_agree_with_homography(
                corners,
                np.asarray(ids, dtype=np.int32),
                homography,
                1280,
                720,
                self.corner_ids,
            )
        )


if __name__ == "__main__":
    unittest.main()
